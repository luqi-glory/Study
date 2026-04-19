import os
import re
import math
import copy
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    model_name: str = "sshleifer/tiny-gpt2"  # 可换成 distilgpt2 / Qwen2.5-0.5B 等
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    total_updates: int = 200
    episodes_per_update: int = 16
    ppo_epochs: int = 4
    mini_batch_size: int = 8

    lr: float = 1e-5
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0

    max_new_tokens_step1: int = 20
    max_new_tokens_step2: int = 20
    temperature: float = 1.0
    top_k: int = 50

    value_loss_clip: float = 0.2
    save_dir: str = "./ppo_agent_ckpt"


CFG = Config()


# =========================================================
# Utils
# =========================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_k_sample(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [vocab]
    return: sampled_token_id [1], logprob_of_sample [1]
    """
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits[indices] = values
        probs = F.softmax(filtered_logits, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)

    dist = torch.distributions.Categorical(probs=probs)
    token = dist.sample()
    logprob = dist.log_prob(token)
    return token.unsqueeze(0), logprob.unsqueeze(0)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


# =========================================================
# Agent Environment
# =========================================================

SYSTEM_PROMPT = """You are an agent that can use exactly one tool:

Tool:
add(a,b) -> returns the integer sum of a and b.

Rules:
1. First output exactly one tool call in the format:
Action: add(a,b)
2. Then after receiving the observation, output exactly:
Final: answer

Do not explain anything.
"""

USER_TEMPLATE = "Question: What is {a} + {b}?\n"


def build_prompt_step1(a: int, b: int) -> str:
    return SYSTEM_PROMPT + "\n" + USER_TEMPLATE.format(a=a, b=b)


def build_prompt_step2(a: int, b: int, action_text: str, obs_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n"
        + USER_TEMPLATE.format(a=a, b=b)
        + action_text.strip()
        + "\n"
        + obs_text.strip()
        + "\n"
    )


ACTION_RE = re.compile(r"Action:\s*add\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
FINAL_RE = re.compile(r"Final:\s*(-?\d+)")


def evaluate_agent_trajectory(a: int, b: int, action_text: str, final_text: str) -> Tuple[float, Dict]:
    """
    奖励设计：
    - action 格式合法：+0.2
    - action 参数正确：+0.4
    - final 格式合法：+0.2
    - final 答案正确：+1.0
    - 若 final 正确但没正确调用工具，不额外加工具奖励
    """
    target = a + b
    reward = 0.0
    info = {
        "action_valid": False,
        "action_correct": False,
        "final_valid": False,
        "final_correct": False,
        "target": target,
    }

    m_action = ACTION_RE.search(action_text)
    if m_action:
        info["action_valid"] = True
        reward += 0.2
        x, y = int(m_action.group(1)), int(m_action.group(2))
        if x == a and y == b:
            info["action_correct"] = True
            reward += 0.4

    m_final = FINAL_RE.search(final_text)
    if m_final:
        info["final_valid"] = True
        reward += 0.2
        ans = int(m_final.group(1))
        if ans == target:
            info["final_correct"] = True
            reward += 1.0

    # 小惩罚：避免啰嗦输出
    length_penalty = 0.002 * max(len(action_text) + len(final_text) - 40, 0)
    reward -= length_penalty
    return reward, info


# =========================================================
# Model wrapper: Causal LM + Value Head
# =========================================================

class PolicyWithValueHead(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.lm.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]      # [B, T, H]
        values = self.value_head(hidden).squeeze(-1)  # [B, T]
        return outputs.logits, values

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        logits, values = self.forward(input_ids, attention_mask)
        # 取最后位置的 next-token distribution 和 state value
        last_logits = logits[:, -1, :]   # [B, V]
        last_values = values[:, -1]      # [B]
        return last_logits, last_values

    def save_pretrained(self, save_dir: str, tokenizer):
        os.makedirs(save_dir, exist_ok=True)
        self.lm.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        torch.save(self.value_head.state_dict(), os.path.join(save_dir, "value_head.pt"))


# =========================================================
# Generation with trajectory recording
# =========================================================

@torch.no_grad()
def generate_segment(
    policy: PolicyWithValueHead,
    ref_model: AutoModelForCausalLM,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: str,
    temperature: float,
    top_k: int,
) -> Dict:
    """
    逐 token 采样，并记录 old_logprob / ref_logprob / old_value
    """
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]

    gen_tokens = []
    old_logprobs = []
    ref_logprobs = []
    old_values = []

    for _ in range(max_new_tokens):
        last_logits, last_value = policy.step(input_ids, attention_mask)
        token, logprob = top_k_sample(last_logits[0], top_k=top_k, temperature=temperature)

        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
        ref_last_logits = ref_outputs.logits[:, -1, :]
        ref_logp_all = F.log_softmax(ref_last_logits, dim=-1)
        ref_logprob = ref_logp_all[0, token.item()].unsqueeze(0)

        gen_tokens.append(token.item())
        old_logprobs.append(logprob.item())
        ref_logprobs.append(ref_logprob.item())
        old_values.append(last_value.item())

        input_ids = torch.cat([input_ids, token.view(1, 1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1
        )

        if token.item() == tokenizer.eos_token_id:
            break

    full_ids = input_ids[0].tolist()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return {
        "prompt_text": prompt_text,
        "prompt_len": prompt_len,
        "full_ids": full_ids,
        "response_ids": gen_tokens,
        "response_text": gen_text,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "old_values": old_values,
    }


# =========================================================
# Recompute current logprobs / values on sampled trajectory
# =========================================================

def recompute_logprobs_and_values(
    policy: PolicyWithValueHead,
    full_input_ids: torch.Tensor,
    prompt_len: int,
    response_len: int,
):
    """
    full_input_ids: [B, T]
    对 response token 的 action logprob 和 value 重新前向计算
    """
    attention_mask = (full_input_ids != -100).long()
    safe_input_ids = full_input_ids.clone()
    safe_input_ids[safe_input_ids == -100] = 0

    logits, values = policy(safe_input_ids, attention_mask=attention_mask)
    logp_all = F.log_softmax(logits, dim=-1)

    B, T = safe_input_ids.shape

    action_logprobs = []
    action_values = []
    action_entropy = []

    for i in range(B):
        lp = []
        vv = []
        ent = []
        for t in range(prompt_len - 1, prompt_len - 1 + response_len):
            # 在位置 t 上预测 token t+1
            target_token = safe_input_ids[i, t + 1]
            lp.append(logp_all[i, t, target_token])
            vv.append(values[i, t])
            probs = torch.exp(logp_all[i, t])
            ent.append(-(probs * logp_all[i, t]).sum())
        action_logprobs.append(torch.stack(lp))
        action_values.append(torch.stack(vv))
        action_entropy.append(torch.stack(ent))

    return (
        torch.stack(action_logprobs, dim=0),   # [B, R]
        torch.stack(action_values, dim=0),     # [B, R]
        torch.stack(action_entropy, dim=0),    # [B, R]
    )


# =========================================================
# Batch building
# =========================================================

def pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(x) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def build_training_batch(episodes: List[Dict], tokenizer):
    """
    这里为了简化，假设同一个 update 内 response 长度可能不同。
    我们按每个 episode 单独训练会太慢，因此把长度统一 pad。
    PPO mask 只算真实 response token。
    """
    pad_id = tokenizer.pad_token_id
    full_ids = [ep["full_ids"] for ep in episodes]
    input_ids = pad_sequences(full_ids, pad_id)

    max_resp = max(len(ep["response_ids"]) for ep in episodes)
    batch_size = len(episodes)

    old_logprobs = torch.zeros(batch_size, max_resp, dtype=torch.float32)
    ref_logprobs = torch.zeros(batch_size, max_resp, dtype=torch.float32)
    old_values = torch.zeros(batch_size, max_resp, dtype=torch.float32)
    returns = torch.zeros(batch_size, max_resp, dtype=torch.float32)
    advantages = torch.zeros(batch_size, max_resp, dtype=torch.float32)
    resp_mask = torch.zeros(batch_size, max_resp, dtype=torch.float32)

    prompt_lens = []
    response_lens = []

    for i, ep in enumerate(episodes):
        rlen = len(ep["response_ids"])
        prompt_lens.append(ep["prompt_len"])
        response_lens.append(rlen)

        old_logprobs[i, :rlen] = torch.tensor(ep["old_logprobs"], dtype=torch.float32)
        ref_logprobs[i, :rlen] = torch.tensor(ep["ref_logprobs"], dtype=torch.float32)
        old_values[i, :rlen] = torch.tensor(ep["old_values"], dtype=torch.float32)

        # 简化版 returns: 全 token 共享同一个 episode reward
        episode_return = ep["reward"]
        returns[i, :rlen] = episode_return

        advantages[i, :rlen] = episode_return - old_values[i, :rlen]
        resp_mask[i, :rlen] = 1.0

    # advantage normalize
    adv_mean = advantages[resp_mask.bool()].mean()
    adv_std = advantages[resp_mask.bool()].std().clamp_min(1e-8)
    advantages = (advantages - adv_mean) / adv_std

    return {
        "input_ids": input_ids,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "old_values": old_values,
        "returns": returns,
        "advantages": advantages,
        "resp_mask": resp_mask,
        "prompt_lens": prompt_lens,
        "response_lens": response_lens,
    }


# =========================================================
# Training
# =========================================================

def collect_episode(policy, ref_model, tokenizer, device: str) -> Dict:
    a = random.randint(0, 20)
    b = random.randint(0, 20)

    # -------- step 1: generate action --------
    prompt1 = build_prompt_step1(a, b)
    seg1 = generate_segment(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompt_text=prompt1,
        max_new_tokens=CFG.max_new_tokens_step1,
        device=device,
        temperature=CFG.temperature,
        top_k=CFG.top_k,
    )

    action_text = seg1["response_text"].strip()
    m_action = ACTION_RE.search(action_text)

    if m_action:
        x, y = int(m_action.group(1)), int(m_action.group(2))
        obs_val = x + y
        obs_text = f"Observation: {obs_val}"
    else:
        obs_text = "Observation: invalid_action"

    # -------- step 2: generate final --------
    prompt2 = build_prompt_step2(a, b, action_text, obs_text)
    seg2 = generate_segment(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompt_text=prompt2,
        max_new_tokens=CFG.max_new_tokens_step2,
        device=device,
        temperature=CFG.temperature,
        top_k=CFG.top_k,
    )

    final_text = seg2["response_text"].strip()

    reward, info = evaluate_agent_trajectory(a, b, action_text, final_text)

    # PPO 更新通常针对实际 sampled trajectory。
    # 这里把 step1 和 step2 拆成两个训练样本，共享同一个 episode reward。
    ep1 = copy.deepcopy(seg1)
    ep1["reward"] = reward
    ep1["meta"] = {
        "a": a, "b": b,
        "action_text": action_text,
        "final_text": final_text,
        **info,
        "phase": "action",
    }

    ep2 = copy.deepcopy(seg2)
    ep2["reward"] = reward
    ep2["meta"] = {
        "a": a, "b": b,
        "action_text": action_text,
        "final_text": final_text,
        **info,
        "phase": "final",
    }

    return {"ep_action": ep1, "ep_final": ep2, "info": info, "reward": reward}


def ppo_update(policy, optimizer, batch, tokenizer, device: str):
    input_ids = batch["input_ids"].to(device)
    old_logprobs = batch["old_logprobs"].to(device)
    ref_logprobs = batch["ref_logprobs"].to(device)
    old_values = batch["old_values"].to(device)
    returns = batch["returns"].to(device)
    advantages = batch["advantages"].to(device)
    resp_mask = batch["resp_mask"].to(device)

    prompt_lens = batch["prompt_lens"]
    response_lens = batch["response_lens"]

    idxs = np.arange(input_ids.size(0))

    stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
    }
    n_steps = 0

    for _ in range(CFG.ppo_epochs):
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), CFG.mini_batch_size):
            mb_idx = idxs[start:start + CFG.mini_batch_size]
            mb_input_ids = input_ids[mb_idx]

            # 逐样本重算，因为每个样本 prompt_len/response_len 不同
            new_logps_list = []
            new_vals_list = []
            ent_list = []

            max_resp = 0
            for i_local, i_global in enumerate(mb_idx):
                full_ids = mb_input_ids[i_local:i_local+1]
                prompt_len = prompt_lens[i_global]
                resp_len = response_lens[i_global]
                max_resp = max(max_resp, resp_len)

                lp, vv, ent = recompute_logprobs_and_values(
                    policy=policy,
                    full_input_ids=full_ids,
                    prompt_len=prompt_len,
                    response_len=resp_len,
                )
                new_logps_list.append(lp[0])
                new_vals_list.append(vv[0])
                ent_list.append(ent[0])

            # pad to same resp length in minibatch
            def pad_tensors_1d(tensors: List[torch.Tensor], pad_value: float = 0.0):
                m = max(t.shape[0] for t in tensors)
                out = torch.full((len(tensors), m), pad_value, device=device)
                mask = torch.zeros((len(tensors), m), device=device)
                for i, t in enumerate(tensors):
                    out[i, :t.shape[0]] = t
                    mask[i, :t.shape[0]] = 1.0
                return out, mask

            new_logprobs, local_mask = pad_tensors_1d(new_logps_list, 0.0)
            new_values, _ = pad_tensors_1d(new_vals_list, 0.0)
            entropy, _ = pad_tensors_1d(ent_list, 0.0)

            mb_old_logprobs = old_logprobs[mb_idx, :new_logprobs.shape[1]]
            mb_ref_logprobs = ref_logprobs[mb_idx, :new_logprobs.shape[1]]
            mb_old_values = old_values[mb_idx, :new_logprobs.shape[1]]
            mb_returns = returns[mb_idx, :new_logprobs.shape[1]]
            mb_advs = advantages[mb_idx, :new_logprobs.shape[1]]
            mb_mask = resp_mask[mb_idx, :new_logprobs.shape[1]] * local_mask

            log_ratio = new_logprobs - mb_old_logprobs
            ratio = torch.exp(log_ratio)

            pg_loss1 = -mb_advs * ratio
            pg_loss2 = -mb_advs * torch.clamp(ratio, 1.0 - CFG.clip_range, 1.0 + CFG.clip_range)
            policy_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_mask)

            # clipped value loss
            value_pred_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                -CFG.value_loss_clip, CFG.value_loss_clip
            )
            vf_loss1 = (new_values - mb_returns) ** 2
            vf_loss2 = (value_pred_clipped - mb_returns) ** 2
            value_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), mb_mask)

            ent_bonus = masked_mean(entropy, mb_mask)

            # KL to frozen reference
            approx_ref_kl = new_logprobs - mb_ref_logprobs
            kl_loss = masked_mean(approx_ref_kl, mb_mask)

            loss = (
                policy_loss
                + CFG.vf_coef * value_loss
                + CFG.kl_coef * kl_loss
                - CFG.ent_coef * ent_bonus
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), CFG.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > CFG.clip_range).float()
                stats["loss"] += loss.item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += ent_bonus.item()
                stats["approx_kl"] += masked_mean(log_ratio, mb_mask).item()
                stats["clipfrac"] += masked_mean(clipfrac, mb_mask).item()
                n_steps += 1

    for k in stats:
        stats[k] /= max(n_steps, 1)
    return stats


def evaluate(policy, ref_model, tokenizer, device: str, n: int = 30):
    rewards = []
    action_correct = 0
    final_correct = 0

    for _ in range(n):
        out = collect_episode(policy, ref_model, tokenizer, device)
        rewards.append(out["reward"])
        info = out["info"]
        action_correct += int(info["action_correct"])
        final_correct += int(info["final_correct"])

    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_action_acc": action_correct / n,
        "eval_final_acc": final_correct / n,
    }


def main():
    seed_everything(CFG.seed)
    os.makedirs(CFG.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = PolicyWithValueHead(CFG.model_name).to(CFG.device)
    ref_model = AutoModelForCausalLM.from_pretrained(CFG.model_name).to(CFG.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=CFG.lr)

    print(f"Device: {CFG.device}")
    print(f"Model : {CFG.model_name}")
    print("Start PPO training...\n")

    for update in trange(1, CFG.total_updates + 1, desc="updates"):
        episodes = []
        rewards = []
        action_acc = 0
        final_acc = 0

        # 收集 rollout
        for _ in range(CFG.episodes_per_update):
            out = collect_episode(policy, ref_model, tokenizer, CFG.device)
            episodes.append(out["ep_action"])
            episodes.append(out["ep_final"])

            rewards.append(out["reward"])
            action_acc += int(out["info"]["action_correct"])
            final_acc += int(out["info"]["final_correct"])

        batch = build_training_batch(episodes, tokenizer)
        train_stats = ppo_update(policy, optimizer, batch, tokenizer, CFG.device)

        mean_reward = float(np.mean(rewards))
        action_acc /= CFG.episodes_per_update
        final_acc /= CFG.episodes_per_update

        if update % 10 == 0 or update == 1:
            eval_stats = evaluate(policy, ref_model, tokenizer, CFG.device, n=20)
            print(
                f"\n[update {update}] "
                f"train_reward={mean_reward:.3f} "
                f"train_action_acc={action_acc:.2%} "
                f"train_final_acc={final_acc:.2%} "
                f"loss={train_stats['loss']:.4f} "
                f"pg={train_stats['policy_loss']:.4f} "
                f"vf={train_stats['value_loss']:.4f} "
                f"ent={train_stats['entropy']:.4f} "
                f"kl={train_stats['approx_kl']:.4f}"
            )
            print(
                f"[eval] reward={eval_stats['eval_reward']:.3f} "
                f"action_acc={eval_stats['eval_action_acc']:.2%} "
                f"final_acc={eval_stats['eval_final_acc']:.2%}"
            )

            # 打印一个样例
            sample = collect_episode(policy, ref_model, tokenizer, CFG.device)
            meta = sample["ep_final"]["meta"]
            print("---- sample ----")
            print(f"Q: {meta['a']} + {meta['b']}")
            print(meta["action_text"])
            print(meta["final_text"])
            print(f"reward={sample['reward']:.3f}")
            print("----------------")

    policy.save_pretrained(CFG.save_dir, tokenizer)
    print(f"\nSaved to: {CFG.save_dir}")


if __name__ == "__main__":
    main()