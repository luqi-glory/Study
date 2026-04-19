#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO post-training for an agent-style LLM.

Features:
- Hugging Face causal LM
- LoRA fine-tuning
- Frozen reference model
- Group sampling per prompt
- Relative advantage normalization within each group
- PPO-style clipped loss
- KL regularization toward reference policy
- Agent reward for tool-use + final answer correctness

This is a runnable demo for agent post-training.
You can replace:
  1) build_dataset()
  2) reward_fn()
with your own agent environment / task.

Recommended first run:
  python train_grpo_agent.py \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --output_dir ./grpo_agent_ckpt

If you want a lighter fallback model, try:
  --model_name distilgpt2

Notes:
- Instruct/chat models usually work much better than plain base models.
- For real post-training, use a stronger base model and a richer reward function.
"""

import os
import re
import math
import json
import copy
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def try_apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Use tokenizer chat template when available; otherwise fallback to a plain text prompt.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return (
        f"System: {system_prompt}\n"
        f"User: {user_prompt}\n"
        f"Assistant:"
    )


# =========================
# Synthetic Agent Dataset
# =========================

def build_dataset(num_samples: int = 256) -> List[Dict[str, Any]]:
    """
    Build a toy agent dataset:
    the model should solve arithmetic by using a calculator action.

    Desired output format:
      Thought: ...
      Action: calculator
      Action Input: <expression>
      Observation: <optional, model may omit>
      Final Answer: <number>

    Reward checks:
    - format compliance
    - valid calculator action
    - action input matches the gold expression
    - final answer correctness
    """
    samples = []
    ops = ["+", "-", "*"]

    for _ in range(num_samples):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(ops)

        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b

        expr = f"{a} {op} {b}"
        question = f"Use the calculator tool to compute: {expr}"

        samples.append(
            {
                "question": question,
                "gold_expression": expr,
                "gold_answer": str(ans),
            }
        )
    return samples


# =========================
# Reward / Agent Parsing
# =========================

ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE)
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE)


def safe_eval_arithmetic(expr: str):
    """
    Very restricted arithmetic evaluator.
    Allows digits, spaces, + - * / ( )
    """
    expr = expr.strip()
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        raise ValueError(f"Unsafe expression: {expr}")
    return eval(expr, {"__builtins__": {}}, {})


def parse_agent_output(text: str) -> Dict[str, Any]:
    action = None
    action_input = None
    final_answer = None

    m = ACTION_RE.search(text)
    if m:
        action = m.group(1).strip()

    m = ACTION_INPUT_RE.search(text)
    if m:
        action_input = m.group(1).strip()

    m = FINAL_RE.search(text)
    if m:
        final_answer = m.group(1).strip()

    return {
        "action": action,
        "action_input": action_input,
        "final_answer": final_answer,
    }


def normalize_number_text(x: str) -> str:
    x = x.strip()
    try:
        v = float(x)
        if abs(v - round(v)) < 1e-8:
            return str(int(round(v)))
        return str(v)
    except Exception:
        return x


def reward_fn(sample: Dict[str, Any], completion_text: str) -> float:
    """
    Reward design:
    - +0.1: has Action
    - +0.2: action == calculator
    - +0.2: has Action Input
    - +0.2: Action Input exactly matches gold expression
    - +0.1: Final Answer present
    - +0.2: Final Answer correct
    - small bonus if calculator(Action Input) actually equals Final Answer

    Total nominal max: 1.0
    """
    parsed = parse_agent_output(completion_text)
    reward = 0.0

    action = parsed["action"]
    action_input = parsed["action_input"]
    final_answer = parsed["final_answer"]

    if action is not None:
        reward += 0.1

    if action is not None and action.strip().lower() == "calculator":
        reward += 0.2

    if action_input is not None:
        reward += 0.2

    if action_input is not None and action_input.strip() == sample["gold_expression"]:
        reward += 0.2

    if final_answer is not None:
        reward += 0.1

    if final_answer is not None and normalize_number_text(final_answer) == normalize_number_text(sample["gold_answer"]):
        reward += 0.2

    # Consistency bonus: tool input should evaluate to final answer.
    if action_input is not None and final_answer is not None:
        try:
            tool_out = safe_eval_arithmetic(action_input)
            if normalize_number_text(str(tool_out)) == normalize_number_text(final_answer):
                reward += 0.2
        except Exception:
            pass

    return min(reward, 1.0)


# =========================
# Prompt Construction
# =========================

SYSTEM_PROMPT = """You are an agent that must solve the user's task by using tools correctly.

Follow this format exactly:
Thought: <brief reasoning>
Action: calculator
Action Input: <arithmetic expression>
Final Answer: <result>

Rules:
- Use the calculator tool for arithmetic.
- Keep Action exactly as: calculator
- Action Input must contain only the arithmetic expression.
- Final Answer must be only the final result.
"""


def build_prompt(tokenizer, sample: Dict[str, Any]) -> str:
    return try_apply_chat_template(
        tokenizer,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=sample["question"],
    )


# =========================
# Logprob Helpers
# =========================

@torch.no_grad()
def compute_logprobs_for_completion(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute token logprobs for completion tokens only.

    Returns:
      per_sample_logprobs: list of shape [completion_len]
      per_sample_masks:    list of shape [completion_len] (all ones here)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # logprobs for predicting token at position t from logits at t-1
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, T-1]

    per_sample_logprobs = []
    per_sample_masks = []

    B = input_ids.size(0)
    for i in range(B):
        # completion starts at original prompt token index prompt_len
        # token_log_probs is aligned to input_ids[:, 1:]
        # so completion token j at absolute position p contributes at token_log_probs[p-1]
        start = int(prompt_lens[i].item()) - 1
        if start < 0:
            start = 0
        lp = token_log_probs[i, start:]
        mask = torch.ones_like(lp, dtype=torch.float32)
        per_sample_logprobs.append(lp)
        per_sample_masks.append(mask)

    return per_sample_logprobs, per_sample_masks


def pad_tensor_list(xs: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length 1D tensors to [B, T_max].
    Returns:
      padded, mask
    """
    max_len = max(x.numel() for x in xs)
    device = xs[0].device
    dtype = xs[0].dtype

    padded = torch.full((len(xs), max_len), pad_value, dtype=dtype, device=device)
    mask = torch.zeros((len(xs), max_len), dtype=torch.float32, device=device)

    for i, x in enumerate(xs):
        L = x.numel()
        padded[i, :L] = x
        mask[i, :L] = 1.0

    return padded, mask


# =========================
# Rollout
# =========================

@torch.no_grad()
def rollout_group(
    model,
    ref_model,
    tokenizer,
    prompt_text: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
):
    """
    Generate a group of completions for one prompt.
    Also compute old policy logprobs and ref logprobs on those sampled completions.
    """
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = attention_mask.sum(dim=1)  # [1]

    # repeat prompt for group sampling
    input_ids = input_ids.repeat(group_size, 1)
    attention_mask = attention_mask.repeat(group_size, 1)
    prompt_lens = prompt_len.repeat(group_size)

    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequences = gen_out  # [G, T_full]

    # rebuild attention mask for full sequences
    full_attention_mask = (sequences != tokenizer.pad_token_id).long()

    # old policy logprobs from sampled rollout policy
    old_logprob_list, _ = compute_logprobs_for_completion(
        model=model,
        input_ids=sequences,
        attention_mask=full_attention_mask,
        prompt_lens=prompt_lens,
    )

    # reference logprobs
    ref_logprob_list, _ = compute_logprobs_for_completion(
        model=ref_model,
        input_ids=sequences,
        attention_mask=full_attention_mask,
        prompt_lens=prompt_lens,
    )

    completion_token_ids = []
    completion_texts = []
    for i in range(group_size):
        seq = sequences[i]
        pl = int(prompt_lens[i].item())
        completion_ids = seq[pl:]

        # truncate at first eos if present
        if tokenizer.eos_token_id is not None:
            eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                completion_ids = completion_ids[: eos_positions[0].item()]

        completion_token_ids.append(completion_ids)
        completion_texts.append(
            tokenizer.decode(completion_ids, skip_special_tokens=True)
        )

    return {
        "prompt_lens": prompt_lens,
        "sequences": sequences,
        "full_attention_mask": full_attention_mask,
        "completion_token_ids": completion_token_ids,
        "completion_texts": completion_texts,
        "old_logprob_list": old_logprob_list,
        "ref_logprob_list": ref_logprob_list,
    }


# =========================
# GRPO Loss
# =========================

def compute_group_advantages(rewards: List[float], eps: float = 1e-6) -> torch.Tensor:
    """
    GRPO-style relative advantage:
      A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    adv = (r - mean) / (std + eps)
    return adv


def grpo_step(
    policy_model,
    ref_model,
    tokenizer,
    optimizer,
    sample: Dict[str, Any],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    clip_eps: float,
    kl_beta: float,
    device: torch.device,
):
    """
    One GRPO update step for one prompt.
    """
    prompt_text = build_prompt(tokenizer, sample)

    rollout = rollout_group(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    completion_texts = rollout["completion_texts"]
    rewards = [reward_fn(sample, t) for t in completion_texts]
    advantages = compute_group_advantages(rewards).to(device)  # [G]

    # recompute current policy logprobs for training graph
    sequences = rollout["sequences"]
    full_attention_mask = rollout["full_attention_mask"]
    prompt_lens = rollout["prompt_lens"]

    cur_logprob_list, _ = compute_logprobs_for_completion(
        model=policy_model,
        input_ids=sequences,
        attention_mask=full_attention_mask,
        prompt_lens=prompt_lens,
    )

    old_logprob_padded, token_mask = pad_tensor_list(
        [x.detach() for x in rollout["old_logprob_list"]],
        pad_value=0.0
    )
    ref_logprob_padded, _ = pad_tensor_list(
        [x.detach() for x in rollout["ref_logprob_list"]],
        pad_value=0.0
    )
    cur_logprob_padded, _ = pad_tensor_list(cur_logprob_list, pad_value=0.0)

    # ratio = exp(log pi_theta - log pi_theta_old)
    ratio = torch.exp(cur_logprob_padded - old_logprob_padded)

    adv = advantages.unsqueeze(1)  # [G, 1]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_term = torch.minimum(unclipped, clipped)

    # lightweight token-level KL regularizer toward reference policy
    # Here we use a simple squared logprob difference proxy for stability.
    kl_proxy = 0.5 * (cur_logprob_padded - ref_logprob_padded) ** 2

    obj = policy_term - kl_beta * kl_proxy
    obj = obj * token_mask

    loss = -obj.sum() / token_mask.sum().clamp_min(1.0)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()

    metrics = {
        "loss": float(loss.item()),
        "reward_mean": float(sum(rewards) / len(rewards)),
        "reward_std": float(torch.tensor(rewards).std(unbiased=False).item()),
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std(unbiased=False).item()),
        "sample_completion_best": completion_texts[int(torch.tensor(rewards).argmax().item())],
        "sample_completion_worst": completion_texts[int(torch.tensor(rewards).argmin().item())],
        "rewards": rewards,
    }
    return metrics


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    device: torch.device,
    num_eval_samples: int = 32,
    max_new_tokens: int = 64,
):
    model.eval()
    subset = random.sample(dataset, min(num_eval_samples, len(dataset)))

    rewards = []
    exact = 0

    for sample in subset:
        prompt = build_prompt(tokenizer, sample)
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        r = reward_fn(sample, text)
        rewards.append(r)

        parsed = parse_agent_output(text)
        if parsed["final_answer"] is not None and normalize_number_text(parsed["final_answer"]) == normalize_number_text(sample["gold_answer"]):
            exact += 1

    return {
        "eval_reward_mean": float(sum(rewards) / len(rewards)),
        "eval_exact_match": exact / len(rewards),
    }


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./grpo_agent_ckpt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--updates_per_epoch", type=int, default=200)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.02)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # 1) data
    full_train = build_dataset(args.train_size)
    full_eval = build_dataset(args.eval_size)

    # 2) tokenizer + model
    print(f"[Info] loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    ensure_pad_token(tokenizer, base_model)

    # 3) reference model (frozen)
    print("[Info] loading frozen reference model")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    ensure_pad_token(tokenizer, ref_model)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # 4) LoRA policy model
    print("[Info] applying LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    )

    policy_model = get_peft_model(base_model, lora_config)
    policy_model.to(device)
    policy_model.train()

    # distilgpt2 / gpt2 fallback compatibility:
    # if target_modules above don't exist, retry with GPT-style module names
    # This block makes the script more portable.
    try:
        policy_model.print_trainable_parameters()
    except Exception:
        pass

    # If LoRA target_modules fail for some architectures, rebuild with alternative targets.
    # We detect this by checking trainable params count.
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    if trainable_params == 0:
        print("[Warn] No LoRA params were attached with default target modules. Retrying GPT-style modules.")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        ensure_pad_token(tokenizer, base_model)
        alt_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"],
        )
        policy_model = get_peft_model(base_model, alt_config)
        policy_model.to(device)
        policy_model.train()
        try:
            policy_model.print_trainable_parameters()
        except Exception:
            pass

    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 5) training
    global_step = 0
    print("[Info] start training")

    for epoch in range(args.epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")

        for step in range(args.updates_per_epoch):
            sample = random.choice(full_train)

            policy_model.train()
            metrics = grpo_step(
                policy_model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                sample=sample,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                clip_eps=args.clip_eps,
                kl_beta=args.kl_beta,
                device=device,
            )

            global_step += 1

            if global_step % 10 == 0:
                print(
                    f"[step {global_step:05d}] "
                    f"loss={metrics['loss']:.4f} "
                    f"reward_mean={metrics['reward_mean']:.4f} "
                    f"reward_std={metrics['reward_std']:.4f}"
                )

            if global_step % args.eval_every == 0:
                eval_metrics = evaluate(
                    model=policy_model,
                    tokenizer=tokenizer,
                    dataset=full_eval,
                    device=device,
                    num_eval_samples=min(32, len(full_eval)),
                    max_new_tokens=args.max_new_tokens,
                )
                print(
                    f"[eval {global_step:05d}] "
                    f"eval_reward_mean={eval_metrics['eval_reward_mean']:.4f} "
                    f"eval_exact_match={eval_metrics['eval_exact_match']:.4f}"
                )
                print("[best sample]")
                print(metrics["sample_completion_best"])
                print("-" * 60)

            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"[Info] saved checkpoint to {ckpt_dir}")

    # final save
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[Info] final model saved to {final_dir}")

    # final eval
    final_eval = evaluate(
        model=policy_model,
        tokenizer=tokenizer,
        dataset=full_eval,
        device=device,
        num_eval_samples=min(64, len(full_eval)),
        max_new_tokens=args.max_new_tokens,
    )
    print("[Final Eval]", json.dumps(final_eval, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()