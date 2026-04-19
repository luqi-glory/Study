#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一个最小可运行的 Agent SFT 训练与推理示例。

功能：
1. 把多轮 ReAct / Tool-use 轨迹转换成监督微调样本
2. 用 Hugging Face Transformers + PEFT(LoRA) 做 SFT
3. 提供一个最小运行时：解析模型输出的 <action>，执行工具，再把 observation 回填

安装依赖（示例）：
pip install torch transformers datasets peft accelerate

"""

import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, TaskType


# =========================
# 1. 配置区
# =========================

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./agent_sft_ckpt")
MAX_LENGTH = 1024
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# 2. 特殊 token 与模板
# =========================

SPECIAL_TOKENS = [
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<thought>",
    "</thought>",
    "<action>",
    "</action>",
    "<final>",
    "</final>",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3. 构造训练样本
# =========================

"""
训练数据建议长这样（JSONL）：

{"messages": [
    {"role": "user", "content": "帮我查一下 OpenAI 官网有没有 API 文档入口。"},
    {"role": "assistant", "content": "<thought>需要先搜索官网，再打开结果页。</thought>\n<action>{\"tool\":\"search_web\",\"query\":\"OpenAI API documentation official site\"}</action>"},
    {"role": "observation", "content": "{\"results\":[{\"title\":\"OpenAI API Platform\",\"url\":\"https://platform.openai.com/docs\"}]}"},
    {"role": "assistant", "content": "<thought>已经找到官方文档入口，可以直接回答。</thought>\n<final>有，官方 API 文档入口在 https://platform.openai.com/docs</final>"}
]}

下面先直接在代码里放几条 demo 数据，方便你跑通。
"""


DEMO_DATA = [
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我查一下 OpenAI 官网有没有 API 文档入口。"
            },
            {
                "role": "assistant",
                "content": (
                    "<thought>需要先搜索官网相关结果。</thought>\n"
                    "<action>{\"tool\":\"search_web\",\"query\":\"OpenAI API documentation official site\"}</action>"
                )
            },
            {
                "role": "observation",
                "content": (
                    "{\"results\":["
                    "{\"title\":\"OpenAI API Platform\",\"url\":\"https://platform.openai.com/docs\"},"
                    "{\"title\":\"OpenAI\",\"url\":\"https://openai.com\"}"
                    "]}"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<thought>已经找到官方文档页面，可以直接给出结果。</thought>\n"
                    "<final>有，官方 API 文档入口是 https://platform.openai.com/docs</final>"
                )
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "请帮我算一下 123 * 456。"
            },
            {
                "role": "assistant",
                "content": (
                    "<thought>这是精确计算任务，直接调用计算器更稳妥。</thought>\n"
                    "<action>{\"tool\":\"calculator\",\"expression\":\"123 * 456\"}</action>"
                )
            },
            {
                "role": "observation",
                "content": "{\"result\":56088}"
            },
            {
                "role": "assistant",
                "content": (
                    "<thought>已经得到精确结果。</thought>\n"
                    "<final>123 * 456 = 56088</final>"
                )
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "法国首都是哪里？"
            },
            {
                "role": "assistant",
                "content": (
                    "<thought>这是基础事实，不需要调用工具。</thought>\n"
                    "<final>法国的首都是巴黎。</final>"
                )
            }
        ]
    }
]


def render_messages(messages: List[Dict[str, str]]) -> str:
    """
    把多轮轨迹渲染成一个单字符串，用于 Causal LM 训练。
    """
    role_map = {
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "observation": "<|observation|>",
    }

    chunks = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role not in role_map:
            raise ValueError(f"Unknown role: {role}")
        chunks.append(f"{role_map[role]}\n{content}\n")

    return "".join(chunks).strip()


class AgentTrajectoryDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data:
            text = render_messages(item["messages"])
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            self.examples.append(tokenized)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        # causal lm: labels = input_ids
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["input_ids"].copy(),
        }


# =========================
# 4. 加载模型与 tokenizer
# =========================

def load_model_and_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    # 有些模型没有 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# =========================
# 5. 训练函数
# =========================

def train():
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer(BASE_MODEL)

    train_dataset = AgentTrajectoryDataset(DEMO_DATA, tokenizer, max_length=MAX_LENGTH)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成，模型已保存到: {OUTPUT_DIR}")


# =========================
# 6. 工具定义（运行时）
# =========================

def tool_search_web(query: str) -> Dict[str, Any]:
    """
    这里是 mock 的搜索工具。
    真正项目中你会替换成：
    - SerpAPI / Tavily / Bing API / 自建搜索
    - 甚至浏览器 agent
    """
    mock_db = {
        "OpenAI API documentation official site": {
            "results": [
                {
                    "title": "OpenAI API Platform",
                    "url": "https://platform.openai.com/docs"
                },
                {
                    "title": "OpenAI",
                    "url": "https://openai.com"
                }
            ]
        },
        "法国首都": {
            "results": [
                {"title": "法国 - 维基百科", "url": "https://zh.wikipedia.org/wiki/法国"}
            ]
        }
    }
    return mock_db.get(query, {"results": []})


def tool_calculator(expression: str) -> Dict[str, Any]:
    """
    非常简化的 calculator。
    注意：生产环境不要直接 eval 未信任输入。
    """
    allowed = re.fullmatch(r"[0-9\.\+\-\*\/\(\) ]+", expression)
    if not allowed:
        return {"error": "非法表达式"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {
    "search_web": tool_search_web,
    "calculator": tool_calculator,
}


# =========================
# 7. 解析模型输出
# =========================

ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)
FINAL_PATTERN = re.compile(r"<final>(.*?)</final>", re.DOTALL)


def extract_action(text: str) -> Optional[Dict[str, Any]]:
    match = ACTION_PATTERN.search(text)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
        return payload
    except json.JSONDecodeError:
        return None


def extract_final(text: str) -> Optional[str]:
    match = FINAL_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


# =========================
# 8. Agent 运行器
# =========================

@dataclass
class AgentConfig:
    max_steps: int = 5
    max_new_tokens: int = 256
    temperature: float = 0.2
    do_sample: bool = False


class SimpleToolAgent:
    def __init__(self, model_path: str, config: AgentConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.config = config

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return text[len(prompt):].strip() if text.startswith(prompt) else text

    def run(self, user_query: str) -> str:
        """
        最小 Agent loop:
        - 构造 prompt
        - 让模型输出 assistant 段
        - 如果有 action，就执行工具
        - 把 observation 接回去
        - 直到得到 final
        """
        transcript = f"<|user|>\n{user_query}\n"

        for step in range(self.config.max_steps):
            prompt = transcript + "<|assistant|>\n"
            assistant_output = self._generate(prompt)

            # 只截取第一段 assistant 输出，避免把后续重复生成带进去
            # 这里做一个简单截断：碰到下一个 role token 就停
            stop_tokens = ["<|user|>", "<|assistant|>", "<|observation|>"]
            cut_pos = len(assistant_output)
            for st in stop_tokens:
                pos = assistant_output.find(st, 1)
                if pos != -1:
                    cut_pos = min(cut_pos, pos)
            assistant_output = assistant_output[:cut_pos].strip()

            print(f"\n[Assistant step {step + 1}]")
            print(assistant_output)

            transcript += f"<|assistant|>\n{assistant_output}\n"

            final_answer = extract_final(assistant_output)
            if final_answer is not None:
                return final_answer

            action = extract_action(assistant_output)
            if action is None:
                return "模型没有输出合法的 <action> 或 <final>。"

            tool_name = action.get("tool")
            if tool_name not in TOOL_REGISTRY:
                observation = {"error": f"未知工具: {tool_name}"}
            else:
                try:
                    kwargs = {k: v for k, v in action.items() if k != "tool"}
                    observation = TOOL_REGISTRY[tool_name](**kwargs)
                except Exception as e:
                    observation = {"error": f"工具执行失败: {str(e)}"}

            obs_text = json.dumps(observation, ensure_ascii=False)
            print("[Observation]")
            print(obs_text)

            transcript += f"<|observation|>\n{obs_text}\n"

        return "超过最大步数，任务未完成。"


# =========================
# 9. 主函数
# =========================

def demo_inference():
    config = AgentConfig(max_steps=5, max_new_tokens=128, temperature=0.2, do_sample=False)
    agent = SimpleToolAgent(OUTPUT_DIR, config)

    user_query = "帮我查一下 OpenAI 官网有没有 API 文档入口。"
    answer = agent.run(user_query)

    print("\n[Final Answer]")
    print(answer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "demo"])
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "demo":
        demo_inference()