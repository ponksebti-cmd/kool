"""
data/dataset_config.py
Dataset sources, sampling weights, and GCS paths for pretraining and SFT.

Edit the GCS paths to point to your bucket before running any data scripts.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ── GCS root ──────────────────────────────────────────────────────────────────
GCS_BUCKET = "gs://<YOUR_BUCKET>"
RAW_DATA_ROOT = f"{GCS_BUCKET}/data/raw"
PROCESSED_ROOT = f"{GCS_BUCKET}/data/pretrain/packed"
SFT_ROOT = f"{GCS_BUCKET}/data/sft/packed"
TOKENIZER_PATH = f"{GCS_BUCKET}/tokenizer/mistral_32k.model"


# ── Pretraining corpus ────────────────────────────────────────────────────────
# Total target: ~19B tokens after deduplication and packing.
# Weights are applied during Grain data loading (not upsampling storage).

PRETRAIN_SOURCES: Dict[str, dict] = {
    "fineweb_edu": {
        "hf_repo": "HuggingFaceFW/fineweb-edu",
        "hf_subset": "sample-100BT",          # use score>=3 filter
        "min_edu_score": 3,
        "target_tokens": 12_000_000_000,
        "weight": 0.63,
        "description": "High-quality educational/reasoning web text (FineWeb-Edu score>=3)",
        "license": "ODC-By",
    },
    "stack_lite": {
        "hf_repo": "bigcode/the-stack-dedup",
        "languages": ["python", "javascript", "bash", "markdown"],
        "target_tokens": 3_000_000_000,
        "weight": 0.16,
        "description": "Deduplicated code (Python, JS, Bash) from The Stack",
        "license": "various permissive",
    },
    "wikipedia": {
        "hf_repo": "wikimedia/wikipedia",
        "hf_subset": "20231101.en",
        "target_tokens": 2_000_000_000,
        "weight": 0.10,
        "description": "English Wikipedia 2024 dump — factual grounding",
        "license": "CC-BY-SA",
    },
    "openwebmath": {
        "hf_repo": "open-web-math/open-web-math",
        "target_tokens": 1_000_000_000,
        "weight": 0.06,
        "description": "Web-scraped mathematical text — math reasoning signal",
        "license": "ODC-By",
    },
    "casual_web": {
        # Use DolmaCasual or CC-News as style diversity supplement.
        "hf_repo": "allenai/dolma",
        "hf_subset": "v1_7-cc_en_head",
        "target_tokens": 1_000_000_000,
        "weight": 0.05,
        "description": "Casual web text — style diversity and recency",
        "license": "ODC-By",
    },
}

# Sanity-check weights sum to 1.0
assert abs(sum(s["weight"] for s in PRETRAIN_SOURCES.values()) - 1.0) < 1e-6, \
    "Pretrain dataset weights must sum to 1.0"

PRETRAIN_TARGET_TOKENS = 19_000_000_000


# ── SFT corpus ────────────────────────────────────────────────────────────────
# Total target: ~300M tokens.
# Loss is computed ONLY on assistant spans (loss_mask=1 for assistant tokens).

SFT_SOURCES: Dict[str, dict] = {
    "tulu3": {
        "hf_repo": "allenai/tulu-3-sft-mixture",
        "target_tokens": 100_000_000,
        "weight": 0.33,
        "description": "Tulu 3 SFT mix — broad instruction coverage",
        "license": "Apache-2.0",
    },
    "openhermes": {
        "hf_repo": "teknium/OpenHermes-2.5",
        "target_tokens": 80_000_000,
        "weight": 0.27,
        "description": "GPT-4 distilled Q&A and reasoning",
        "license": "CC-BY-4.0",
    },
    "metamathqa": {
        "hf_repo": "meta-math/MetaMathQA",
        "target_tokens": 30_000_000,
        "weight": 0.10,
        "description": "Augmented math word problems with step-by-step solutions",
        "license": "MIT",
    },
    "evol_code": {
        "hf_repo": "nickrosh/Evol-Instruct-Code-80k-v1",
        "target_tokens": 30_000_000,
        "weight": 0.10,
        "description": "Code instruction-following (Evol-Instruct)",
        "license": "Apache-2.0",
    },
    "cot_distilled": {
        # Generated offline by generate_cot.py
        "local_path": "./data/cot_distilled.jsonl",
        "gcs_path": f"{GCS_BUCKET}/data/cot_distilled.jsonl",
        "target_tokens": 60_000_000,
        "weight": 0.20,
        "description": "Teacher-distilled CoT: math, logic, commonsense, code, QA",
        "license": "generated",
    },
}

assert abs(sum(s["weight"] for s in SFT_SOURCES.values()) - 1.0) < 1e-6, \
    "SFT dataset weights must sum to 1.0"

SFT_TARGET_TOKENS = 300_000_000


# ── Chat template ─────────────────────────────────────────────────────────────
CHAT_TEMPLATE = (
    "<|system|>{system}<|end|>\n"
    "<|user|>{user}<|end|>\n"
    "<|assistant|>{assistant}<|end|>\n"
)
SYSTEM_DEFAULT = "You are a helpful assistant."

# Special token IDs (must match tokenizer vocabulary)
# These will be added as special tokens to the Mistral tokenizer if not present.
SPECIAL_TOKENS = {
    "<|system|>":    32001,
    "<|user|>":      32002,
    "<|assistant|>": 32003,
    "<|end|>":       32004,
}


# ── CoT seed domains ─────────────────────────────────────────────────────────
COT_SEED_DOMAINS: List[dict] = [
    {
        "name": "math",
        "hf_source": "gsm8k",
        "hf_split": "train",
        "n_prompts": 7500,
        "system": "Solve the following math problem step by step, showing all your work.",
        "template": "Problem: {question}\nSolve step by step.",
    },
    {
        "name": "logic",
        "hf_source": "lucasmccabe/logiqa",
        "hf_split": "train",
        "n_prompts": 3000,
        "system": "Think through this logic problem carefully, explaining your reasoning.",
        "template": "Question: {question}\nChoices: {choices}\nExplain your reasoning step by step.",
    },
    {
        "name": "commonsense",
        "hf_source": "commonsense_qa",
        "hf_split": "train",
        "n_prompts": 9000,
        "system": "Answer the following question and explain why.",
        "template": "Question: {question}\nChoices: {choices}\nAnswer and explain.",
    },
    {
        "name": "code",
        "hf_source": "openai_humaneval",
        "hf_split": "test",
        "n_prompts": 3000,
        "system": "Write the requested function and explain your implementation.",
        "template": "{prompt}\nWrite the function and explain how it works.",
    },
    {
        "name": "general_qa",
        "hf_source": "trivia_qa",
        "hf_split": "train",
        "n_prompts": 10000,
        "system": "Answer the question and briefly explain your reasoning.",
        "template": "Question: {question}\nAnswer and explain.",
    },
]

TOTAL_COT_PROMPTS = sum(d["n_prompts"] for d in COT_SEED_DOMAINS)
