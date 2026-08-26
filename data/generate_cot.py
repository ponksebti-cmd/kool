"""
data/generate_cot.py
Offline teacher CoT distillation — generates step-by-step chain-of-thought examples
using a teacher model API (Gemini, GPT-4o-mini, or local model).

Runs on CPU before TPU training starts. Does NOT consume TPU budget.
Expected cost: ~$15–50 in API calls for 33K prompts.
Expected output: ~60M tokens of diverse CoT instruction data.

Usage:
    # Using Gemini Flash (recommended — cheapest capable teacher)
    GOOGLE_API_KEY=<key> python data/generate_cot.py \
        --teacher gemini \
        --model gemini-1.5-flash-8b \
        --output data/cot_distilled.jsonl \
        --workers 32

    # Using OpenAI GPT-4o-mini
    OPENAI_API_KEY=<key> python data/generate_cot.py \
        --teacher openai \
        --model gpt-4o-mini \
        --output data/cot_distilled.jsonl \
        --workers 32

    # Upload to GCS when done
    gsutil cp data/cot_distilled.jsonl gs://<BUCKET>/data/sft/raw/cot_distilled.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ── Prompt seeds (sampled from HuggingFace datasets at runtime) ───────────────

def load_seeds_from_datasets(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load seed prompts from HuggingFace datasets as defined in dataset_config.COT_SEED_DOMAINS.
    Returns a list of dicts with keys: domain, system, prompt.
    """
    from datasets import load_dataset
    from data.dataset_config import COT_SEED_DOMAINS

    all_seeds = []
    for domain_cfg in COT_SEED_DOMAINS:
        logger.info(f"Loading seeds for domain: {domain_cfg['name']} ({domain_cfg['n_prompts']} prompts)")
        try:
            ds = load_dataset(domain_cfg["hf_source"], split=domain_cfg["hf_split"], streaming=True)
            count = 0
            for ex in ds:
                if count >= domain_cfg["n_prompts"]:
                    break
                try:
                    prompt = _format_seed(ex, domain_cfg)
                    if prompt:
                        all_seeds.append({
                            "domain": domain_cfg["name"],
                            "system": domain_cfg["system"],
                            "prompt": prompt,
                        })
                        count += 1
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Failed to load {domain_cfg['name']}: {e}")

    random.shuffle(all_seeds)
    logger.info(f"Loaded {len(all_seeds)} total seed prompts across {len(COT_SEED_DOMAINS)} domains.")
    return all_seeds


def _format_seed(ex: dict, domain_cfg: dict) -> Optional[str]:
    """Format a dataset example into a prompt string using the domain template."""
    template = domain_cfg["template"]
    domain = domain_cfg["name"]

    if domain == "math":
        return template.format(question=ex.get("question", ""))
    elif domain == "logic":
        choices = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(ex.get("options", [])))
        return template.format(question=ex.get("query", ""), choices=choices)
    elif domain == "commonsense":
        choices_obj = ex.get("question", {}).get("choices", {})
        labels = choices_obj.get("label", [])
        texts = choices_obj.get("text", [])
        choices = "\n".join(f"  {l}) {t}" for l, t in zip(labels, texts))
        q = ex.get("question", {}).get("stem", "")
        return template.format(question=q, choices=choices)
    elif domain == "code":
        return template.format(prompt=ex.get("prompt", ""))
    elif domain == "general_qa":
        return template.format(question=ex.get("question", ""))
    return None


# ── Teacher API clients ────────────────────────────────────────────────────────

class TeacherClient:
    """Abstract base for teacher model API clients."""

    def generate(self, system: str, prompt: str) -> str:
        raise NotImplementedError

    async def agenerate(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(TeacherClient):
    """Gemini teacher via google-generativeai."""

    def __init__(self, model: str = "gemini-1.5-flash-8b"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=(
                "You are a helpful teacher. Always think step by step. "
                "Wrap your reasoning in <think>...</think> tags, "
                "then give a clear final answer."
            ),
        )
        self._genai = genai

    def generate(self, system: str, prompt: str) -> str:
        full_prompt = f"{system}\n\n{prompt}"
        response = self.model.generate_content(full_prompt)
        return response.text

    async def agenerate(self, system: str, prompt: str) -> str:
        full_prompt = f"{system}\n\n{prompt}"
        response = await self.model.generate_content_async(full_prompt)
        return response.text


class OpenAIClient(TeacherClient):
    """OpenAI teacher via openai SDK."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        import asyncio
        return asyncio.run(self.agenerate(system, prompt))

    async def agenerate(self, system: str, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a helpful teacher. Always think step by step. "
                    "Wrap your reasoning in <think>...</think> tags, "
                    "then give a clear final answer."
                )},
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return resp.choices[0].message.content


# ── Generation pipeline ────────────────────────────────────────────────────────

async def generate_batch(
    seeds: List[Dict[str, Any]],
    client: TeacherClient,
    output_path: str,
    workers: int = 32,
    max_retries: int = 3,
    resume: bool = True,
) -> None:
    """
    Async batch generation with rate limiting, retries, and resume support.

    Writes results incrementally to output_path (JSONL) so the run can be
    resumed if interrupted.
    """
    # Resume: check which prompts are already done
    done_prompts: set = set()
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    done_prompts.add(ex.get("prompt_hash", ""))
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resume: {len(done_prompts)} prompts already done.")

    pending = [s for s in seeds if _hash(s["prompt"]) not in done_prompts]
    logger.info(f"Generating {len(pending)} remaining prompts with {workers} workers...")

    semaphore = asyncio.Semaphore(workers)
    output_file = open(output_path, "a", encoding="utf-8")
    completed = 0
    errors = 0

    async def process_one(seed: Dict[str, Any]) -> None:
        nonlocal completed, errors
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    response = await client.agenerate(seed["system"], seed["prompt"])
                    record = {
                        "domain": seed["domain"],
                        "system": seed["system"],
                        "instruction": seed["prompt"],
                        "output": response,
                        "prompt_hash": _hash(seed["prompt"]),
                        "teacher": getattr(client, "model", "unknown"),
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed += 1
                    if completed % 500 == 0:
                        logger.info(f"Progress: {completed}/{len(pending)} ({errors} errors)")
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.warning(f"Failed after {max_retries} attempts: {e}")
                        errors += 1

    tasks = [process_one(s) for s in pending]
    await asyncio.gather(*tasks)
    output_file.close()
    logger.info(f"Generation complete. {completed} successful, {errors} failed.")


def _hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher CoT distillation data generation.")
    p.add_argument("--teacher", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--model", default="gemini-1.5-flash-8b",
                   help="Teacher model name (passed to API client).")
    p.add_argument("--output", default="data/cot_distilled.jsonl")
    p.add_argument("--workers", type=int, default=32,
                   help="Number of concurrent API requests.")
    p.add_argument("--no_resume", action="store_true",
                   help="Start from scratch even if output file exists.")
    p.add_argument("--gcs_upload", default=None,
                   help="If set, upload output to this GCS path after completion.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Load seeds
    seeds = load_seeds_from_datasets()
    logger.info(f"Total seeds: {len(seeds)}")

    # Build client
    if args.teacher == "gemini":
        client = GeminiClient(model=args.model)
    elif args.teacher == "openai":
        client = OpenAIClient(model=args.model)
    else:
        raise ValueError(f"Unknown teacher: {args.teacher}")

    # Generate
    asyncio.run(
        generate_batch(
            seeds=seeds,
            client=client,
            output_path=args.output,
            workers=args.workers,
            resume=not args.no_resume,
        )
    )

    # Upload to GCS
    if args.gcs_upload:
        import subprocess
        logger.info(f"Uploading {args.output} → {args.gcs_upload}")
        subprocess.run(["gsutil", "cp", args.output, args.gcs_upload], check=True)
        logger.info("Upload complete.")


if __name__ == "__main__":
    main()
