"""
eval/perplexity.py
Compute perplexity on a held-out dataset (default: Pile test set).

Usage:
    python eval/perplexity.py \
        --model_path /tmp/lm300m_hf \
        --dataset_name EleutherAI/the_pile_deduplicated \
        --dataset_split test \
        --n_examples 2000 \
        --seq_len 2048 \
        --output eval_results/perplexity.json

Target: perplexity <= 12.0 on Pile test set.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def compute_perplexity(
    model_path: str,
    dataset_name: str = "EleutherAI/the_pile_deduplicated",
    dataset_split: str = "test",
    n_examples: int = 2000,
    seq_len: int = 2048,
    stride: int = 512,
    batch_size: int = 4,
) -> dict:
    """
    Compute sliding-window perplexity on a text dataset.

    Uses a stride to handle sequences longer than seq_len without
    double-penalizing the prefix (standard PPL evaluation methodology).

    Args:
        model_path: Path to HuggingFace model directory.
        dataset_name: HuggingFace dataset name.
        dataset_split: Dataset split to evaluate on.
        n_examples: Number of examples to evaluate (for speed).
        seq_len: Maximum sequence length (should match training context length).
        stride: Sliding window stride (smaller = more accurate, slower).
        batch_size: Batch size for inference.

    Returns:
        Dict with 'perplexity', 'nll', 'n_tokens', 'n_examples'.
    """
    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    logger.info(f"Loading dataset: {dataset_name} / {dataset_split}")
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)

    # Concatenate text into one long token stream
    all_tokens = []
    count = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        count += 1
        if count >= n_examples:
            break

    logger.info(f"Collected {len(all_tokens):,} tokens from {count} examples.")

    # Sliding-window NLL
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for begin in range(0, len(all_tokens) - seq_len, stride):
        end = min(begin + seq_len, len(all_tokens))
        input_ids = torch.tensor([all_tokens[begin:end]], device=device)

        # Only count loss on tokens not in the prefix (first stride tokens)
        target_len = end - begin - stride if begin > 0 else end - begin
        labels = input_ids.clone()
        labels[:, :-target_len] = -100  # mask prefix

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            nll = outputs.loss.item()

        total_nll += nll * target_len
        total_tokens += target_len

        if total_tokens % 100_000 == 0:
            running_ppl = math.exp(total_nll / total_tokens)
            logger.info(
                f"  {total_tokens:,} tokens processed | Running PPL: {running_ppl:.3f}"
            )

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    logger.info(f"Final perplexity: {perplexity:.4f} (NLL: {avg_nll:.4f})")
    logger.info(f"Total tokens evaluated: {total_tokens:,}")

    return {
        "perplexity": perplexity,
        "nll": avg_nll,
        "n_tokens": total_tokens,
        "n_examples": count,
        "dataset": dataset_name,
        "split": dataset_split,
        "model_path": model_path,
        "target_ppl": 12.0,
        "pass": perplexity <= 12.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_name", default="EleutherAI/the_pile_deduplicated")
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--n_examples", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="eval_results/perplexity.json")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results = compute_perplexity(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        n_examples=args.n_examples,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    status = "✓ PASS" if results["pass"] else "✗ MISS (above target)"
    print(f"\nPerplexity: {results['perplexity']:.4f} — Target: ≤ 12.0 — {status}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
