"""
data/tokenize_and_pack.py
Tokenize and sequence-pack pretraining and SFT data into fixed-length arrays.

For PRETRAINING:
  - Reads deduplicated JSONL from GCS
  - Tokenizes with Mistral SentencePiece (32K vocab)
  - Greedy sequence packing: fill 2048-token windows with EOS-separated documents
  - Outputs uint16 numpy arrays to GCS as sharded files

For SFT:
  - Applies the chat template (system/user/assistant)
  - Emits a parallel loss_mask array (1 = assistant tokens, 0 = masked)
  - Packs by conversation (no cross-conversation packing)

Usage:
    # Pretrain
    python data/tokenize_and_pack.py \
        --mode pretrain \
        --input_glob "gs://<BUCKET>/data/deduped/**/*.jsonl.gz" \
        --output_dir "gs://<BUCKET>/data/pretrain/packed/" \
        --tokenizer_path "gs://<BUCKET>/tokenizer/mistral_32k.model" \
        --seq_len 2048 \
        --num_workers 32 \
        --shard_size_mb 256

    # SFT
    python data/tokenize_and_pack.py \
        --mode sft \
        --input_glob "gs://<BUCKET>/data/sft/raw/**/*.jsonl" \
        --output_dir "gs://<BUCKET>/data/sft/packed/" \
        --tokenizer_path "gs://<BUCKET>/tokenizer/mistral_32k.model" \
        --seq_len 2048 \
        --num_workers 8
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
from pathlib import Path
from typing import Generator, Iterator, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SEQ_LEN = 2048


# ── Tokenizer wrapper ──────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_path: str):
    """Load SentencePiece tokenizer from a local or GCS path."""
    import sentencepiece as spm
    local_path = tokenizer_path
    if tokenizer_path.startswith("gs://"):
        import subprocess, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
        subprocess.run(["gsutil", "cp", tokenizer_path, tmp.name], check=True)
        local_path = tmp.name
    sp = spm.SentencePieceProcessor()
    sp.load(local_path)
    return sp


# ── Pretrain packing ───────────────────────────────────────────────────────────

def read_jsonl_gz(path: str) -> Generator[str, None, None]:
    """Yield text strings from a gzipped JSONL file (local or GCS)."""
    if path.startswith("gs://"):
        import subprocess
        result = subprocess.run(["gsutil", "cat", path], capture_output=True)
        fobj = io.BytesIO(result.stdout)
    else:
        fobj = open(path, "rb")

    with gzip.open(fobj, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue


def greedy_pack_pretrain(
    token_stream: Iterator[List[int]],
    seq_len: int,
    eos_id: int,
) -> Generator[np.ndarray, None, None]:
    """
    Greedily pack variable-length tokenized documents into fixed seq_len windows.

    Documents are separated by EOS. No document spans across sequences.
    If a document is longer than seq_len, it is truncated (rare for web text).

    Yields uint16 numpy arrays of shape (seq_len,).
    """
    buffer: List[int] = []
    for tokens in token_stream:
        tokens = tokens[:seq_len - 1]  # leave room for EOS
        doc = tokens + [eos_id]
        if len(buffer) + len(doc) > seq_len:
            # Pad with EOS and yield
            padding = [eos_id] * (seq_len - len(buffer))
            yield np.array(buffer + padding, dtype=np.uint16)
            buffer = []
        buffer.extend(doc)
    # Flush remaining
    if buffer:
        padding = [eos_id] * (seq_len - len(buffer))
        yield np.array(buffer + padding, dtype=np.uint16)


# ── SFT packing ───────────────────────────────────────────────────────────────

CHAT_TEMPLATE = (
    "<|system|>{system}<|end|>\n"
    "<|user|>{user}<|end|>\n"
    "<|assistant|>{assistant}<|end|>\n"
)
SYSTEM_DEFAULT = "You are a helpful assistant."


def format_conversation(example: dict, sp) -> Tuple[List[int], List[int]]:
    """
    Format a conversation example into token IDs and a loss mask.

    Supports both single-turn {"instruction": ..., "output": ...}
    and multi-turn {"messages": [{"role": ..., "content": ...}, ...]} formats.

    Returns:
        tokens: List[int] — token IDs
        mask:   List[int] — 1 for assistant tokens, 0 for system/user tokens
    """
    tokens: List[int] = []
    mask: List[int] = []

    def encode(text: str, is_assistant: bool) -> None:
        ids = sp.encode(text, out_type=int)
        tokens.extend(ids)
        mask.extend([1 if is_assistant else 0] * len(ids))

    if "messages" in example:
        # Multi-turn format
        system = example.get("system", SYSTEM_DEFAULT)
        encode(f"<|system|>{system}<|end|>\n", is_assistant=False)
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                encode(f"<|user|>{content}<|end|>\n", is_assistant=False)
            elif role == "assistant":
                encode(f"<|assistant|>{content}<|end|>\n", is_assistant=True)
            # Skip system messages inside messages list (already handled above)
    else:
        # Single-turn format: {"instruction": ..., "output": ...}
        system = example.get("system", SYSTEM_DEFAULT)
        instruction = example.get("instruction", example.get("input", ""))
        output = example.get("output", example.get("response", ""))
        encode(f"<|system|>{system}<|end|>\n", is_assistant=False)
        encode(f"<|user|>{instruction}<|end|>\n", is_assistant=False)
        encode(f"<|assistant|>{output}<|end|>\n", is_assistant=True)

    return tokens, mask


def pack_sft(
    examples: Iterator[dict],
    sp,
    seq_len: int,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Pack SFT examples into fixed-length sequences.

    Unlike pretrain packing, SFT packs by conversation boundary
    (no cross-conversation packing to avoid leaking context).

    Yields (tokens, loss_mask) tuples of shape (seq_len,).
    """
    eos_id = sp.eos_id()
    buffer_tok: List[int] = []
    buffer_mask: List[int] = []

    for ex in examples:
        tok, msk = format_conversation(ex, sp)
        tok = tok[:seq_len - 1]
        msk = msk[:seq_len - 1]
        doc_tok = tok + [eos_id]
        doc_msk = msk + [1]  # EOS is part of assistant output

        if len(buffer_tok) + len(doc_tok) > seq_len:
            # Flush and pad
            pad_len = seq_len - len(buffer_tok)
            yield (
                np.array(buffer_tok + [eos_id] * pad_len, dtype=np.uint16),
                np.array(buffer_mask + [0] * pad_len, dtype=np.uint8),
            )
            buffer_tok, buffer_mask = [], []

        buffer_tok.extend(doc_tok)
        buffer_mask.extend(doc_msk)

    # Flush remaining
    if buffer_tok:
        pad_len = seq_len - len(buffer_tok)
        yield (
            np.array(buffer_tok + [eos_id] * pad_len, dtype=np.uint16),
            np.array(buffer_mask + [0] * pad_len, dtype=np.uint8),
        )


# ── Shard writer ───────────────────────────────────────────────────────────────

def write_shard(
    sequences: List[np.ndarray],
    output_dir: str,
    shard_idx: int,
    masks: List[np.ndarray] | None = None,
) -> None:
    """Write a shard of packed sequences to GCS as a compressed numpy archive."""
    fname = f"shard_{shard_idx:05d}.npz"
    local_path = f"/tmp/{fname}"

    data = {"tokens": np.stack(sequences)}
    if masks is not None:
        data["loss_mask"] = np.stack(masks)

    np.savez_compressed(local_path, **data)

    if output_dir.startswith("gs://"):
        import subprocess
        gcs_path = os.path.join(output_dir, fname)
        subprocess.run(["gsutil", "-m", "cp", local_path, gcs_path], check=True)
        os.unlink(local_path)
        logger.info(f"Wrote shard {shard_idx} → {gcs_path}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        shutil.move(local_path, os.path.join(output_dir, fname))
        logger.info(f"Wrote shard {shard_idx} → {output_dir}/{fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    p.add_argument("--input_glob", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--shard_size", type=int, default=4096,
                   help="Number of sequences per shard file.")
    p.add_argument("--num_workers", type=int, default=32)
    return p.parse_args()


def expand_glob(pattern: str) -> List[str]:
    """Expand a GCS or local glob pattern to a list of file paths."""
    if pattern.startswith("gs://"):
        import subprocess
        result = subprocess.run(
            ["gsutil", "ls", pattern], capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n")
    else:
        import glob
        return glob.glob(pattern, recursive=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Input: {args.input_glob}")
    logger.info(f"Output: {args.output_dir}")

    sp = load_tokenizer(args.tokenizer_path)
    eos_id = sp.eos_id()

    files = expand_glob(args.input_glob)
    logger.info(f"Found {len(files)} input files.")

    shard_idx = 0
    total_seqs = 0
    shard_tokens: List[np.ndarray] = []
    shard_masks: List[np.ndarray] = []

    if args.mode == "pretrain":
        def token_stream():
            for fpath in files:
                for text in read_jsonl_gz(fpath):
                    yield sp.encode(text, out_type=int)

        for seq in greedy_pack_pretrain(token_stream(), args.seq_len, eos_id):
            shard_tokens.append(seq)
            total_seqs += 1
            if len(shard_tokens) >= args.shard_size:
                write_shard(shard_tokens, args.output_dir, shard_idx)
                shard_idx += 1
                shard_tokens = []
                if total_seqs % 100_000 == 0:
                    logger.info(f"Packed {total_seqs:,} sequences ({total_seqs * args.seq_len / 1e9:.1f}B tokens)")

        if shard_tokens:
            write_shard(shard_tokens, args.output_dir, shard_idx)

    elif args.mode == "sft":
        def example_stream():
            for fpath in files:
                open_fn = gzip.open if fpath.endswith(".gz") else open
                with open_fn(fpath, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

        for tok_arr, msk_arr in pack_sft(example_stream(), sp, args.seq_len):
            shard_tokens.append(tok_arr)
            shard_masks.append(msk_arr)
            total_seqs += 1
            if len(shard_tokens) >= args.shard_size:
                write_shard(shard_tokens, args.output_dir, shard_idx, masks=shard_masks)
                shard_idx += 1
                shard_tokens, shard_masks = [], []

        if shard_tokens:
            write_shard(shard_tokens, args.output_dir, shard_idx, masks=shard_masks)

    logger.info(f"Done. Total sequences: {total_seqs:,} | "
                f"Total tokens: {total_seqs * args.seq_len / 1e9:.2f}B | "
                f"Shards: {shard_idx + 1}")


if __name__ == "__main__":
    main()
