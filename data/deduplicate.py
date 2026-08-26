"""
data/deduplicate.py
MinHash LSH deduplication across all pretraining sources.

Uses the `datatrove` library for scalable, streaming deduplication.
Runs on CPU VMs (not TPU), typically 2–4 hours for 19B tokens.

Usage:
    python data/deduplicate.py \
        --input_base gs://<YOUR_BUCKET>/data/raw \
        --output_base gs://<YOUR_BUCKET>/data/deduped \
        --num_workers 64 \
        --threshold 0.8

Prerequisites:
    pip install datatrove[gcs] datasets

Estimate: ~$15–25 in GCE n2-standard-64 time for 19B tokens.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MinHash LSH deduplication for pretraining corpus.")
    p.add_argument("--input_base", required=True,
                   help="GCS or local path to raw data directory.")
    p.add_argument("--output_base", required=True,
                   help="GCS or local path for deduplicated output.")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Jaccard similarity threshold for duplicate detection.")
    p.add_argument("--num_perm", type=int, default=128,
                   help="Number of MinHash permutations (higher = more accurate, slower).")
    p.add_argument("--shingle_size", type=int, default=13,
                   help="N-gram shingle size for MinHash.")
    p.add_argument("--num_workers", type=int, default=64,
                   help="Number of parallel workers.")
    p.add_argument("--sources", nargs="+",
                   default=["fineweb_edu", "stack_lite", "wikipedia", "openwebmath", "casual_web"],
                   help="Dataset source names to deduplicate.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print plan without executing.")
    return p.parse_args()


def build_pipeline(args: argparse.Namespace):
    """
    Build a datatrove deduplication pipeline.

    Pipeline stages:
      1. JsonlReader — read raw JSONL from GCS per source
      2. MinHashDeduplicator — compute and filter duplicates
      3. JsonlWriter — write deduplicated output to GCS

    The MinHashDeduplicator is run in two passes:
      Pass 1 (minhash_stage): compute MinHash signatures → build LSH index
      Pass 2 (dedup_stage):   filter documents flagged as duplicates
    """
    try:
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers import JsonlWriter
        from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupFilter
        from datatrove.executor.local import LocalPipelineExecutor
        from datatrove.executor.slurm import SlurmPipelineExecutor
    except ImportError:
        raise ImportError(
            "datatrove is required: pip install datatrove[gcs]\n"
            "See https://github.com/huggingface/datatrove"
        )

    minhash_dir = f"{args.output_base}/.minhash_index"

    # Stage 1: Compute MinHash signatures
    sig_pipeline = [
        JsonlReader(
            data_folder=args.input_base,
            text_key="text",
            glob_pattern="**/*.jsonl.gz",
        ),
        MinhashDedupSignature(
            output_folder=minhash_dir,
            num_buckets=14,            # 14 bands × 9 rows ≈ threshold 0.8
            hashes_per_bucket=9,
            n_grams=args.shingle_size,
            num_permutations=args.num_perm,
        ),
    ]

    # Stage 2: Build LSH buckets
    bucket_pipeline = [
        MinhashDedupBuckets(
            input_folder=minhash_dir,
            output_folder=f"{minhash_dir}/buckets",
            num_buckets=14,
        ),
    ]

    # Stage 3: Filter duplicates and write output
    filter_pipeline = [
        JsonlReader(
            data_folder=args.input_base,
            text_key="text",
            glob_pattern="**/*.jsonl.gz",
        ),
        MinhashDedupFilter(
            input_folder=f"{minhash_dir}/buckets",
            exclusion_writer=JsonlWriter(f"{args.output_base}/duplicates/"),
        ),
        JsonlWriter(
            output_folder=args.output_base,
            output_filename="${rank}.jsonl.gz",
            compression="gzip",
        ),
    ]

    return sig_pipeline, bucket_pipeline, filter_pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    logger.info(f"Deduplication plan:")
    logger.info(f"  Input:     {args.input_base}")
    logger.info(f"  Output:    {args.output_base}")
    logger.info(f"  Threshold: {args.threshold}")
    logger.info(f"  Workers:   {args.num_workers}")
    logger.info(f"  Sources:   {args.sources}")

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    sig_pipeline, bucket_pipeline, filter_pipeline = build_pipeline(args)

    # Run locally (or swap LocalPipelineExecutor for SlurmPipelineExecutor on a cluster)
    try:
        from datatrove.executor.local import LocalPipelineExecutor
    except ImportError:
        raise

    logger.info("Stage 1/3: Computing MinHash signatures...")
    LocalPipelineExecutor(
        pipeline=sig_pipeline,
        tasks=args.num_workers,
        workers=args.num_workers,
        logging_dir=f"{args.output_base}/logs/sig",
    ).run()

    logger.info("Stage 2/3: Building LSH buckets...")
    LocalPipelineExecutor(
        pipeline=bucket_pipeline,
        tasks=14,
        workers=14,
        logging_dir=f"{args.output_base}/logs/buckets",
    ).run()

    logger.info("Stage 3/3: Filtering duplicates and writing output...")
    LocalPipelineExecutor(
        pipeline=filter_pipeline,
        tasks=args.num_workers,
        workers=args.num_workers,
        logging_dir=f"{args.output_base}/logs/filter",
    ).run()

    logger.info(f"Deduplication complete. Output at: {args.output_base}")


if __name__ == "__main__":
    main()
