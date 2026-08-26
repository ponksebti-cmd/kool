#!/usr/bin/env bash
# eval/run_evals.sh
# Run automated evaluations on the final SFT checkpoint.
#
# Usage:
#   bash eval/run_evals.sh \
#       --model_path gs://<BUCKET>/runs/lm300m-sft/checkpoints/final/ \
#       --output_dir ./eval_results/
#
# Prerequisites:
#   pip install lm-eval[api]          # lm-evaluation-harness
#   pip install evaluate sacrebleu    # optional extra metrics
#
# Converts GCS checkpoint → local HuggingFace-format for lm-eval compatibility.
# Expected runtime: 30–60 minutes on a CPU VM.

set -euo pipefail

MODEL_PATH="${1:-}"
OUTPUT_DIR="${2:-./eval_results}"
HF_MODEL_DIR="/tmp/lm300m_hf"

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_path) MODEL_PATH="$2"; shift 2 ;;
            --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
}
parse_args "$@"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: bash eval/run_evals.sh --model_path <path> [--output_dir <dir>]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "============================================"
echo " kind-archimeds — Evaluation Suite"
echo " Model: $MODEL_PATH"
echo " Output: $OUTPUT_DIR"
echo "============================================"

# ── Step 1: Convert checkpoint to HuggingFace format ─────────────────────────
echo ""
echo "[1/3] Converting checkpoint to HuggingFace format..."
echo "      (MaxText → HF using MaxText's export script)"

if [[ "$MODEL_PATH" == gs://* ]]; then
    gsutil -m cp -r "${MODEL_PATH}" "${HF_MODEL_DIR}_maxtext/"
    LOCAL_CKPT="${HF_MODEL_DIR}_maxtext"
else
    LOCAL_CKPT="$MODEL_PATH"
fi

# Use MaxText's built-in HuggingFace export
python ./maxtext/MaxText/export_maxtext_to_huggingface.py \
    --checkpoint_path "$LOCAL_CKPT" \
    --model_config ./configs/300m_pretrain.yml \
    --output_dir "$HF_MODEL_DIR" \
    --tokenizer_path /tmp/mistral_32k.model \
    || {
        echo "⚠ MaxText HF export failed. Check export script compatibility."
        echo "  Alternatively, use MaxText's inference server with lm-eval --model maxtext"
        exit 1
    }

echo "✓ Checkpoint exported to $HF_MODEL_DIR"

# ── Step 2: Run lm-evaluation-harness benchmarks ─────────────────────────────
echo ""
echo "[2/3] Running lm-evaluation-harness..."

TASKS="hellaswag,arc_easy,piqa,winogrande,gsm8k"

lm_eval \
    --model hf \
    --model_args "pretrained=${HF_MODEL_DIR},dtype=bfloat16" \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size 32 \
    --output_path "${OUTPUT_DIR}/lm_eval_results.json" \
    --log_samples \
    2>&1 | tee "${OUTPUT_DIR}/lm_eval.log"

# GSM8K with 4-shot CoT (separate run)
lm_eval \
    --model hf \
    --model_args "pretrained=${HF_MODEL_DIR},dtype=bfloat16" \
    --tasks gsm8k \
    --num_fewshot 4 \
    --batch_size 8 \
    --output_path "${OUTPUT_DIR}/gsm8k_4shot.json" \
    2>&1 | tee "${OUTPUT_DIR}/gsm8k_4shot.log"

echo "✓ lm-eval benchmarks complete"

# ── Step 3: Perplexity on held-out Pile test set ─────────────────────────────
echo ""
echo "[3/3] Computing perplexity on Pile test set..."

python eval/perplexity.py \
    --model_path "$HF_MODEL_DIR" \
    --dataset_name "EleutherAI/the_pile_deduplicated" \
    --dataset_split "test" \
    --n_examples 2000 \
    --seq_len 2048 \
    --output "${OUTPUT_DIR}/perplexity.json" \
    2>&1 | tee "${OUTPUT_DIR}/perplexity.log"

echo "✓ Perplexity computed"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Evaluation Summary"
echo "============================================"
python3 - <<PYEOF
import json, sys

targets = {
    "hellaswag": ("HellaSwag (0-shot)", 0.65),
    "arc_easy":  ("ARC-Easy  (0-shot)", 0.62),
    "piqa":      ("PIQA      (0-shot)", 0.72),
    "winogrande":("Winogrande(0-shot)", 0.63),
    "gsm8k":     ("GSM8K     (4-shot)", 0.05),
}

try:
    with open("${OUTPUT_DIR}/lm_eval_results.json") as f:
        results = json.load(f)["results"]
    print(f"{'Benchmark':<25} {'Score':>8} {'Target':>8} {'Status':>8}")
    print("-" * 55)
    for key, (label, target) in targets.items():
        if key in results:
            score = results[key].get("acc,none", results[key].get("acc", 0))
            status = "✓ PASS" if score >= target else "✗ MISS"
            print(f"{label:<25} {score:>8.3f} {target:>8.3f} {status:>8}")
except Exception as e:
    print(f"Could not parse results: {e}")

try:
    with open("${OUTPUT_DIR}/perplexity.json") as f:
        ppl = json.load(f)["perplexity"]
    status = "✓ PASS" if ppl <= 12.0 else "✗ MISS"
    print(f"{'Perplexity (Pile)':<25} {ppl:>8.2f} {'<= 12.0':>8} {status:>8}")
except Exception as e:
    print(f"Could not parse perplexity: {e}")
PYEOF

echo ""
echo "Full results: ${OUTPUT_DIR}/"
echo "Next: python eval/spot_check.py --model_path ${HF_MODEL_DIR}"
