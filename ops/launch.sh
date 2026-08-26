#!/usr/bin/env bash
# ops/launch.sh
# Full end-to-end launch script for kind-archimeds.
# Run this from the kind-archimeds project root.
#
# Usage:
#   bash ops/launch.sh [--dry-run]
#
# Prerequisites:
#   1. MaxText cloned to ./maxtext/  (git clone https://github.com/google/maxtext)
#   2. Patches applied:  bash patches/apply_patches.sh
#   3. Data prepared:    python data/deduplicate.py && python data/tokenize_and_pack.py
#   4. CoT data ready:   python data/generate_cot.py
#   5. GCS bucket configured in configs/*.yml
#   6. TPU v5e-8 provisioned and SSH accessible

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
BUCKET="gs://<YOUR_BUCKET>"
MAXTEXT_DIR="./maxtext"
PRETRAIN_CONFIG="./configs/300m_pretrain.yml"
SFT_CONFIG="./configs/300m_sft.yml"
RUN_DATE=$(date +%Y%m%d_%H%M)
LOG_DIR="./logs/${RUN_DATE}"
TRAINING_START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — commands will be printed but not executed ==="
fi

run() {
    echo ">>> $*"
    if [[ "$DRY_RUN" == "false" ]]; then
        "$@"
    fi
}

mkdir -p "$LOG_DIR"

echo "=========================================="
echo " kind-archimeds — Training Launch"
echo " Start: $TRAINING_START_ISO"
echo "=========================================="

# ── Step 0: Sanity checks ─────────────────────────────────────────────────────
echo ""
echo "[0/4] Pre-flight checks..."

if [[ ! -d "$MAXTEXT_DIR" ]]; then
    echo "ERROR: MaxText not found at $MAXTEXT_DIR"
    echo "       Run: git clone https://github.com/google/maxtext && bash patches/apply_patches.sh"
    exit 1
fi

# Check GCS connectivity
run gsutil ls "${BUCKET}/data/pretrain/packed/" > /dev/null
echo "✓ GCS pretrain data accessible"

run gsutil ls "${BUCKET}/data/sft/packed/" > /dev/null
echo "✓ GCS SFT data accessible"

run gsutil ls "${BUCKET}/tokenizer/mistral_32k.model" > /dev/null
echo "✓ Tokenizer accessible"

echo "✓ Pre-flight passed"

# ── Step 1: Start trigger_decay monitor in background ─────────────────────────
echo ""
echo "[1/4] Starting WSD decay trigger monitor..."

run python ops/trigger_decay.py \
    --run_name "lm300m-pretrain" \
    --bucket "$BUCKET" \
    --budget_hours 18.0 \
    --decay_trigger_hours 10.5 \
    --decay_steps 2500 \
    --poll_interval 60 \
    --start_time_iso "$TRAINING_START_ISO" \
    &> "${LOG_DIR}/trigger_decay.log" &

TRIGGER_PID=$!
echo "✓ Decay trigger running (PID: $TRIGGER_PID)"

# ── Step 2: Start live monitor in background ───────────────────────────────────
echo ""
echo "[2/4] Starting live training monitor..."

run python ops/monitor.py \
    --metrics_path "${BUCKET}/runs/lm300m-pretrain/metrics.jsonl" \
    --poll_interval 30 \
    --alert_loss_spike 0.3 \
    --alert_min_mfu 0.40 \
    &> "${LOG_DIR}/monitor.log" &

MONITOR_PID=$!
echo "✓ Monitor running (PID: $MONITOR_PID)"

# ── Step 3: Pretrain ───────────────────────────────────────────────────────────
echo ""
echo "[3/4] Starting pretrain..."
echo "      Expected duration: ~11 h (budget: 15.5 h)"
echo "      Target tokens:     19B"
echo "      Checkpoints:       every 500 steps → ${BUCKET}/runs/lm300m-pretrain/"

run python "${MAXTEXT_DIR}/MaxText/train.py" \
    "$PRETRAIN_CONFIG" \
    run_name="lm300m-pretrain" \
    steps=72500 \
    2>&1 | tee "${LOG_DIR}/pretrain.log"

PRETRAIN_EXIT=$?
if [[ $PRETRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Pretrain exited with code $PRETRAIN_EXIT"
    echo "       Check ${LOG_DIR}/pretrain.log"
    kill $TRIGGER_PID $MONITOR_PID 2>/dev/null || true
    exit $PRETRAIN_EXIT
fi

echo "✓ Pretrain complete"
kill $TRIGGER_PID 2>/dev/null || true

# ── Step 4: SFT fine-tune ──────────────────────────────────────────────────────
echo ""
echo "[4/4] Starting SFT fine-tune..."
echo "      Expected duration: ~0.6 h"
echo "      Target tokens:     300M"

# Find latest pretrain checkpoint
PRETRAIN_CKPT="${BUCKET}/runs/lm300m-pretrain/checkpoints/$(
    gsutil ls "${BUCKET}/runs/lm300m-pretrain/checkpoints/" \
    | sort | tail -1 | sed 's|/$||' | xargs basename
)"
echo "      Loading checkpoint: $PRETRAIN_CKPT"

run python "${MAXTEXT_DIR}/MaxText/train.py" \
    "$SFT_CONFIG" \
    run_name="lm300m-sft" \
    load_parameters_path="${PRETRAIN_CKPT}" \
    steps=1200 \
    2>&1 | tee "${LOG_DIR}/sft.log"

SFT_EXIT=$?
if [[ $SFT_EXIT -ne 0 ]]; then
    echo "ERROR: SFT exited with code $SFT_EXIT"
    echo "       Check ${LOG_DIR}/sft.log"
    kill $MONITOR_PID 2>/dev/null || true
    exit $SFT_EXIT
fi

echo "✓ SFT complete"
kill $MONITOR_PID 2>/dev/null || true

# ── Done ───────────────────────────────────────────────────────────────────────
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo ""
echo "=========================================="
echo " Training complete!"
echo " End: $END_TIME"
echo " Logs: $LOG_DIR"
echo " Next: bash eval/run_evals.sh"
echo "=========================================="
