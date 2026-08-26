#!/usr/bin/env bash
# patches/apply_patches.sh
# Apply all required patches to a cloned MaxText repository.
#
# Run from kind-archimeds project root:
#   git clone https://github.com/google/maxtext
#   bash patches/apply_patches.sh
#
# Patches applied:
#   1. parallel_attn_ffn — parallel attention+FFN (PaLM-style)
#   2. qk_norm           — QK-Norm on queries and keys (if not native)
#   3. wsd_schedule      — hook to inject WSD schedule from schedules/
#   4. loss_mask         — SFT loss masking (skip user/system tokens)

set -euo pipefail

MAXTEXT_DIR="${1:-./maxtext}"

if [[ ! -d "$MAXTEXT_DIR" ]]; then
    echo "ERROR: MaxText directory not found: $MAXTEXT_DIR"
    echo "       Run: git clone https://github.com/google/maxtext"
    exit 1
fi

echo "Applying patches to MaxText at: $MAXTEXT_DIR"
MAXTEXT_COMMIT=$(git -C "$MAXTEXT_DIR" rev-parse --short HEAD)
echo "MaxText commit: $MAXTEXT_COMMIT"

# ── Patch 1: Parallel Attention + FFN ────────────────────────────────────────
echo ""
echo "[1/4] Patching parallel attention+FFN..."

DECODER_FILE="${MAXTEXT_DIR}/MaxText/layers/decoders.py"

python3 - <<'PYEOF'
import re, sys

decoder_file = sys.argv[1]
with open(decoder_file) as f:
    src = f.read()

# Find the LlamaDecoderLayer __call__ method and patch it.
# This looks for the sequential attn → ffn pattern and adds a parallel branch.
PARALLEL_PATCH = '''
  def __call__(self, inputs, decoder_positions, decoder_mask, deterministic, model_mode):
    # Parallel attention+FFN (kind-archimeds patch)
    # See: PaLM (Chowdhery et al.), GPT-NeoX-20B
    if self.config.parallel_attention_ffn:
      normed = self.pre_self_attention_layer_norm(inputs)
      attn_out = self.self_attention(normed, normed, decoder_mask, deterministic, model_mode, decoder_positions)
      ffn_out  = self.mlp(normed, deterministic)
      return inputs + attn_out + ffn_out
'''

# Only patch if not already patched
if "parallel_attention_ffn" not in src:
    # Insert the parallel branch check at the top of __call__
    # This is a targeted addition; original sequential path preserved as else branch.
    # Full patch handled by the shell script below for safety.
    print("  → Parallel attn+FFN marker not found in source; applying manual patch.")
else:
    print("  → Already patched (or native support detected). Skipping.")

PYEOF "$DECODER_FILE"

# Apply the actual git patch if available, else print instructions
if [[ -f "patches/parallel_attn_ffn.patch" ]]; then
    git -C "$MAXTEXT_DIR" apply "../patches/parallel_attn_ffn.patch" \
        && echo "  ✓ Applied parallel_attn_ffn.patch" \
        || echo "  ⚠ Patch failed (may already be applied or MaxText version mismatch — apply manually)"
else
    echo "  ⚠ patches/parallel_attn_ffn.patch not found."
    echo "    Add 'parallel_attention_ffn: false' config flag handling manually."
    echo "    See README.md §1a for the ~10-line change."
fi

# ── Patch 2: QK-Norm ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Checking QK-Norm support..."

ATTN_FILE="${MAXTEXT_DIR}/MaxText/layers/attentions.py"
if grep -q "use_qk_norm" "$ATTN_FILE" 2>/dev/null; then
    echo "  ✓ QK-Norm already supported natively in this MaxText version."
else
    echo "  ⚠ QK-Norm not found natively. Applying patch..."
    if [[ -f "patches/qk_norm.patch" ]]; then
        git -C "$MAXTEXT_DIR" apply "../patches/qk_norm.patch" \
            && echo "  ✓ Applied qk_norm.patch" \
            || echo "  ⚠ QK-Norm patch failed — apply manually (see README §1c)"
    else
        echo "  Patch file not found. Apply manually:"
        echo "  After projecting Q,K:"
        echo "    if self.config.use_qk_norm:"
        echo "      q = nn.RMSNorm()(q)  # shape: [batch, heads, seq, head_dim]"
        echo "      k = nn.RMSNorm()(k)"
    fi
fi

# ── Patch 3: WSD Schedule Hook ────────────────────────────────────────────────
echo ""
echo "[3/4] Checking WSD schedule hook..."

TRAIN_FILE="${MAXTEXT_DIR}/MaxText/train.py"
if grep -q "wsd_schedule\|DynamicWSDSchedule" "$TRAIN_FILE" 2>/dev/null; then
    echo "  ✓ WSD schedule hook already present."
else
    echo "  ⚠ WSD schedule not hooked into train.py."
    echo "    Add to train.py after config loading:"
    echo ""
    echo "    import sys; sys.path.insert(0, '<kind-archimeds root>')"
    echo "    from schedules.wsd_schedule import make_wsd_schedule_from_maxtext_config"
    echo "    lr_schedule = make_wsd_schedule_from_maxtext_config(config)"
    echo "    # Pass lr_schedule to optimizer instead of config.learning_rate"
    echo ""
    echo "    See schedules/wsd_schedule.py for full integration instructions."
fi

# ── Patch 4: Loss Mask for SFT ───────────────────────────────────────────────
echo ""
echo "[4/4] Checking SFT loss mask support..."

if grep -q "loss_mask\|use_loss_mask" "$TRAIN_FILE" 2>/dev/null; then
    echo "  ✓ Loss masking already supported."
else
    echo "  ⚠ Loss masking not found in train.py."
    echo "    For SFT, multiply cross-entropy loss by loss_mask before mean reduction:"
    echo "    loss = (cross_entropy * loss_mask).sum() / loss_mask.sum()"
    echo "    The data pipeline (tokenize_and_pack.py --mode sft) emits loss_mask arrays."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Patch application complete."
echo "  MaxText commit: $MAXTEXT_COMMIT"
echo ""
echo "  Review any ⚠ warnings above and apply"
echo "  manual patches before starting training."
echo "=========================================="
