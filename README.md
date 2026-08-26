# kind-archimeds

**300M-parameter instruction-following transformer**
JAX/MaxText · TPU v5e-8 · 18-hour wall-clock budget

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Layers | 24 |
| Hidden size | 1024 |
| Attention heads | 16 (GQA, 4 KV heads) |
| Head dim | 64 |
| FFN intermediate | 2816 (SwiGLU) |
| Vocab | 32,000 (tied embeddings) |
| Context length | 2048 |
| Parameters | ~300M |
| Precision | bf16 |

**Upgrades over baseline:** QK-Norm, parallel attn+FFN blocks, logit soft-capping,
z-loss, WSD learning rate schedule.

---

## Directory Structure

```
kind-archimeds/
├── configs/
│   ├── 300m_pretrain.yml      # MaxText pretrain config
│   └── 300m_sft.yml           # MaxText SFT config
├── maxtext_patches/
│   └── parallel_attn_ffn.patch  # Patch for parallel attention+FFN
├── data/
│   ├── deduplicate.py         # MinHash LSH deduplication
│   ├── tokenize_and_pack.py   # Tokenize + sequence packing
│   ├── generate_cot.py        # Teacher CoT distillation
│   ├── dataset_config.py      # Dataset weights & sources
│   └── cot_seeds.jsonl        # Seed prompts for CoT generation
├── schedules/
│   └── wsd_schedule.py        # Warmup-Stable-Decay LR schedule
├── patches/
│   └── apply_patches.sh       # Apply all MaxText patches
├── ops/
│   ├── trigger_decay.py       # Wall-clock WSD decay trigger
│   ├── monitor.py             # Live training monitor
│   └── launch.sh              # Full launch script
├── eval/
│   ├── run_evals.sh           # lm-evaluation-harness runner
│   ├── spot_check.py          # Manual 20-prompt spot check
│   └── eval_targets.md        # Expected score targets
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clone and patch MaxText
git clone https://github.com/google/maxtext.git
cd maxtext && bash ../patches/apply_patches.sh && cd ..

# 3. Prepare data (run 4-6 hours before TPU start)
python data/deduplicate.py --config data/dataset_config.py
python data/tokenize_and_pack.py --seq_len 2048

# 4. Generate CoT distillation data (parallel, CPU)
python data/generate_cot.py --seeds data/cot_seeds.jsonl

# 5. Launch training
bash ops/launch.sh
```

---

## Timeline

| Phase | Duration | Steps |
|---|---|---|
| Warmup | ~18 min | 2,000 |
| Stable pretrain | ~10.5 h | ~68,000 |
| WSD decay | ~23 min | 2,500 |
| SFT fine-tune | ~0.6 h | ~1,200 |
| **Total** | **~11.5–12 h** | — |

4.5 h of margin for JIT warmup, GCS I/O, restarts.
