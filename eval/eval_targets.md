# Evaluation Targets

Expected scores for the kind-archimeds 300M SFT model.
Calibrated for a ~63 tokens/parameter pretrain run (19B tokens / 300M params).

> [!NOTE]
> These targets reflect a fast prototype model, NOT a benchmark-competitive one.
> Comparable open models achieving higher scores train 10–150× more tokens.

## Automated Benchmarks

| Benchmark | Shots | Target | Rationale |
|---|---|---|---|
| HellaSwag | 0 | ≥ 65% | Basic commonsense / sentence completion |
| ARC-Easy | 0 | ≥ 62% | Elementary science Q&A |
| PIQA | 0 | ≥ 72% | Physical commonsense reasoning |
| Winogrande | 0 | ≥ 63% | Pronoun/reference disambiguation |
| GSM8K | 4 | ≥ 5% | Math reasoning — low bar; model can attempt steps |
| HumanEval pass@1 | 0 | ≥ 5% | Basic code — syntax bar, not correctness |
| Perplexity (Pile test) | — | ≤ 12.0 | Language model quality sanity check |

## Context: Comparable Models

| Model | Params | Tokens | HellaSwag |
|---|---|---|---|
| **kind-archimeds (ours)** | 300M | 19B | ~65% (target) |
| TinyLlama-1.1B | 1.1B | 3T | ~59% |
| OPT-350M | 350M | 180B | ~54% |
| GPT-2 | 117M | ~100B | ~52% |

> We achieve our target at 19B tokens primarily because of FineWeb-Edu quality filtering,
> not raw scale. A raw CommonCrawl run at the same token count would score ~5pp lower.

## Spot Check (Human Eval)

20 fixed prompts across 7 categories. See `eval/spot_check.py`.

| Category | # Prompts | Pass Criterion |
|---|---|---|
| Format compliance | 2 | Correct template; no repetition loops |
| Single-clause instructions | 4 | Correct on ≥ 3/4 |
| Multi-turn (context) | 2 | Retains context in both turns |
| CoT format | 3 | Shows sequential steps in ≥ 2/3 |
| Known limitations | 4 | Document behavior (not a pass/fail) |
| Code | 2 | Valid syntax in ≥ 1/2 |
| Diverse | 3 | Sensible in ≥ 2/3 |

## Divergence Indicators (Training — not post-hoc)

Watch these live during training. If any alert fires, act immediately:

| Metric | Alert Threshold | Action |
|---|---|---|
| Loss spike (single step) | Δ > 0.3 | Roll back 3 checkpoints; reduce LR 20% |
| Z-loss | > 1e-2 | Reduce LR or increase soft-cap threshold |
| Grad norm (consistently at clip) | ≥ 1.0 for >50 steps | Reduce LR 10% |
| MFU | < 40% | Check data pipeline / GCS I/O |
| Loss after warmup | > 5.0 at step 5,000 | Possible config error; inspect and restart |

## What to Declare Done

The run is done and passes evaluation when ALL of:

- [ ] Loss curves are smooth through stable + decay phases (no divergence)
- [ ] Perplexity ≤ 12.0 on Pile test
- [ ] HellaSwag ≥ 65%, ARC-Easy ≥ 62%, PIQA ≥ 72%, Winogrande ≥ 63%
- [ ] Spot check: format compliance passes on both prompts
- [ ] Spot check: single-clause instructions pass ≥ 3/4
- [ ] Total wall-clock ≤ 18 hours (pretrain + SFT)

Known limitations (limit_* category in spot check) are **documented, not fixed**.
They reflect the model's inherent capability ceiling at this scale and compute budget.
