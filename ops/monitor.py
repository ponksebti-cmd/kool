"""
ops/monitor.py
Live training monitor — watches GCS metrics and prints a summary dashboard.

Logs: loss, grad_norm, z_loss, learning_rate, mfu, tokens_per_second.
Alerts if divergence (loss spike), low MFU, or LR schedule anomalies detected.

Usage:
    python ops/monitor.py \
        --metrics_path "gs://<BUCKET>/runs/lm300m-pretrain/metrics.jsonl" \
        --poll_interval 30 \
        --alert_loss_spike 0.3 \
        --alert_min_mfu 0.40

Runs in the background alongside training. Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections import deque
from typing import Deque, Optional

logger = logging.getLogger(__name__)

ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training metrics monitor.")
    p.add_argument("--metrics_path", required=True, help="GCS path to metrics.jsonl")
    p.add_argument("--poll_interval", type=int, default=30)
    p.add_argument("--alert_loss_spike", type=float, default=0.3,
                   help="Alert if loss increases by more than this amount in one log step.")
    p.add_argument("--alert_min_mfu", type=float, default=0.40,
                   help="Alert if MFU drops below this threshold.")
    p.add_argument("--alert_z_loss_max", type=float, default=1e-2,
                   help="Alert if z_loss exceeds this threshold.")
    p.add_argument("--window", type=int, default=20,
                   help="Rolling window size for smoothed metrics.")
    return p.parse_args()


def fetch_metrics(gcs_path: str, last_line_count: int = 0) -> list[dict]:
    """Fetch new metric lines from GCS since last_line_count."""
    try:
        result = subprocess.run(
            ["gsutil", "cat", gcs_path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
        lines = [l for l in result.stdout.strip().split("\n") if l]
        new_lines = lines[last_line_count:]
        records = []
        for line in new_lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
    except Exception as e:
        logger.debug(f"Fetch failed: {e}")
        return []


def colorize(value: float, good_threshold: float, bad_threshold: float,
             higher_is_better: bool = True) -> str:
    """Color a float value green/yellow/red based on thresholds."""
    if higher_is_better:
        if value >= good_threshold:
            color = ANSI_GREEN
        elif value >= bad_threshold:
            color = ANSI_YELLOW
        else:
            color = ANSI_RED
    else:
        if value <= good_threshold:
            color = ANSI_GREEN
        elif value <= bad_threshold:
            color = ANSI_YELLOW
        else:
            color = ANSI_RED
    return f"{color}{value:.4f}{ANSI_RESET}"


def print_dashboard(metrics: dict, smoothed: dict, alerts: list[str]) -> None:
    step = metrics.get("step", "?")
    loss = metrics.get("loss", float("nan"))
    lr = metrics.get("learning_rate", float("nan"))
    grad_norm = metrics.get("grad_norm", float("nan"))
    z_loss = metrics.get("z_loss", float("nan"))
    mfu = metrics.get("mfu", float("nan"))
    tok_s = metrics.get("tokens_per_second", float("nan"))

    print(f"\n{'─'*60}")
    print(f"{ANSI_BOLD}Step {step}{ANSI_RESET}  |  "
          f"Loss: {colorize(loss, 2.5, 4.0, higher_is_better=False)}  |  "
          f"LR: {lr:.2e}  |  "
          f"GradNorm: {grad_norm:.3f}")
    print(f"  Z-Loss: {colorize(z_loss, 1e-4, 1e-2, higher_is_better=False)}  |  "
          f"MFU: {colorize(mfu, 0.50, 0.40)}  |  "
          f"Tok/s: {tok_s:,.0f}")
    if smoothed:
        s_loss = smoothed.get("loss", float("nan"))
        print(f"  Smoothed loss (window): {s_loss:.4f}")
    if alerts:
        print(f"\n{ANSI_RED}{ANSI_BOLD}⚠ ALERTS:{ANSI_RESET}")
        for alert in alerts:
            print(f"  {ANSI_RED}→ {alert}{ANSI_RESET}")


def detect_alerts(
    records: Deque[dict],
    latest: dict,
    args: argparse.Namespace,
    prev_loss: Optional[float],
) -> list[str]:
    alerts = []
    loss = latest.get("loss")
    mfu = latest.get("mfu")
    z_loss = latest.get("z_loss")
    grad_norm = latest.get("grad_norm")

    if loss and prev_loss and (loss - prev_loss) > args.alert_loss_spike:
        alerts.append(
            f"Loss spike: {prev_loss:.4f} → {loss:.4f} "
            f"(Δ={loss-prev_loss:.4f}). "
            "Consider rolling back 3 checkpoints and reducing LR by 20%."
        )
    if mfu and mfu < args.alert_min_mfu:
        alerts.append(
            f"Low MFU: {mfu:.3f} < {args.alert_min_mfu}. "
            "Check data pipeline — possible GCS I/O bottleneck."
        )
    if z_loss and z_loss > args.alert_z_loss_max:
        alerts.append(
            f"Z-loss high: {z_loss:.2e} > {args.alert_z_loss_max:.2e}. "
            "Logit explosion risk — reduce LR or increase soft-cap."
        )
    if grad_norm and grad_norm >= 1.0:
        alerts.append(
            f"Grad norm at clip threshold ({grad_norm:.3f}). "
            "LR may be too high."
        )
    return alerts


def rolling_mean(records: Deque[dict], key: str) -> Optional[float]:
    vals = [r.get(key) for r in records if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()

    print(f"{ANSI_BOLD}kind-archimeds training monitor{ANSI_RESET}")
    print(f"Watching: {args.metrics_path}")
    print(f"Poll interval: {args.poll_interval}s | Window: {args.window} steps")

    line_count = 0
    window: Deque[dict] = deque(maxlen=args.window)
    prev_loss: Optional[float] = None

    while True:
        new_records = fetch_metrics(args.metrics_path, line_count)
        if new_records:
            line_count += len(new_records)
            for rec in new_records:
                window.append(rec)

            latest = new_records[-1]
            smoothed = {
                "loss": rolling_mean(window, "loss"),
            }
            alerts = detect_alerts(window, latest, args, prev_loss)
            print_dashboard(latest, smoothed, alerts)
            prev_loss = latest.get("loss", prev_loss)
        else:
            print(".", end="", flush=True)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
