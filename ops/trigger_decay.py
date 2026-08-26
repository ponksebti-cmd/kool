"""
ops/trigger_decay.py
Wall-clock triggered WSD decay signal.

Monitors elapsed wall-clock time and writes a GCS signal file when
the 18-hour budget is ~45 minutes from exhaustion, instructing the
trainer (via DynamicWSDSchedule.poll()) to enter the decay phase.

Run this as a background process alongside the trainer:
    python ops/trigger_decay.py \
        --run_name lm300m-pretrain \
        --bucket gs://<YOUR_BUCKET> \
        --budget_hours 18.0 \
        --decay_trigger_hours 10.5 \
        --decay_steps 2500 \
        --poll_interval 60 &

The trainer polls the GCS signal file every 500 steps (see DynamicWSDSchedule).
When the signal is found, the trainer rebuilds its LR schedule with the new
stable_end_step = current_step, entering cosine decay for decay_steps steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WSD decay trigger monitor.")
    p.add_argument("--run_name", required=True)
    p.add_argument("--bucket", required=True, help="GCS bucket root, e.g. gs://my-bucket")
    p.add_argument("--budget_hours", type=float, default=18.0,
                   help="Total training wall-clock budget in hours.")
    p.add_argument("--decay_trigger_hours", type=float, default=10.5,
                   help="Wall-clock hour at which to trigger decay.")
    p.add_argument("--decay_steps", type=int, default=2500,
                   help="Number of steps for the decay phase.")
    p.add_argument("--poll_interval", type=int, default=60,
                   help="How often to check wall clock, in seconds.")
    p.add_argument("--start_time_iso", default=None,
                   help="ISO 8601 training start time. If not set, uses current time.")
    return p.parse_args()


def write_signal(gcs_path: str, payload: dict) -> bool:
    """Write a JSON signal file to GCS. Returns True on success."""
    content = json.dumps(payload, indent=2)
    try:
        result = subprocess.run(
            ["gsutil", "cp", "-", gcs_path],
            input=content.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"Signal written to {gcs_path}: {payload}")
            return True
        else:
            logger.error(f"gsutil cp failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        logger.error(f"Failed to write signal: {e}")
        return False


def read_metrics(gcs_metrics_path: str) -> dict:
    """Read the latest metrics JSON line from GCS to get current step."""
    try:
        result = subprocess.run(
            ["gsutil", "cat", gcs_metrics_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l]
            if lines:
                return json.loads(lines[-1])
    except Exception:
        pass
    return {}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    if args.start_time_iso:
        start_time = datetime.fromisoformat(args.start_time_iso)
    else:
        start_time = datetime.now(tz=timezone.utc)
        logger.info(f"Training start time set to now: {start_time.isoformat()}")

    signal_path = f"{args.bucket}/runs/{args.run_name}/decay_signal.json"
    metrics_path = f"{args.bucket}/runs/{args.run_name}/metrics.jsonl"

    triggered = False
    logger.info(
        f"Monitoring run '{args.run_name}'. "
        f"Decay trigger at hour {args.decay_trigger_hours:.1f} / "
        f"budget {args.budget_hours:.1f}h."
    )

    while not triggered:
        now = datetime.now(tz=timezone.utc)
        elapsed_hours = (now - start_time).total_seconds() / 3600.0

        logger.info(f"Elapsed: {elapsed_hours:.2f}h / trigger at {args.decay_trigger_hours:.1f}h")

        if elapsed_hours >= args.decay_trigger_hours:
            # Read current step from metrics to set a precise stable_end_step
            metrics = read_metrics(metrics_path)
            current_step = metrics.get("step", None)

            payload = {
                "event": "decay_trigger",
                "trigger_time_iso": now.isoformat(),
                "elapsed_hours": elapsed_hours,
                "decay_steps": args.decay_steps,
                "stable_end_step": current_step,  # None = trainer uses current step
                "reason": f"wall_clock >= {args.decay_trigger_hours}h",
            }

            success = write_signal(signal_path, payload)
            if success:
                triggered = True
                logger.info(
                    f"✓ Decay triggered at step {current_step} "
                    f"(elapsed: {elapsed_hours:.2f}h). "
                    f"Trainer will decay over {args.decay_steps} steps."
                )
            else:
                logger.warning("Signal write failed — will retry next poll.")

        if not triggered:
            time.sleep(args.poll_interval)

    logger.info("trigger_decay.py exiting normally.")


if __name__ == "__main__":
    main()
