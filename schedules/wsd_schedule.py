"""
schedules/wsd_schedule.py
Warmup–Stable–Decay (WSD) learning rate schedule for optax.

Usage in MaxText trainer (train.py):
    from schedules.wsd_schedule import make_wsd_schedule
    lr_schedule = make_wsd_schedule(config)
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clipping),
        optax.adamw(learning_rate=lr_schedule, ...),
    )

The stable_end_step can be overridden at runtime by ops/trigger_decay.py,
which writes a new value to GCS. The schedule polls this value every
`poll_interval_steps` steps and adjusts the decay boundary accordingly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)


@dataclass
class WSDConfig:
    peak_lr: float = 3.0e-3
    min_lr: float = 3.0e-4
    warmup_steps: int = 2000
    stable_end_step: int = 70000    # overridden by trigger_decay.py if running
    decay_steps: int = 2500


def make_wsd_schedule(cfg: WSDConfig) -> optax.Schedule:
    """
    Returns a piecewise WSD schedule:
      [0, warmup_steps)               → linear 0 → peak_lr
      [warmup_steps, stable_end_step) → constant peak_lr
      [stable_end_step, ...)          → cosine decay peak_lr → min_lr

    Args:
        cfg: WSDConfig instance with schedule hyperparameters.

    Returns:
        An optax.Schedule callable: step (int) → learning_rate (float).
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=cfg.peak_lr,
        transition_steps=cfg.warmup_steps,
    )
    stable = optax.constant_schedule(cfg.peak_lr)
    decay = optax.cosine_decay_schedule(
        init_value=cfg.peak_lr,
        decay_steps=cfg.decay_steps,
        alpha=cfg.min_lr / cfg.peak_lr,
    )
    schedule = optax.join_schedules(
        schedules=[warmup, stable, decay],
        boundaries=[cfg.warmup_steps, cfg.stable_end_step],
    )
    return schedule


def make_wsd_schedule_from_maxtext_config(config) -> optax.Schedule:
    """
    Convenience wrapper that reads schedule parameters from a MaxText config object.

    Args:
        config: MaxText config namespace with attributes:
            learning_rate, min_learning_rate, warmup_steps,
            wsd_stable_end_step, wsd_decay_steps

    Returns:
        An optax.Schedule callable.
    """
    cfg = WSDConfig(
        peak_lr=config.learning_rate,
        min_lr=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        stable_end_step=config.wsd_stable_end_step,
        decay_steps=config.wsd_decay_steps,
    )
    return make_wsd_schedule(cfg)


# ── Runtime schedule update (for trigger_decay.py integration) ────────────────

class DynamicWSDSchedule:
    """
    WSD schedule that can have its stable_end_step updated at runtime.

    Wraps the static schedule and rebuilds it whenever trigger_decay.py
    signals a new stable_end_step via a GCS signal file.

    Example:
        dynamic = DynamicWSDSchedule(cfg, gcs_signal_path="gs://bucket/decay_signal.json")
        # In training loop:
        lr = dynamic(current_step)
        if step % 500 == 0:
            dynamic.poll()   # check for updated stable_end_step
    """

    def __init__(self, cfg: WSDConfig, gcs_signal_path: Optional[str] = None):
        self.cfg = cfg
        self.gcs_signal_path = gcs_signal_path
        self._schedule = make_wsd_schedule(cfg)
        self._current_stable_end = cfg.stable_end_step

    def __call__(self, step: int) -> float:
        return self._schedule(step)

    def poll(self) -> None:
        """Check GCS for an updated stable_end_step from trigger_decay.py."""
        if self.gcs_signal_path is None:
            return
        try:
            import subprocess, json
            result = subprocess.run(
                ["gsutil", "cat", self.gcs_signal_path],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                signal = json.loads(result.stdout)
                new_end = signal.get("stable_end_step")
                if new_end and new_end != self._current_stable_end:
                    logger.info(
                        f"WSD: stable_end_step updated {self._current_stable_end} → {new_end}"
                    )
                    self._current_stable_end = new_end
                    self.cfg.stable_end_step = new_end
                    self._schedule = make_wsd_schedule(self.cfg)
        except Exception as e:
            logger.warning(f"WSD poll failed (non-fatal): {e}")
