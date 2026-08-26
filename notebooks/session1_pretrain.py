# %% [markdown]
# # kind-archimeds — Session 1: Pretrain
# **TPU v5e-8 · ~9 hours · Saves checkpoint every 15 minutes**
#
# **Before running:**
# 1. Settings → Accelerator → TPU v5e-8
# 2. Settings → Internet → On
# 3. Run All
#
# **When session ends:** Go to the Output tab, click "New Dataset" to save
# `/kaggle/working/checkpoints/` — you'll load it in Session 2.

# %% [markdown]
# ## 1. Install Dependencies

# %%
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("--upgrade", "flax", "optax")
pip("datasets==2.19.0")
pip("transformers==4.41.0")
pip("sentencepiece==0.2.0")

print("✓ Dependencies installed")

# %% [markdown]
# ## 2. Imports & TPU Setup

# %%
import os, json, time, pickle, shutil, math, threading, queue
from functools import partial
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax import struct

# Verify TPU
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
assert len(jax.devices()) == 8, f"Expected 8 TPU chips, got {len(jax.devices())}"
print("✓ TPU v5e-8 confirmed (8 chips)")

# bf16 matmul for TPU
jax.config.update("jax_default_matmul_precision", "bfloat16")

# %% [markdown]
# ## 3. Model Configuration

# %%
@dataclass
class ModelConfig:
    # Architecture (locked — do not change)
    vocab_size:      int   = 32000
    n_layers:        int   = 24
    d_model:         int   = 1024
    n_heads:         int   = 16       # query heads
    n_kv_heads:      int   = 4        # key/value heads (GQA)
    head_dim:        int   = 64       # d_model / n_heads
    ffn_dim:         int   = 2816     # SwiGLU intermediate
    max_seq_len:     int   = 2048
    rope_base:       float = 10000.0
    norm_eps:        float = 1e-6
    # Stability upgrades
    logit_soft_cap:  float = 50.0     # tanh soft-cap on output logits
    z_loss_weight:   float = 1e-4     # auxiliary z-loss
    use_qk_norm:     bool  = True     # normalize Q and K before dot product
    parallel_attn_ffn: bool = True    # PaLM-style parallel blocks
    # Training
    dtype:           Any   = jnp.bfloat16

CFG = ModelConfig()

# Sanity check param count
def count_params(n_layers, d_model, n_heads, n_kv_heads, head_dim, ffn_dim, vocab_size):
    embed    = vocab_size * d_model                          # tied
    attn     = n_layers * (d_model * d_model +               # Q proj
                           2 * d_model * (n_kv_heads * head_dim) +  # K, V proj
                           d_model * d_model)                # O proj
    ffn      = n_layers * (d_model * ffn_dim * 2 +           # gate + up
                           ffn_dim * d_model)                # down
    norms    = n_layers * 2 * d_model + d_model              # pre-norm + final
    return embed + attn + ffn + norms

approx_params = count_params(CFG.n_layers, CFG.d_model, CFG.n_heads,
                              CFG.n_kv_heads, CFG.head_dim, CFG.ffn_dim, CFG.vocab_size)
print(f"Approximate parameter count: {approx_params/1e6:.1f}M")

# %% [markdown]
# ## 4. Model Definition

# %%
# ── RMSNorm ───────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        scale = scale.astype(jnp.float32)
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.epsilon)
        normed = (x_f32 / rms) * scale
        return normed.astype(self.dtype)


# ── RoPE ──────────────────────────────────────────────────────────────────────
def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute cos/sin tables for RoPE. Returns [max_seq_len, head_dim//2]."""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)          # [seq, head_dim//2]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return jnp.array(cos), jnp.array(sin)  # [seq, head_dim//2]

ROPE_COS, ROPE_SIN = precompute_rope_freqs(CFG.head_dim, CFG.max_seq_len, CFG.rope_base)

def apply_rope(x, cos, sin):
    """
    Apply RoPE to x of shape [batch, heads, seq, head_dim].
    cos/sin shape: [seq, head_dim//2]
    """
    # Reshape cos/sin for broadcasting: [1, 1, seq, head_dim//2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., ::2]   # even dims
    x2 = x[..., 1::2]  # odd dims
    # Rotate: (x1 + ix2) * (cos + i*sin)
    x_rot_even = x1 * cos - x2 * sin
    x_rot_odd  = x1 * sin + x2 * cos
    # Interleave back
    return jnp.stack([x_rot_even, x_rot_odd], axis=-1).reshape(x.shape)


# ── Grouped-Query Attention with QK-Norm ──────────────────────────────────────
class GQAttention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask=None):
        cfg = self.config
        B, T, C = x.shape
        dtype = cfg.dtype

        # Projections
        q = nn.Dense(cfg.n_heads * cfg.head_dim,     use_bias=False, dtype=dtype, name="q_proj")(x)
        k = nn.Dense(cfg.n_kv_heads * cfg.head_dim,  use_bias=False, dtype=dtype, name="k_proj")(x)
        v = nn.Dense(cfg.n_kv_heads * cfg.head_dim,  use_bias=False, dtype=dtype, name="v_proj")(x)

        # Reshape: [B, T, heads, head_dim] → [B, heads, T, head_dim]
        q = q.reshape(B, T, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(0, 2, 1, 3)

        # QK-Norm (normalize before RoPE and dot product)
        if cfg.use_qk_norm:
            q_scale = self.param("q_norm_scale", nn.initializers.ones,
                                 (cfg.n_heads, 1, cfg.head_dim))
            k_scale = self.param("k_norm_scale", nn.initializers.ones,
                                 (cfg.n_kv_heads, 1, cfg.head_dim))
            q = _rms_norm_heads(q, q_scale, cfg.norm_eps)
            k = _rms_norm_heads(k, k_scale, cfg.norm_eps)

        # RoPE
        cos = ROPE_COS[:T]
        sin = ROPE_SIN[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Repeat K,V to match Q head count (GQA expansion)
        n_rep = cfg.n_heads // cfg.n_kv_heads   # = 4
        k = jnp.repeat(k, n_rep, axis=1)        # [B, n_heads, T, head_dim]
        v = jnp.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        scale = cfg.head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale  # [B, heads, T, T]

        # Causal mask
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn = jnp.where(causal[None, None, :, :], attn, jnp.finfo(dtype).min)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(dtype)

        # Attend and project
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)          # [B, heads, T, head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)      # [B, T, C]
        out = nn.Dense(C, use_bias=False, dtype=dtype, name="o_proj")(out)
        return out

def _rms_norm_heads(x, scale, eps):
    """Per-head RMS normalization. x: [B, heads, T, head_dim]"""
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * scale).astype(x.dtype)


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        dtype = cfg.dtype
        gate = nn.Dense(cfg.ffn_dim, use_bias=False, dtype=dtype, name="gate_proj")(x)
        up   = nn.Dense(cfg.ffn_dim, use_bias=False, dtype=dtype, name="up_proj")(x)
        x    = jax.nn.silu(gate) * up
        x    = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, name="down_proj")(x)
        return x


# ── Transformer Block (Parallel Attn + FFN) ───────────────────────────────────
class TransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        normed = RMSNorm(epsilon=cfg.norm_eps, dtype=cfg.dtype, name="pre_norm")(x)

        if cfg.parallel_attn_ffn:
            # PaLM-style: both branches see same normed input, sum into residual
            attn_out = GQAttention(cfg, name="attention")(normed)
            ffn_out  = SwiGLUFFN(cfg, name="ffn")(normed)
            x = x + attn_out + ffn_out
        else:
            # Sequential (fallback)
            x = x + GQAttention(cfg, name="attention")(normed)
            x = x + SwiGLUFFN(cfg, name="ffn")(
                RMSNorm(epsilon=cfg.norm_eps, dtype=cfg.dtype, name="post_attn_norm")(x)
            )
        return x


# ── Full Transformer ───────────────────────────────────────────────────────────
class Transformer(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids):
        cfg = self.config
        B, T = input_ids.shape

        # Token embedding
        embed = self.param(
            "token_embed",
            nn.initializers.normal(stddev=0.02),
            (cfg.vocab_size, cfg.d_model),
        )
        x = embed[input_ids].astype(cfg.dtype)   # [B, T, d_model]

        # Transformer blocks
        for i in range(cfg.n_layers):
            x = TransformerBlock(cfg, name=f"block_{i}")(x)

        # Final norm
        x = RMSNorm(epsilon=cfg.norm_eps, dtype=cfg.dtype, name="final_norm")(x)

        # Output logits — tied to input embedding (no extra param)
        logits = x @ embed.T                     # [B, T, vocab_size]
        logits = logits.astype(jnp.float32)

        # Logit soft-cap: prevents explosion at high LR
        if cfg.logit_soft_cap > 0:
            logits = cfg.logit_soft_cap * jnp.tanh(logits / cfg.logit_soft_cap)

        return logits

# %% [markdown]
# ## 5. Tokenizer

# %%
from transformers import AutoTokenizer

print("Loading tokenizer (Mistral 32k vocab)...")
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    use_fast=True,
)
assert tokenizer.vocab_size == 32000 == CFG.vocab_size
EOS_ID = tokenizer.eos_token_id
print(f"✓ Tokenizer loaded. Vocab: {tokenizer.vocab_size}, EOS: {EOS_ID}")

# %% [markdown]
# ## 6. Data Pipeline

# %%
from datasets import load_dataset

# ── Dataset mix (pretrain) ────────────────────────────────────────────────────
# We stream FineWeb-Edu as the primary source.
# Wikipedia and OpenWebMath supplement for diversity.
# Weights: FineWeb-Edu 75%, Wikipedia 15%, OpenWebMath 10%

PRETRAIN_SOURCES = [
    {
        "name": "fineweb_edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "weight": 0.75,
    },
    {
        "name": "wikipedia",
        "repo": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "weight": 0.15,
    },
    {
        "name": "openwebmath",
        "repo": "open-web-math/open-web-math",
        "subset": None,
        "split": "train",
        "weight": 0.10,
    },
]

def make_dataset_stream(source: dict):
    """Make a streaming HuggingFace dataset."""
    if source["subset"]:
        return load_dataset(source["repo"], source["subset"],
                            split=source["split"], streaming=True, trust_remote_code=True)
    return load_dataset(source["repo"], split=source["split"],
                        streaming=True, trust_remote_code=True)

def greedy_pack_generator(ds_streams, weights, tokenizer, seq_len, eos_id):
    """
    Interleave streams by weight and greedily pack into fixed-length sequences.
    Yields numpy arrays of shape [seq_len].
    """
    rng = np.random.default_rng(42)
    iters = [iter(s) for s in ds_streams]
    weights = np.array(weights)
    weights = weights / weights.sum()

    buffer = []
    while True:
        # Sample which stream to pull from
        src_idx = int(rng.choice(len(iters), p=weights))
        try:
            example = next(iters[src_idx])
        except StopIteration:
            # Restart exhausted streams
            iters[src_idx] = iter(ds_streams[src_idx])
            example = next(iters[src_idx])

        text = example.get("text", "") or example.get("content", "")
        if not text.strip():
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Truncate very long docs to avoid buffer bloat
        tokens = tokens[: seq_len * 2]
        doc = tokens + [eos_id]

        buffer.extend(doc)

        while len(buffer) >= seq_len:
            yield np.array(buffer[:seq_len], dtype=np.int32)
            buffer = buffer[seq_len:]


def make_batch_loader(seq_len, per_device_batch, n_devices, queue_maxsize=8):
    """
    Returns an iterator that yields [n_devices, per_device_batch, seq_len] int32 arrays.
    Runs the data generator in a background thread.
    """
    global_batch = n_devices * per_device_batch

    print("Initialising dataset streams...")
    streams = [make_dataset_stream(s) for s in PRETRAIN_SOURCES]
    weights = [s["weight"] for s in PRETRAIN_SOURCES]
    gen = greedy_pack_generator(streams, weights, tokenizer, seq_len, EOS_ID)
    print("✓ Streams ready")

    q = queue.Queue(maxsize=queue_maxsize)

    def fill():
        batch_seqs = []
        for seq in gen:
            batch_seqs.append(seq)
            if len(batch_seqs) == global_batch:
                arr = np.stack(batch_seqs, axis=0)          # [global_batch, seq_len]
                arr = arr.reshape(n_devices, per_device_batch, seq_len)
                q.put(arr)
                batch_seqs = []

    t = threading.Thread(target=fill, daemon=True)
    t.start()

    while True:
        yield q.get()

# %% [markdown]
# ## 7. WSD Learning Rate Schedule

# %%
def make_wsd_schedule(
    peak_lr: float = 3e-3,
    min_lr:  float = 3e-4,
    warmup_steps: int = 2000,
    stable_end_step: int = 68000,
    decay_steps: int = 2500,
) -> optax.Schedule:
    """Warmup → Stable → Cosine Decay."""
    warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
    stable = optax.constant_schedule(peak_lr)
    decay  = optax.cosine_decay_schedule(peak_lr, decay_steps, alpha=min_lr / peak_lr)
    return optax.join_schedules(
        [warmup, stable, decay],
        boundaries=[warmup_steps, stable_end_step],
    )

# Session 1 covers ~68,000 pretrain steps (≈ 17.8B tokens at 262k tok/step).
# Decay is NOT triggered in Session 1 — that happens in Session 2.
# We set stable_end_step far beyond Session 1's step count.
SESSION1_TOTAL_STEPS = 68000
WSD_STABLE_END = 999999  # decay not triggered in this session

schedule = make_wsd_schedule(
    peak_lr=3e-3,
    min_lr=3e-4,
    warmup_steps=2000,
    stable_end_step=WSD_STABLE_END,
    decay_steps=2500,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=schedule,
        b1=0.9, b2=0.95, eps=1e-8,
        weight_decay=0.1,
        mask=lambda p: jax.tree_util.tree_map(
            lambda x: x.ndim >= 2, p  # only apply wd to matrices
        ),
    ),
)

# %% [markdown]
# ## 8. Checkpointing Utilities
# **Saves every 15 minutes. Keeps 3 rolling checkpoints. Atomic writes.**

# %%
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
CHECKPOINT_INTERVAL = 900  # seconds (15 minutes)
KEEP_N_CHECKPOINTS = 3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def _save_pytree(pytree, filepath):
    """Save a JAX/numpy pytree to a .npz file + treedef pickle. Atomic write."""
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    np_leaves = [np.array(leaf) for leaf in leaves]
    tmp_path = filepath + ".tmp"
    np.savez_compressed(tmp_path, *np_leaves)
    # Save treedef alongside
    with open(filepath + "_treedef.pkl", "wb") as f:
        pickle.dump(treedef, f)
    # Atomic rename
    os.replace(tmp_path, filepath)


def _load_pytree(filepath, template=None):
    """Load a JAX pytree from .npz + treedef pickle."""
    with np.load(filepath, allow_pickle=False) as data:
        np_leaves = [data[f"arr_{i}"] for i in range(len(data.files))]
    with open(filepath + "_treedef.pkl", "rb") as f:
        treedef = pickle.load(f)
    leaves = [jnp.array(leaf) for leaf in np_leaves]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def save_checkpoint(state, step, metrics: dict):
    """
    Save a checkpoint to CHECKPOINT_DIR/step_XXXXXXX/.
    Uses atomic file writes (write temp → rename) to prevent corruption
    if the session is killed mid-save.
    Rolling window: keeps KEEP_N_CHECKPOINTS most recent.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:07d}")
    os.makedirs(ckpt_path, exist_ok=True)

    print(f"\n[CKPT] Saving checkpoint at step {step}...", flush=True)
    t0 = time.time()

    # Unreplicate: pull from device 0 (all devices have same params after pmean grad sync)
    params   = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
    opt_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.opt_state))

    # Atomic saves
    _save_pytree(params,    os.path.join(ckpt_path, "params.npz"))
    _save_pytree(opt_state, os.path.join(ckpt_path, "opt_state.npz"))

    # Metadata
    meta = {
        "step": step,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - TRAIN_START_TIME,
    }
    meta_path = os.path.join(ckpt_path, "meta.json")
    meta_tmp  = meta_path + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, meta_path)

    # Update latest pointer (atomic)
    latest_tmp = os.path.join(CHECKPOINT_DIR, "latest.txt.tmp")
    with open(latest_tmp, "w") as f:
        f.write(f"step_{step:07d}")
    os.replace(latest_tmp, os.path.join(CHECKPOINT_DIR, "latest.txt"))

    elapsed_save = time.time() - t0
    print(f"[CKPT] ✓ Saved in {elapsed_save:.1f}s → {ckpt_path}", flush=True)

    # Cleanup old checkpoints
    _cleanup_old_checkpoints(step)

    return ckpt_path


def _cleanup_old_checkpoints(current_step):
    """Delete checkpoints older than the KEEP_N most recent."""
    all_ckpts = sorted([
        d for d in os.listdir(CHECKPOINT_DIR)
        if d.startswith("step_") and
           os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
    ])
    to_delete = all_ckpts[:-KEEP_N_CHECKPOINTS]
    for d in to_delete:
        full = os.path.join(CHECKPOINT_DIR, d)
        shutil.rmtree(full, ignore_errors=True)
        print(f"[CKPT]   Deleted old checkpoint: {d}", flush=True)


def load_latest_checkpoint(checkpoint_dir, state_template):
    """
    Load the latest checkpoint and return (params, opt_state, step).
    state_template must have the same pytree structure as the saved state.
    """
    latest_file = os.path.join(checkpoint_dir, "latest.txt")
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    with open(latest_file) as f:
        latest = f.read().strip()

    ckpt_path = os.path.join(checkpoint_dir, latest)

    with open(os.path.join(ckpt_path, "meta.json")) as f:
        meta = json.load(f)

    params    = _load_pytree(os.path.join(ckpt_path, "params.npz"))
    opt_state = _load_pytree(os.path.join(ckpt_path, "opt_state.npz"))

    print(f"[CKPT] ✓ Loaded checkpoint: step {meta['step']} from {ckpt_path}")
    return params, opt_state, meta["step"], meta

# %% [markdown]
# ## 9. Loss Function & Train Step

# %%
def compute_loss(logits, input_ids, z_loss_weight=1e-4):
    """
    Cross-entropy next-token prediction + z-loss.
    logits: [B, T, vocab]   (already soft-capped in model)
    input_ids: [B, T]
    """
    # Shift: predict token t+1 from token t
    targets = input_ids[:, 1:]        # [B, T-1]
    logits  = logits[:, :-1, :]      # [B, T-1, vocab]

    # Standard cross-entropy
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # [B, T-1]
    ce_loss = jnp.mean(xent)

    # Z-loss: penalises log-partition function magnitude
    log_z = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)  # [B, T-1]
    z_loss = z_loss_weight * jnp.mean(log_z ** 2)

    total = ce_loss + z_loss
    return total, {"loss": ce_loss, "z_loss": z_loss, "total_loss": total}


@partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def train_step(state, batch):
    """
    One gradient step, executed in parallel across all 8 TPU chips.
    batch: [per_device_batch, seq_len] int32
    Returns: (new_state, metrics)
    """
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch)
        loss, metrics = compute_loss(logits, batch, z_loss_weight=CFG.z_loss_weight)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # All-reduce gradients across devices
    grads   = jax.lax.pmean(grads,   axis_name="devices")
    metrics = jax.lax.pmean(metrics, axis_name="devices")

    # Gradient norm (for monitoring)
    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics

# %% [markdown]
# ## 10. Initialise Model & Training State

# %%
PER_DEVICE_BATCH = 16    # 16 seqs × 8 chips = 128 global batch
N_DEVICES = len(jax.devices())
GLOBAL_BATCH = PER_DEVICE_BATCH * N_DEVICES
TOKENS_PER_STEP = GLOBAL_BATCH * CFG.max_seq_len
print(f"Global batch: {GLOBAL_BATCH} seqs | {TOKENS_PER_STEP:,} tokens/step")

# Dummy input for init
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, CFG.max_seq_len), dtype=jnp.int32)

model = Transformer(CFG)
print("Initialising model parameters...")
params = model.init(key, dummy_input)["params"]

# Count actual params
flat_params = jax.tree_util.tree_leaves(params)
n_params = sum(p.size for p in flat_params)
print(f"✓ Model initialised: {n_params/1e6:.1f}M parameters")

# Build TrainState
class TrainState(train_state.TrainState):
    pass

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)

# Replicate across all 8 TPU devices
state = jax.device_put_replicated(state, jax.devices())
print("✓ State replicated across 8 TPU chips")

# %% [markdown]
# ## 11. Training Loop (with 15-minute checkpointing)

# %%
TRAIN_START_TIME = time.time()
last_checkpoint_time = TRAIN_START_TIME
last_log_time = TRAIN_START_TIME
LOG_INTERVAL = 50       # log every 50 steps
SESSION_BUDGET_HOURS = 8.75  # stop slightly before Kaggle's 9-hour cutoff

print("Starting data loader...")
data_iter = make_batch_loader(
    seq_len=CFG.max_seq_len,
    per_device_batch=PER_DEVICE_BATCH,
    n_devices=N_DEVICES,
)
# Warm up data loader (wait for first batch)
print("Warming up data pipeline (may take 30–60s)...")
first_batch = next(data_iter)
print(f"✓ First batch shape: {first_batch.shape}")

# JIT warmup (compile train_step on first real batch)
print("Compiling train_step (first step may take 2–5 minutes)...")
compile_start = time.time()
batch_device = jax.device_put_sharded(list(first_batch), jax.devices())
state, metrics = train_step(state, batch_device)
jax.block_until_ready(metrics)
compile_time = time.time() - compile_start
print(f"✓ Compilation done in {compile_time:.1f}s")

# ── Main training loop ────────────────────────────────────────────────────────
start_step = 0
tokens_trained = 0
total_steps = SESSION1_TOTAL_STEPS

print(f"\n{'='*60}")
print(f" Training: steps {start_step} → {total_steps}")
print(f" Checkpoint every: 15 minutes")
print(f" Session budget: {SESSION_BUDGET_HOURS}h")
print(f"{'='*60}\n")

for step in range(start_step, total_steps):
    # Check session budget FIRST — save and exit if near cutoff
    elapsed_hours = (time.time() - TRAIN_START_TIME) / 3600
    if elapsed_hours >= SESSION_BUDGET_HOURS:
        print(f"\n[SESSION] {elapsed_hours:.2f}h elapsed — approaching Kaggle cutoff.")
        print("[SESSION] Saving final checkpoint and stopping gracefully...")
        m = {k: float(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))[k])
             for k in metrics}
        save_checkpoint(state, step, m)
        print(f"[SESSION] ✓ Stopped at step {step}. Total tokens: {tokens_trained/1e9:.2f}B")
        break

    # Get batch and put on devices
    batch_np = next(data_iter)
    batch_device = jax.device_put_sharded(list(batch_np), jax.devices())

    # Train step
    state, metrics = train_step(state, batch_device)
    tokens_trained += TOKENS_PER_STEP

    # ── 15-minute checkpoint ──────────────────────────────────────────────────
    time_since_ckpt = time.time() - last_checkpoint_time
    if time_since_ckpt >= CHECKPOINT_INTERVAL:
        # Block until current step is done before saving
        jax.block_until_ready(metrics)
        m = {k: float(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))[k])
             for k in metrics}
        m["tokens_trained"] = tokens_trained
        m["elapsed_hours"] = (time.time() - TRAIN_START_TIME) / 3600
        save_checkpoint(state, step, m)
        last_checkpoint_time = time.time()

    # ── Logging ───────────────────────────────────────────────────────────────
    if time.time() - last_log_time >= 20:
        jax.block_until_ready(metrics)
        m = {k: float(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))[k])
             for k in metrics}
        elapsed = time.time() - TRAIN_START_TIME
        tok_per_sec = tokens_trained / elapsed if elapsed > 0 else 0
        tokens_per_param = tokens_trained / approx_params if approx_params > 0 else 0
        lr = float(schedule(step))
        print(
            f"step {step:6d} | "
            f"loss={m['loss']:.4f} | "
            f"z={m['z_loss']:.2e} | "
            f"gnorm={m.get('grad_norm', 0):.3f} | "
            f"lr={lr:.2e} | "
            f"{tok_per_sec:,.0f} tok/s | "
            f"{tokens_per_param:.2f} tok/param | "
            f"{tokens_trained/1e9:.2f}B tok | "
            f"{elapsed/3600:.2f}h",
            flush=True,
        )
        last_log_time = time.time()

# ── Final checkpoint at end of session ────────────────────────────────────────
print("\n[SESSION] Training loop complete. Saving final checkpoint...")
jax.block_until_ready(metrics)
m = {k: float(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))[k])
     for k in metrics}
m["tokens_trained"] = tokens_trained
m["elapsed_hours"] = (time.time() - TRAIN_START_TIME) / 3600
save_checkpoint(state, step, m)

# %% [markdown]
# ## 12. Session 1 Done
#
# **What to do next:**
# 1. Go to the **Output** tab of this notebook
# 2. Click **"New Dataset"** → name it `kind-archimeds-ckpt-s1`
# 3. Wait for it to finish saving (a few minutes)
# 4. Open Session 2 notebook → add `kind-archimeds-ckpt-s1` as an Input Dataset
# 5. Run Session 2

# %%
print("\n" + "="*60)
print(" SESSION 1 COMPLETE")
print(f" Steps trained:  {step:,}")
print(f" Tokens trained: {tokens_trained/1e9:.2f}B")
print(f" Wall clock:     {(time.time()-TRAIN_START_TIME)/3600:.2f}h")
print(f" Checkpoints at: {CHECKPOINT_DIR}")
print()
print(" NEXT: Output tab → New Dataset → 'kind-archimeds-ckpt-s1'")
print("="*60)

# List saved checkpoints
for entry in sorted(os.listdir(CHECKPOINT_DIR)):
    entry_path = os.path.join(CHECKPOINT_DIR, entry)
    if os.path.isdir(entry_path):
        meta_path = os.path.join(entry_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            size_mb = sum(
                os.path.getsize(os.path.join(entry_path, f))
                for f in os.listdir(entry_path)
            ) / 1e6
            print(f"  {entry}: step={meta['step']}, loss={meta['metrics'].get('loss', '?'):.4f}, {size_mb:.0f}MB")
