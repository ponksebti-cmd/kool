# %% [markdown]
# # kind-archimeds — Session 2: Decay + SFT
# **TPU v5e-8 · ~3 hours**
#
# **Before running:**
# 1. Settings → Accelerator → TPU v5e-8
# 2. Settings → Internet → On
# 3. **Input Data** → Add Dataset → select your saved `kind-archimeds-ckpt-s1`
# 4. Run All
#
# **What this does:**
# 1. Loads the latest checkpoint from Session 1.
# 2. Runs the **WSD Decay phase** (~2500 steps) on pretraining data to cleanly land the model.
# 3. Switches to **SFT phase** (~1200 steps) on instruction data with a new low learning rate.
# 4. Saves the final SFT model.

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
# ## 2. Imports, TPU Setup & Architecture
# *(Same definitions as Session 1)*

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
assert len(jax.devices()) == 8, f"Expected 8 TPU chips, got {len(jax.devices())}"
jax.config.update("jax_default_matmul_precision", "bfloat16")

@dataclass
class ModelConfig:
    vocab_size:      int   = 32000
    n_layers:        int   = 24
    d_model:         int   = 1024
    n_heads:         int   = 16
    n_kv_heads:      int   = 4
    head_dim:        int   = 64
    ffn_dim:         int   = 2816
    max_seq_len:     int   = 2048
    rope_base:       float = 10000.0
    norm_eps:        float = 1e-6
    logit_soft_cap:  float = 50.0
    z_loss_weight:   float = 1e-4
    use_qk_norm:     bool  = True
    parallel_attn_ffn: bool = True
    dtype:           Any   = jnp.bfloat16

CFG = ModelConfig()

# ── Model Code (identical to Session 1) ───────────────────────────────────────
class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        scale = scale.astype(jnp.float32)
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((x_f32 / rms) * scale).astype(self.dtype)

def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return jnp.array(np.cos(freqs)), jnp.array(np.sin(freqs))

ROPE_COS, ROPE_SIN = precompute_rope_freqs(CFG.head_dim, CFG.max_seq_len, CFG.rope_base)

def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot_even = x1 * cos - x2 * sin
    x_rot_odd  = x1 * sin + x2 * cos
    return jnp.stack([x_rot_even, x_rot_odd], axis=-1).reshape(x.shape)

def _rms_norm_qk(x, scale, eps):
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * scale).astype(x.dtype)

_rms_norm_heads = _rms_norm_qk  # alias

class GQAttention(nn.Module):
    config: ModelConfig
    @nn.compact
    def __call__(self, x, mask=None):
        cfg = self.config
        B, T, C = x.shape
        dtype = cfg.dtype

        q = nn.Dense(cfg.n_heads * cfg.head_dim,    use_bias=False, dtype=dtype, name="q_proj")(x)
        k = nn.Dense(cfg.n_kv_heads * cfg.head_dim, use_bias=False, dtype=dtype, name="k_proj")(x)
        v = nn.Dense(cfg.n_kv_heads * cfg.head_dim, use_bias=False, dtype=dtype, name="v_proj")(x)

        # Flash-attention native layout: [B, T, heads, head_dim]
        q = q.reshape(B, T, cfg.n_heads,    cfg.head_dim)
        k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim)

        if cfg.use_qk_norm:
            q_scale = self.param("q_norm_scale", nn.initializers.ones, (cfg.n_heads, cfg.head_dim))
            k_scale = self.param("k_norm_scale", nn.initializers.ones, (cfg.n_kv_heads, cfg.head_dim))
            q = _rms_norm_qk(q, q_scale, cfg.norm_eps)
            k = _rms_norm_qk(k, k_scale, cfg.norm_eps)

        # RoPE in [B, heads, T, head_dim], then back
        q_h = q.transpose(0, 2, 1, 3)
        k_h = k.transpose(0, 2, 1, 3)
        q_h = apply_rope(q_h, ROPE_COS[:T], ROPE_SIN[:T])
        k_h = apply_rope(k_h, ROPE_COS[:T], ROPE_SIN[:T])
        q = q_h.transpose(0, 2, 1, 3)
        k = k_h.transpose(0, 2, 1, 3)

        # GQA expand
        n_rep = cfg.n_heads // cfg.n_kv_heads
        k = jnp.repeat(k, n_rep, axis=2)  # [B, T, n_heads, head_dim]
        v = jnp.repeat(v, n_rep, axis=2)

        # Flash attention — never materialises O(seq²) matrix
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)  # [B, T, n_heads, head_dim]
        out = out.reshape(B, T, C)
        return nn.Dense(C, use_bias=False, dtype=dtype, name="o_proj")(out)

class SwiGLUFFN(nn.Module):
    config: ModelConfig
    @nn.compact
    def __call__(self, x):
        cfg = self.config
        gate = nn.Dense(cfg.ffn_dim, use_bias=False, dtype=cfg.dtype, name="gate_proj")(x)
        up   = nn.Dense(cfg.ffn_dim, use_bias=False, dtype=cfg.dtype, name="up_proj")(x)
        x    = jax.nn.silu(gate) * up
        return nn.Dense(cfg.d_model, use_bias=False, dtype=cfg.dtype, name="down_proj")(x)

class TransformerBlock(nn.Module):
    config: ModelConfig
    @nn.compact
    def __call__(self, x):
        normed = RMSNorm(epsilon=self.config.norm_eps, dtype=self.config.dtype, name="pre_norm")(x)
        if self.config.parallel_attn_ffn:
            attn_out = GQAttention(self.config, name="attention")(normed)
            ffn_out  = SwiGLUFFN(self.config, name="ffn")(normed)
            x = x + attn_out + ffn_out
        else:
            x = x + GQAttention(self.config, name="attention")(normed)
            x = x + SwiGLUFFN(self.config, name="ffn")(
                RMSNorm(epsilon=self.config.norm_eps, dtype=self.config.dtype, name="post_attn_norm")(x)
            )
        return x

class Transformer(nn.Module):
    config: ModelConfig
    @nn.compact
    def __call__(self, input_ids):
        cfg = self.config
        B, T = input_ids.shape
        embed = self.param("token_embed", nn.initializers.normal(stddev=0.02), (cfg.vocab_size, cfg.d_model))
        x = embed[input_ids].astype(cfg.dtype)
        RematBlock = nn.remat(TransformerBlock, prevent_cse=False)
        for i in range(cfg.n_layers):
            x = RematBlock(cfg, name=f"block_{i}")(x)
        x = RMSNorm(epsilon=cfg.norm_eps, dtype=cfg.dtype, name="final_norm")(x)
        logits = (x @ embed.T).astype(jnp.float32)
        if cfg.logit_soft_cap > 0:
            logits = cfg.logit_soft_cap * jnp.tanh(logits / cfg.logit_soft_cap)
        return logits


# %% [markdown]
# ## 3. Checkpoint Loading (from Session 1)

# %%
def _load_pytree(filepath):
    """Load a JAX pytree from .npz + treedef pickle."""
    with np.load(filepath, allow_pickle=False) as data:
        np_leaves = [data[f"arr_{i}"] for i in range(len(data.files))]
    with open(filepath + "_treedef.pkl", "rb") as f:
        treedef = pickle.load(f)
    leaves = [jnp.array(leaf) for leaf in np_leaves]
    return jax.tree_util.tree_unflatten(treedef, leaves)

def find_checkpoint_dataset():
    """Find the mounted Session 1 checkpoint dataset in /kaggle/input/."""
    input_dir = "/kaggle/input/"
    if not os.path.exists(input_dir):
        # Local fallback if not on Kaggle
        return "/kaggle/working/checkpoints"

    datasets = os.listdir(input_dir)
    # Filter to likely names
    candidates = [d for d in datasets if "ckpt" in d.lower() or "checkpoint" in d.lower()]
    if not candidates:
        raise ValueError(f"Could not find a checkpoint dataset in {input_dir}. Did you Add Dataset?")

    # Pick the most recently modified if multiple
    best_candidate = candidates[0]
    best_path = os.path.join(input_dir, best_candidate)

    latest_file = os.path.join(best_path, "latest.txt")
    if not os.path.exists(latest_file):
        raise ValueError(f"Found {best_path} but it lacks a latest.txt file.")

    return best_path

def load_latest_checkpoint(checkpoint_dir):
    latest_file = os.path.join(checkpoint_dir, "latest.txt")
    with open(latest_file) as f:
        latest = f.read().strip()
    ckpt_path = os.path.join(checkpoint_dir, latest)

    with open(os.path.join(ckpt_path, "meta.json")) as f:
        meta = json.load(f)

    print(f"Loading checkpoint from {ckpt_path} (step {meta['step']})...")
    t0 = time.time()
    params = _load_pytree(os.path.join(ckpt_path, "params.npz"))
    opt_state = _load_pytree(os.path.join(ckpt_path, "opt_state.npz"))
    print(f"✓ Loaded in {time.time()-t0:.1f}s")
    return params, opt_state, meta

# Init model structure
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, CFG.max_seq_len), dtype=jnp.int32)
model = Transformer(CFG)
_ = model.init(key, dummy_input)

class TrainState(train_state.TrainState):
    pass

CKPT_INPUT_DIR = find_checkpoint_dataset()
print(f"Using checkpoint dataset: {CKPT_INPUT_DIR}")
loaded_params, loaded_opt_state, loaded_meta = load_latest_checkpoint(CKPT_INPUT_DIR)
SESSION_1_STEP = loaded_meta["step"]

# %% [markdown]
# ## 4. Decay Phase (Finish Pretraining)
# **Target:** ~2500 steps cosine decay to safely conclude pretraining.

# %%
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
EOS_ID = tokenizer.eos_token_id

# Same pretrain streams as Session 1 (restarted, but randomness means different batches)
PRETRAIN_SOURCES = [
    {"repo": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train", "weight": 0.75},
    {"repo": "wikimedia/wikipedia", "subset": "20231101.en", "split": "train", "weight": 0.15},
    {"repo": "open-web-math/open-web-math", "subset": None, "split": "train", "weight": 0.10},
]

def greedy_pack_generator(ds_streams, weights, tokenizer, seq_len, eos_id):
    rng = np.random.default_rng(SESSION_1_STEP) # Use step as seed so it's different from Session 1
    iters = [iter(s) for s in ds_streams]
    weights = np.array(weights) / np.array(weights).sum()
    buffer = []
    while True:
        src_idx = int(rng.choice(len(iters), p=weights))
        try:
            example = next(iters[src_idx])
        except StopIteration:
            iters[src_idx] = iter(ds_streams[src_idx])
            example = next(iters[src_idx])

        text = example.get("text", "") or example.get("content", "")
        if not text.strip(): continue
        tokens = tokenizer.encode(text, add_special_tokens=False)[: seq_len * 2] + [eos_id]
        buffer.extend(tokens)
        while len(buffer) >= seq_len:
            yield np.array(buffer[:seq_len], dtype=np.int32)
            buffer = buffer[seq_len:]

def make_batch_loader(streams, seq_len, per_device_batch, n_devices):
    global_batch = n_devices * per_device_batch
    weights = [s["weight"] for s in PRETRAIN_SOURCES]
    gen = greedy_pack_generator(streams, weights, tokenizer, seq_len, EOS_ID)
    q = queue.Queue(maxsize=4)
    def fill():
        batch_seqs = []
        for seq in gen:
            batch_seqs.append(seq)
            if len(batch_seqs) == global_batch:
                arr = np.stack(batch_seqs, axis=0).reshape(n_devices, per_device_batch, seq_len)
                q.put(arr)
                batch_seqs = []
    t = threading.Thread(target=fill, daemon=True)
    t.start()
    while True: yield q.get()

# ── Decay Schedule ────────────────────────────────────────────────────────────
DECAY_STEPS = 2500

# We create a schedule that *starts* at the decay phase, pretending we are at step 0
decay_schedule = optax.cosine_decay_schedule(init_value=3e-3, decay_steps=DECAY_STEPS, alpha=0.1)

decay_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=decay_schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
)

state_decay = TrainState.create(
    apply_fn=model.apply,
    params=loaded_params,
    tx=decay_optimizer,
)
# Force opt_state to be the loaded one (but step count might mismatch if we reset schedule to 0)
# To be safe, we just use the loaded opt_state and let the schedule run from step 0 of this session.
state_decay = state_decay.replace(opt_state=loaded_opt_state, step=0)
import flax
state_decay = flax.jax_utils.replicate(state_decay)

def compute_loss(logits, input_ids, z_loss_weight=1e-4):
    targets = input_ids[:, 1:]
    logits  = logits[:, :-1, :]
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    ce_loss = jnp.mean(xent)
    log_z = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
    z_loss = z_loss_weight * jnp.mean(log_z ** 2)
    return ce_loss + z_loss, {"loss": ce_loss, "z_loss": z_loss}

@partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def train_step_pretrain(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch)
        return compute_loss(logits, batch, CFG.z_loss_weight)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads   = jax.lax.pmean(grads,   axis_name="devices")
    metrics = jax.lax.pmean(metrics, axis_name="devices")
    metrics["grad_norm"] = optax.global_norm(grads)
    return state.apply_gradients(grads=grads), metrics

# Run decay loop
print("\n" + "="*60)
print(f" Starting Decay Phase ({DECAY_STEPS} steps)")
print("="*60)

pretrain_streams = [
    load_dataset(s["repo"], s["subset"], split=s["split"], streaming=True, trust_remote_code=True)
    if s["subset"] else
    load_dataset(s["repo"], split=s["split"], streaming=True, trust_remote_code=True)
    for s in PRETRAIN_SOURCES
]
decay_loader = make_batch_loader(pretrain_streams, CFG.max_seq_len, 8, 8)

last_decay_log_time = time.time()
decay_start_time = time.time()
tokens_decay_trained = 0
DECAY_TOKENS_PER_STEP = 16 * 8 * CFG.max_seq_len
APPROX_PARAMS = 300 * 10**6

for step in range(DECAY_STEPS):
    batch = next(decay_loader)
    state_decay, metrics = train_step_pretrain(state_decay, batch)
    tokens_decay_trained += DECAY_TOKENS_PER_STEP

    if time.time() - last_decay_log_time >= 20 or step == DECAY_STEPS - 1:
        jax.block_until_ready(metrics)
        m = {k: float(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))[k]) for k in metrics}
        lr = float(decay_schedule(step))
        elapsed = time.time() - decay_start_time
        tok_per_sec = tokens_decay_trained / elapsed if elapsed > 0 else 0
        tok_per_param = tokens_decay_trained / APPROX_PARAMS
        print(f"Decay Step {step:4d}/{DECAY_STEPS} | loss={m['loss']:.4f} | z={m['z_loss']:.2e} | lr={lr:.2e} | {tok_per_sec:,.0f} tok/s | {tok_per_param:.2f} tok/param", flush=True)
        last_decay_log_time = time.time()

print("✓ Decay phase complete.")
# Extract parameters for SFT
final_pretrain_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state_decay.params))


# %% [markdown]
# ## 5. SFT Phase
# **Target:** ~1200 steps on Tulu 3 SFT mixture. Loss is masked to assistant tokens only.

# %%
SFT_STEPS = 1200
SFT_BATCH_PER_DEVICE = 8  # Smaller batch for SFT
SFT_GLOBAL_BATCH = SFT_BATCH_PER_DEVICE * 8

# SFT Schedule: very short warmup, then flat low LR, then decay
sft_schedule = optax.join_schedules(
    [
        optax.linear_schedule(0.0, 5e-5, 50),
        optax.cosine_decay_schedule(5e-5, SFT_STEPS - 50, alpha=0.1)
    ],
    boundaries=[50]
)

sft_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=sft_schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.05)
)

state_sft = TrainState.create(
    apply_fn=model.apply,
    params=final_pretrain_params,
    tx=sft_optimizer,
)
state_sft = flax.jax_utils.replicate(state_sft)


# ── SFT Data Loader (with loss masking) ───────────────────────────────────────

# We use allenai/tulu-3-sft-mixture which is well-formatted multi-turn chat.
sft_stream = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

def format_sft_conversation(example, sp, seq_len):
    """Format Tulu3 messages into tokens and loss mask."""
    tokens = []
    mask = []

    # System prompt
    sys_prompt = "<|system|>You are a helpful assistant.<|end|>\n"
    ids = sp.encode(sys_prompt, add_special_tokens=False)
    tokens.extend(ids)
    mask.extend([0] * len(ids))

    for msg in example.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted = f"<|user|>{content}<|end|>\n"
            ids = sp.encode(formatted, add_special_tokens=False)
            tokens.extend(ids)
            mask.extend([0] * len(ids))
        elif role == "assistant":
            formatted = f"<|assistant|>{content}<|end|>\n"
            ids = sp.encode(formatted, add_special_tokens=False)
            tokens.extend(ids)
            mask.extend([1] * len(ids))

    # Pad or truncate
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
        mask = mask[:seq_len]
    else:
        pad_len = seq_len - len(tokens)
        tokens.extend([sp.eos_token_id] * pad_len)
        mask.extend([0] * pad_len)

    return np.array(tokens, dtype=np.int32), np.array(mask, dtype=np.int32)

def make_sft_loader(stream, tokenizer, seq_len, per_device_batch, n_devices):
    global_batch = n_devices * per_device_batch
    q = queue.Queue(maxsize=4)
    def fill():
        batch_tok, batch_msk = [], []
        for ex in stream:
            try:
                tok, msk = format_sft_conversation(ex, tokenizer, seq_len)
                batch_tok.append(tok)
                batch_msk.append(msk)
                if len(batch_tok) == global_batch:
                    arr_t = np.stack(batch_tok).reshape(n_devices, per_device_batch, seq_len)
                    arr_m = np.stack(batch_msk).reshape(n_devices, per_device_batch, seq_len)
                    q.put((arr_t, arr_m))
                    batch_tok, batch_msk = [], []
            except Exception:
                continue
    t = threading.Thread(target=fill, daemon=True)
    t.start()
    while True: yield q.get()

sft_loader = make_sft_loader(sft_stream, tokenizer, CFG.max_seq_len, SFT_BATCH_PER_DEVICE, 8)


# ── SFT Train Step ────────────────────────────────────────────────────────────

def compute_sft_loss(logits, input_ids, mask, z_loss_weight=1e-5):
    targets = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    mask = mask[:, 1:]

    xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    ce_loss = jnp.sum(xent * mask) / (jnp.sum(mask) + 1e-8)

    log_z = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
    z_loss = z_loss_weight * (jnp.sum((log_z ** 2) * mask) / (jnp.sum(mask) + 1e-8))

    return ce_loss + z_loss, {"loss": ce_loss, "z_loss": z_loss}

@partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def train_step_sft(state, batch_tokens, batch_mask):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_tokens)
        return compute_sft_loss(logits, batch_tokens, batch_mask, CFG.z_loss_weight)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads   = jax.lax.pmean(grads,   axis_name="devices")
    metrics = jax.lax.pmean(metrics, axis_name="devices")
    metrics["grad_norm"] = optax.global_norm(grads)
    return state.apply_gradients(grads=grads), metrics


# Run SFT loop
print("\n" + "="*60)
print(f" Starting SFT Phase ({SFT_STEPS} steps)")
print("="*60)

last_sft_log_time = time.time()
sft_start_time = time.time()
tokens_sft_trained = 0
SFT_TOKENS_PER_STEP = SFT_BATCH_PER_DEVICE * 8 * CFG.max_seq_len

for step in range(SFT_STEPS):
    batch_tok, batch_msk = next(sft_loader)

    state_sft, metrics = train_step_sft(state_sft, batch_tok, batch_msk)
    tokens_sft_trained += SFT_TOKENS_PER_STEP

    if time.time() - last_sft_log_time >= 20 or step == SFT_STEPS - 1:
        jax.block_until_ready(metrics)
        m = {k: float(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))[k]) for k in metrics}
        lr = float(sft_schedule(step))
        elapsed = time.time() - sft_start_time
        tok_per_sec = tokens_sft_trained / elapsed if elapsed > 0 else 0
        tok_per_param = tokens_sft_trained / APPROX_PARAMS
        print(f"SFT Step {step:4d}/{SFT_STEPS} | loss={m['loss']:.4f} | z={m['z_loss']:.2e} | lr={lr:.2e} | {tok_per_sec:,.0f} tok/s | {tok_per_param:.2f} tok/param", flush=True)
        last_sft_log_time = time.time()

print("✓ SFT phase complete.")

# %% [markdown]
# ## 6. Save Final Model

# %%
FINAL_DIR = "/kaggle/working/final_model"
os.makedirs(FINAL_DIR, exist_ok=True)
print(f"\nSaving final SFT parameters to {FINAL_DIR}/params.npz ...")

final_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state_sft.params))
leaves, treedef = jax.tree_util.tree_flatten(final_params)
np_leaves = [np.array(leaf) for leaf in leaves]
np.savez_compressed(os.path.join(FINAL_DIR, "params.npz"), *np_leaves)

with open(os.path.join(FINAL_DIR, "params_treedef.pkl"), "wb") as f:
    pickle.dump(treedef, f)

print("✓ Final model saved.")
print("\n" + "="*60)
print(" KIND-ARCHIMEDS TRAINING COMPLETE!")
print(" To use this model, you can download `final_model` from the Output tab.")
print("="*60)
