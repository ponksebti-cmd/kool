import os
import sys
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import sentencepiece as spm
from dataclasses import dataclass

# Prevent JAX from allocating all GPU memory if run on a machine with a GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 1. Model Definition (Copied from session1_pretrain.py to keep chat.py standalone)
@dataclass
class ModelConfig:
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    head_dim: int = 64
    ffn_dim: int = 4096
    max_seq_len: int = 2048
    parallel_attn_ffn: bool = False
    use_qk_norm: bool = False
    logit_soft_cap: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

CFG = ModelConfig()

class RMSNorm(nn.Module):
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.bfloat16
    @nn.compact
    def __call__(self, x):
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.epsilon)
        normed = (x_f32 / rms).astype(self.dtype)
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        return normed * scale

def apply_rope(xq, freqs_cos, freqs_sin):
    xq_f32 = xq.astype(jnp.float32)
    x1, x2 = xq_f32[..., 0::2], xq_f32[..., 1::2]
    xq_rot_even = x1 * freqs_cos - x2 * freqs_sin
    xq_rot_odd  = x1 * freqs_sin + x2 * freqs_cos
    xq_out = jnp.stack([xq_rot_even, xq_rot_odd], axis=-1).reshape(xq.shape)
    return xq_out.astype(xq.dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs)
    return jnp.array(np.cos(freqs)), jnp.array(np.sin(freqs))

ROPE_COS, ROPE_SIN = precompute_freqs_cis(CFG.head_dim, CFG.max_seq_len, CFG.rope_theta)

def _rms_norm_qk(x, scale, eps):
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * scale).astype(x.dtype)

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

        q = q.reshape(B, T, cfg.n_heads,    cfg.head_dim)
        k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim)

        if cfg.use_qk_norm:
            q_scale = self.param("q_norm_scale", nn.initializers.ones, (cfg.n_heads, cfg.head_dim))
            k_scale = self.param("k_norm_scale", nn.initializers.ones, (cfg.n_kv_heads, cfg.head_dim))
            q = _rms_norm_qk(q, q_scale, cfg.norm_eps)
            k = _rms_norm_qk(k, k_scale, cfg.norm_eps)

        q_h = q.transpose(0, 2, 1, 3)
        k_h = k.transpose(0, 2, 1, 3)
        q_h = apply_rope(q_h, ROPE_COS[:T], ROPE_SIN[:T])
        k_h = apply_rope(k_h, ROPE_COS[:T], ROPE_SIN[:T])
        q = q_h.transpose(0, 2, 1, 3)
        k = k_h.transpose(0, 2, 1, 3)

        n_rep = cfg.n_heads // cfg.n_kv_heads
        k = jnp.repeat(k, n_rep, axis=2)
        v = jnp.repeat(v, n_rep, axis=2)

        out = jax.nn.dot_product_attention(
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            v.astype(jnp.float32),
            is_causal=True,
        ).astype(dtype)

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

        class ScannedBlock(nn.Module):
            config: ModelConfig
            @nn.compact
            def __call__(self, carry, _):
                out = TransformerBlock(self.config, name="block")(carry)
                return out, None

        ScanLayers = nn.scan(
            ScannedBlock,
            variable_axes={'params': 0},
            variable_broadcast=False,
            split_rngs={'params': False},
            length=cfg.n_layers,
        )
        x, _ = ScanLayers(cfg, name="layers")(x, jnp.arange(cfg.n_layers))
        
        x = RMSNorm(epsilon=cfg.norm_eps, dtype=cfg.dtype, name="final_norm")(x)
        logits = (x @ embed.T).astype(jnp.float32)
        if cfg.logit_soft_cap > 0:
            logits = cfg.logit_soft_cap * jnp.tanh(logits / cfg.logit_soft_cap)
        return logits


# 2. Loading Checkpoints
def load_checkpoint(filepath):
    print(f"Loading weights from {filepath} ...")
    with np.load(filepath, allow_pickle=False) as data:
        np_leaves = [data[f"arr_{i}"] for i in range(len(data.files))]
    
    dummy_model = Transformer(CFG)
    dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)
    print("Initialising dummy model for tree structure...")
    dummy_params = dummy_model.init(jax.random.PRNGKey(0), dummy_input)["params"]
    treedef = jax.tree_util.tree_structure(dummy_params)
    
    params = jax.tree_util.tree_unflatten(treedef, np_leaves)
    return params

def sample_top_k(logits, k, key):
    vals, _ = jax.lax.top_k(logits, k)
    min_val = vals[..., -1, None]
    logits = jnp.where(logits < min_val, -1e10, logits)
    return jax.random.categorical(key, logits)

# 3. Chat Loop
def chat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to local checkpoint .npz file")
    parser.add_argument("--hf-repo", type=str, default="", help="Hugging Face repo to pull latest checkpoint from (e.g. sebtiwho/archimedes-ckpts)")
    parser.add_argument("--hf-token", type=str, default="", help="HF Token if repo is private")
    parser.add_argument("--tokenizer", type=str, default="notebooks/tokenizer.model")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_gen", type=int, default=500)
    args = parser.parse_args()

    if not args.ckpt and not args.hf_repo:
        print("Error: You must provide either --ckpt or --hf-repo")
        return

    # Load tokenizer
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at {args.tokenizer}")
        return
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    # Resolve Checkpoint Path
    ckpt_path = args.ckpt
    if args.hf_repo:
        from huggingface_hub import hf_hub_download, list_repo_files
        print(f"Checking Hugging Face Hub for {args.hf_repo}...")
        try:
            token = args.hf_token or os.environ.get("HF_TOKEN")
            # Get latest.txt
            latest_txt = hf_hub_download(repo_id=args.hf_repo, filename="latest.txt", token=token)
            with open(latest_txt, "r") as f:
                latest_step_dir = f.read().strip()
            print(f"Found latest step: {latest_step_dir}")
            
            # Download params.npz
            ckpt_path = hf_hub_download(
                repo_id=args.hf_repo, 
                filename=f"{latest_step_dir}/params.npz", 
                token=token
            )
        except Exception as e:
            print(f"Failed to pull from Hugging Face: {e}")
            return

    # Load Model
    params = load_checkpoint(ckpt_path)
    model = Transformer(CFG)
    
    # Ensure eager mode to prevent recompiling for every sequence length
    jax.config.update('jax_disable_jit', True)

    print("\n" + "="*50)
    print(" 🚀 Archimedes Chat Ready!")
    print(" Type 'quit' to exit.")
    print("="*50 + "\n")

    key = jax.random.PRNGKey(42)
    history = []

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["quit", "exit"]:
                break
        except (KeyboardInterrupt, EOFError):
            break

        history.append({"role": "user", "content": user_input})
        
        # Build prompt string using the SFT format
        prompt = ""
        for msg in history:
            prompt += f"<|{msg['role']}|>{msg['content']}<|end|>\n"
        prompt += "<|assistant|>"
        
        tokens = sp.encode(prompt, add_special_tokens=False)
        print("Assistant: ", end="", flush=True)
        
        generated_tokens = []
        for _ in range(args.max_gen):
            input_ids = jnp.array([tokens], dtype=jnp.int32)
            
            # Forward pass (executed eagerly on CPU/GPU)
            logits = model.apply({"params": params}, input_ids)
            next_token_logits = logits[0, -1, :] / args.temperature
            
            key, subkey = jax.random.split(key)
            next_token = int(sample_top_k(next_token_logits, k=40, key=subkey))
            
            if next_token == sp.eos_token_id or next_token == 2: # Mistral EOS
                break
                
            generated_tokens.append(next_token)
            tokens.append(next_token)
            
            word = sp.decode([next_token])
            print(word, end="", flush=True)
            
        print()
        history.append({"role": "assistant", "content": sp.decode(generated_tokens)})

if __name__ == "__main__":
    chat()
