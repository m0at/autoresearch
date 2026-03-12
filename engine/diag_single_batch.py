#!/usr/bin/env python3
"""
Single-batch forward/backward diagnostic: compare Python GPT against Rust engine.

Standalone script (no imports from compare_train.py or prepare.py).
Reads a batch from the binary shard format, runs one forward + backward pass,
and prints loss, logit stats, and per-parameter gradient norms.

Usage (on H100):
    python3 diag_single_batch.py --init-weights init_weights_d8.safetensors
    python3 diag_single_batch.py   # uses torch.manual_seed(42) init
"""

import argparse
import struct
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# GPT Model (standalone, matches compare_train.py exactly)
# ---------------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, n_layer, layer_idx):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ve_gate = (
            nn.Linear(32, n_kv_head, bias=False)
            if has_ve(layer_idx, n_layer)
            else None
        )
        self.layer_idx = layer_idx

    def _try_flash_attn(self, q, k, v, window_size, seq_t):
        """Try flash-attn-3 (Hopper), fall back to SDPA."""
        try:
            from kernels import get_kernel
            cap = torch.cuda.get_device_capability()
            if cap == (9, 0):
                repo = "varunneal/flash-attention-3"
            else:
                repo = "kernels-community/flash-attn3"
            fa3 = get_kernel(repo).flash_attn_interface
            if window_size < seq_t:
                ws = (window_size, 0)
            else:
                ws = (-1, -1)
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=ws)
            return y
        except Exception:
            pass
        # SDPA fallback
        q2 = q.transpose(1, 2)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        if 0 < window_size < seq_t:
            mask = torch.ones(seq_t, seq_t, device=q.device, dtype=torch.bool).tril()
            for i in range(seq_t):
                start = max(0, i - window_size + 1)
                if start > 0:
                    mask[i, :start] = False
            mask = mask.unsqueeze(0).unsqueeze(0)
            y = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        return y.transpose(1, 2)

    def forward(self, x, ve, cos, sin, window_size):
        B, T, _C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))
            v = v + gate.unsqueeze(-1) * ve
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = self._try_flash_attn(q, k, v, window_size, T)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, n_layer, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(
            n_embd, n_head, n_kv_head, n_layer, layer_idx
        )
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x, ve, cos, sin, window_size):
        x = x + self.attn(norm(x), ve, cos, sin, window_size)
        h = self.c_fc(norm(x))
        h = F.relu(h).square()
        x = x + self.c_proj(h)
        return x


class GPT(nn.Module):
    def __init__(self, vocab, n_embd, n_head, n_kv_head, n_layer, seq_len):
        super().__init__()
        self.vocab = vocab
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.seq_len = seq_len

        self.wte = nn.Embedding(vocab, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, n_kv_head, n_layer, i) for i in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        kv_dim = n_kv_head * self.head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(vocab, kv_dim)
                for i in range(n_layer)
                if has_ve(i, n_layer)
            }
        )

        # RoPE (precompute for seq_len * 10 like reference)
        rotary_len = seq_len * 10
        channel_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (channel_range / self.head_dim))
        t_pos = torch.arange(rotary_len, dtype=torch.float32)
        freqs = torch.outer(t_pos, inv_freq)
        self.register_buffer("cos", freqs.cos().bfloat16()[None, :, None, :])
        self.register_buffer("sin", freqs.sin().bfloat16()[None, :, None, :])

        # Window sizes: SSSL pattern
        pattern = "SSSL"
        ws = []
        for i in range(n_layer):
            c = pattern[i % len(pattern)]
            ws.append(seq_len // 2 if c == "S" else seq_len)
        ws[-1] = seq_len
        self.window_sizes = ws

    def init_weights(self):
        """Match Python reference init_weights() RNG order exactly."""
        nn.init.normal_(self.wte.weight, 0, 1)
        nn.init.normal_(self.lm_head.weight, 0, 0.001)
        s = 3**0.5 * self.n_embd**-0.5
        for b in self.blocks:
            nn.init.uniform_(b.attn.c_q.weight, -s, s)
            nn.init.uniform_(b.attn.c_k.weight, -s, s)
            nn.init.uniform_(b.attn.c_v.weight, -s, s)
            nn.init.zeros_(b.attn.c_proj.weight)
            nn.init.uniform_(b.c_fc.weight, -s, s)
            nn.init.zeros_(b.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        for b in self.blocks:
            if b.attn.ve_gate is not None:
                nn.init.zeros_(b.attn.ve_gate.weight)
        self.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = str(i)
            ve = self.value_embeds[ve_key](idx) if ve_key in self.value_embeds else None
            x = block(x, ve, cos, sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 15.0 * torch.tanh(logits / 15.0)
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            return logits, loss
        return logits


# ---------------------------------------------------------------------------
# Safetensors weight loading (handles both engine and Python naming)
# ---------------------------------------------------------------------------


def load_engine_weights(model, path):
    from safetensors.torch import load_file

    tensors = load_file(path, device=str(model.wte.weight.device))

    is_python = "transformer.wte.weight" in tensors

    mapping = {}
    if is_python:
        mapping["wte.weight"] = "transformer.wte.weight"
        mapping["lm_head.weight"] = "lm_head.weight"
        mapping["resid_lambdas"] = "resid_lambdas"
        mapping["x0_lambdas"] = "x0_lambdas"
        for i in range(model.n_layer):
            mapping[f"blocks.{i}.attn.c_q.weight"] = (
                f"transformer.h.{i}.attn.c_q.weight"
            )
            mapping[f"blocks.{i}.attn.c_k.weight"] = (
                f"transformer.h.{i}.attn.c_k.weight"
            )
            mapping[f"blocks.{i}.attn.c_v.weight"] = (
                f"transformer.h.{i}.attn.c_v.weight"
            )
            mapping[f"blocks.{i}.attn.c_proj.weight"] = (
                f"transformer.h.{i}.attn.c_proj.weight"
            )
            mapping[f"blocks.{i}.c_fc.weight"] = (
                f"transformer.h.{i}.mlp.c_fc.weight"
            )
            mapping[f"blocks.{i}.c_proj.weight"] = (
                f"transformer.h.{i}.mlp.c_proj.weight"
            )
            if has_ve(i, model.n_layer):
                mapping[f"blocks.{i}.attn.ve_gate.weight"] = (
                    f"transformer.h.{i}.attn.ve_gate.weight"
                )
                mapping[f"value_embeds.{i}.weight"] = f"value_embeds.{i}.weight"
    else:
        mapping["wte.weight"] = "wte.weight"
        mapping["lm_head.weight"] = "lm_head.weight"
        mapping["resid_lambdas"] = "resid_lambdas"
        mapping["x0_lambdas"] = "x0_lambdas"
        for i in range(model.n_layer):
            mapping[f"blocks.{i}.attn.c_q.weight"] = f"h.{i}.attn.c_q.weight"
            mapping[f"blocks.{i}.attn.c_k.weight"] = f"h.{i}.attn.c_k.weight"
            mapping[f"blocks.{i}.attn.c_v.weight"] = f"h.{i}.attn.c_v.weight"
            mapping[f"blocks.{i}.attn.c_proj.weight"] = f"h.{i}.attn.c_proj.weight"
            mapping[f"blocks.{i}.c_fc.weight"] = f"h.{i}.mlp.c_fc.weight"
            mapping[f"blocks.{i}.c_proj.weight"] = f"h.{i}.mlp.c_proj.weight"
            if has_ve(i, model.n_layer):
                mapping[f"blocks.{i}.attn.ve_gate.weight"] = (
                    f"h.{i}.attn.ve_gate.weight"
                )
                mapping[f"value_embeds.{i}.weight"] = f"ve.{i}.weight"

    state = model.state_dict()
    loaded = 0
    for model_name, st_name in mapping.items():
        if st_name not in tensors:
            print(f"  WARNING: {st_name} not in safetensors")
            continue
        if model_name not in state:
            print(f"  WARNING: {model_name} not in model state_dict")
            continue
        t = tensors[st_name]
        if t.shape != state[model_name].shape:
            print(
                f"  WARNING: shape mismatch {model_name}: "
                f"{t.shape} vs {state[model_name].shape}"
            )
            continue
        state[model_name] = t.to(dtype=state[model_name].dtype)
        loaded += 1

    model.load_state_dict(state)
    fmt = "Python" if is_python else "engine"
    print(f"Loaded {loaded} tensors from {path} ({fmt} format)")


# ---------------------------------------------------------------------------
# Shard reader (Rust engine binary format)
# ---------------------------------------------------------------------------

SHARD_MAGIC = b"TKNS"


def read_shard(path, max_rows=None):
    """Read binary shard. Returns (seq_len, np.array[num_rows, seq_len] u16)."""
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] == SHARD_MAGIC:
        version, vocab_size, seq_len, num_rows = struct.unpack_from(
            "<IIII", raw, 4
        )
        hdr = 20
        print(
            f"Shard: magic=TKNS, version={version}, vocab={vocab_size}, "
            f"seq_len={seq_len}, num_rows={num_rows}"
        )
    else:
        version, seq_len, num_rows, _ = struct.unpack_from("<IIII", raw, 0)
        hdr = 16
        print(
            f"Shard (old): version={version}, "
            f"seq_len={seq_len}, num_rows={num_rows}"
        )

    if max_rows is not None:
        num_rows = min(num_rows, max_rows)

    tokens = np.frombuffer(raw[hdr:], dtype=np.uint16).reshape(-1, seq_len)
    return seq_len, tokens[:num_rows]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Single-batch forward/backward diagnostic"
    )
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Path to safetensors (engine or Python format)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="/root/.cache/autoresearch/shards/shard_00000.bin",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-batch", type=str, default="/root/diag_batch.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    B = args.batch_size
    T = 2048

    VOCAB = 8192
    D_MODEL = 512
    N_HEAD = 4
    N_KV_HEAD = 4
    N_LAYER = 8

    print("=" * 70)
    print("SINGLE-BATCH FORWARD/BACKWARD DIAGNOSTIC")
    print("=" * 70)
    print(
        f"Config: depth={N_LAYER}, d_model={D_MODEL}, n_head={N_HEAD}, "
        f"head_dim={D_MODEL // N_HEAD}, mlp_dim={4 * D_MODEL}, vocab={VOCAB}"
    )

    # --- Build model ---
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    model = GPT(VOCAB, D_MODEL, N_HEAD, N_KV_HEAD, N_LAYER, T).to(device)

    if args.init_weights:
        print(f"\nLoading weights from: {args.init_weights}")
        load_engine_weights(model, args.init_weights)
        # Recompute rotary on device + cast embeds to bf16
        channel_range = torch.arange(
            0, model.head_dim, 2, dtype=torch.float32, device=device
        )
        inv_freq = 1.0 / (10000 ** (channel_range / model.head_dim))
        t_pos = torch.arange(T * 10, dtype=torch.float32, device=device)
        freqs = torch.outer(t_pos, inv_freq)
        model.cos = freqs.cos().bfloat16()[None, :, None, :]
        model.sin = freqs.sin().bfloat16()[None, :, None, :]
        model.wte.to(dtype=torch.bfloat16)
        for ve in model.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
    else:
        print("\nUsing torch.manual_seed(42) default init")
        model.init_weights()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Window sizes: {model.window_sizes}")
    ve_layers = [i for i in range(N_LAYER) if has_ve(i, N_LAYER)]
    print(f"VE layers: {ve_layers}")

    # --- Load shard ---
    print(f"\nLoading shard: {args.shard}")
    seq_len, all_tokens = read_shard(args.shard, max_rows=B)
    assert all_tokens.shape[0] >= B, f"Need {B} rows, got {all_tokens.shape[0]}"
    assert seq_len >= T + 1, f"Shard seq_len={seq_len}, need >= {T + 1}"

    batch_np = all_tokens[:B].astype(np.int64)
    input_ids = torch.from_numpy(batch_np[:, :T]).to(device)
    targets = torch.from_numpy(batch_np[:, 1 : T + 1]).to(device)

    print(
        f"Input shape: {input_ids.shape}, "
        f"range: [{input_ids.min().item()}, {input_ids.max().item()}]"
    )
    print(
        f"Target shape: {targets.shape}, "
        f"range: [{targets.min().item()}, {targets.max().item()}]"
    )
    print(f"First 20 input tokens:  {input_ids[0, :20].tolist()}")
    print(f"First 20 target tokens: {targets[0, :20].tolist()}")

    # Save batch
    try:
        torch.save(
            {"input_ids": input_ids.cpu(), "targets": targets.cpu()},
            args.save_batch,
        )
        print(f"Saved batch to {args.save_batch}")
    except Exception as e:
        print(f"Could not save batch: {e}")

    # --- Forward pass ---
    print("\n" + "=" * 70)
    print("FORWARD PASS (bf16 autocast, f32 softcap + CE)")
    print("=" * 70)

    model.train()
    model.zero_grad(set_to_none=True)
    autocast_ctx = torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16
    )

    with autocast_ctx:
        logits, loss = model(input_ids, targets)

    print(f"Loss (mean CE):       {loss.item():.8f}")
    print(f"Loss (bpb):           {loss.item() / 0.6931472:.8f}")

    # Logit stats
    print(f"\nLogits shape: {logits.shape}, dtype: {logits.dtype}")
    print(
        f"Logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]"
    )
    print(f"Logits mean:  {logits.mean().item():.8f}")
    print(f"Logits std:   {logits.std().item():.8f}")

    first_logits = logits[0, 0, :10].detach()
    print(f"\nFirst 10 logits [0,0,:10]: {first_logits.tolist()}")

    first_probs = F.softmax(logits[0, 0, :].detach(), dim=-1)
    print(f"Softmax probs [0,0,:10]:   {first_probs[:10].tolist()}")
    print(f"Predicted token [0,0]:     {first_probs.argmax().item()}")
    print(f"Target token [0,0]:        {targets[0, 0].item()}")

    # --- Backward pass ---
    print("\n" + "=" * 70)
    print("BACKWARD PASS")
    print("=" * 70)

    loss.backward()

    header = (
        f"{'Name':50s} {'Shape':20s} {'Grad Norm':>12s} {'Param Norm':>12s}"
    )
    print(f"\n{header}")
    print("-" * 98)

    grad_groups = OrderedDict()
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.float().norm().item()
            pn = p.data.float().norm().item()
            shape_str = str(list(p.shape))
            print(f"{name:50s} {shape_str:20s} {gn:12.6e} {pn:12.6f}")

            if "wte" in name:
                group = "wte"
            elif "lm_head" in name:
                group = "lm_head"
            elif "value_embeds" in name:
                group = "value_embeds"
            elif "ve_gate" in name:
                group = "ve_gate"
            elif "resid_lambdas" in name:
                group = "resid_lambdas"
            elif "x0_lambdas" in name:
                group = "x0_lambdas"
            elif "attn" in name:
                group = "attn_matrices"
            elif "c_fc" in name or "c_proj" in name:
                group = "mlp_matrices"
            else:
                group = "other"

            grad_groups.setdefault(group, []).append(gn)
        else:
            shape_str = str(list(p.shape))
            print(f"{name:50s} {shape_str:20s} {'NO GRAD':>12s}")

    print("\nGradient norm summary by group:")
    print(
        f"{'Group':25s} {'Count':>6s} {'Mean':>14s} "
        f"{'Max':>14s} {'Min':>14s}"
    )
    print("-" * 75)
    for group, norms in grad_groups.items():
        mean_n = sum(norms) / len(norms)
        max_n = max(norms)
        min_n = min(norms)
        print(
            f"{group:25s} {len(norms):6d} {mean_n:14.6e} "
            f"{max_n:14.6e} {min_n:14.6e}"
        )

    # --- Parameter norms summary ---
    print("\nParameter norms:")
    print(f"  wte:           {model.wte.weight.float().norm().item():.6f}")
    print(f"  lm_head:       {model.lm_head.weight.float().norm().item():.6f}")
    print(f"  resid_lambdas: {model.resid_lambdas.data.tolist()}")
    print(f"  x0_lambdas:    {model.x0_lambdas.data.tolist()}")

    # --- Layer-by-layer activation trace ---
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER ACTIVATION TRACE (eval, no grad)")
    print("=" * 70)

    model.eval()
    with torch.no_grad(), autocast_ctx:
        x = model.wte(input_ids)
        xf = x.float()
        print(
            f"\nPost-embedding:   "
            f"mean={xf.mean().item():+.8f}  "
            f"std={xf.std().item():.8f}  "
            f"norm={xf.norm().item():.4f}"
        )

        x = norm(x)
        xf = x.float()
        print(
            f"Post-emb-norm:    "
            f"mean={xf.mean().item():+.8f}  "
            f"std={xf.std().item():.8f}  "
            f"norm={xf.norm().item():.4f}"
        )

        x0 = x.clone()
        cos_buf = model.cos[:, :T]
        sin_buf = model.sin[:, :T]

        for i, block in enumerate(model.blocks):
            x_scaled = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve_key = str(i)
            ve = (
                model.value_embeds[ve_key](input_ids)
                if ve_key in model.value_embeds
                else None
            )
            x = block(x_scaled, ve, cos_buf, sin_buf, model.window_sizes[i])
            xf = x.float()
            print(
                f"Layer {i} output:   "
                f"mean={xf.mean().item():+.8f}  "
                f"std={xf.std().item():.8f}  "
                f"norm={xf.norm().item():.4f}"
            )

        x = norm(x)
        xf = x.float()
        print(
            f"Final norm:       "
            f"mean={xf.mean().item():+.8f}  "
            f"std={xf.std().item():.8f}  "
            f"norm={xf.norm().item():.4f}"
        )

        logits_raw = model.lm_head(x)
        logits_f32 = logits_raw.float()
        logits_capped = 15.0 * torch.tanh(logits_f32 / 15.0)
        print(
            f"Logits raw bf16:  "
            f"mean={logits_raw.float().mean().item():+.8f}  "
            f"std={logits_raw.float().std().item():.8f}"
        )
        print(
            f"Logits f32 cap:   "
            f"mean={logits_capped.mean().item():+.8f}  "
            f"std={logits_capped.std().item():.8f}"
        )

    # --- Manual CE check ---
    print("\n" + "=" * 70)
    print("MANUAL CROSS-ENTROPY CHECK")
    print("=" * 70)

    with torch.no_grad():
        log_probs = F.log_softmax(
            logits_capped.view(-1, VOCAB), dim=-1
        )
        flat_targets = targets.view(-1)
        arange = torch.arange(log_probs.size(0), device=device)
        per_token_loss = -log_probs[arange, flat_targets]
        manual_mean = per_token_loss.mean().item()
        rust_style = per_token_loss.sum().item() / (B * T)

        print(f"Manual mean CE (PyTorch-style): {manual_mean:.8f}")
        print(f"Rust-style loss (sum/BT):       {rust_style:.8f}")
        print(f"Model loss from forward():      {loss.item():.8f}")
        diff = abs(manual_mean - loss.item())
        print(f"Diff (manual vs model):         {diff:.2e}")

        ptl_mean = per_token_loss.mean().item()
        ptl_std = per_token_loss.std().item()
        ptl_min = per_token_loss.min().item()
        ptl_max = per_token_loss.max().item()
        print(f"\nPer-token loss stats:")
        print(
            f"  mean={ptl_mean:.6f}  std={ptl_std:.6f}  "
            f"min={ptl_min:.6f}  max={ptl_max:.6f}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
