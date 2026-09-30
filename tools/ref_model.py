#!/usr/bin/env python3
"""Independent NumPy reference forward pass for GGUF decoder models.

Generates golden logits for the C++ runtime tests. Written from the Hugging
Face modeling code of each family (not from the C++ runtime), in fp32 with no
KV cache (full recompute per step), so it is easy to audit.

Supported: llama, qwen2, qwen3, gemma, gemma2, gemma3, phi3. Weights may be
f32/f16 or any GGML quantized type (dequantized with gguf-py's reference code).

    PYTHONUTF8=1 python tools/ref_model.py models/SmolLM2-135M-Instruct-f16.gguf \
        --tokens 1 2 3 --steps 8 --out tests/data/ref_smollm2.txt

Output file:
    tokens: <prompt ids>
    greedy: <ids generated greedily>
    top: <id>:<logit> ... (top 32 logits after the prompt)
    stats: <sum of logits> <sum of squares>
"""
import argparse

import gguf
import gguf.quants
import numpy as np
from gguf import GGUFReader


def load(path):
    r = GGUFReader(path)
    kv = {}
    for f in r.fields.values():
        if f.types and f.types[0].name != "ARRAY":
            v = f.parts[f.data[0]]
            kv[f.name] = bytes(v).decode() if f.types[0].name == "STRING" else v.tolist()[0]
    w = Weights()
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape)]
        a = np.asarray(t.data)
        if a.dtype in (np.float32, np.float16):
            w.raw[t.name] = a.reshape(shape)
        else:
            # Quantized: dequantized lazily by gguf-py's reference implementation.
            w.quant[t.name] = (a, t.tensor_type, shape)
    return kv, w


class Weights:
    """Memory-mapped tensors, upcast to fp32 only when used (keeps a 0.5B
    model's reference run within a few hundred MB of RAM)."""

    def __init__(self):
        self.raw = {}
        self.quant = {}

    def __contains__(self, name):
        return name in self.raw or name in self.quant

    def __getitem__(self, name):
        if name in self.quant:
            data, qtype, shape = self.quant[name]
            return gguf.quants.dequantize(data, qtype).astype(np.float32).reshape(shape)
        return self.raw[name].astype(np.float32)

    def rows(self, name, idx):
        """Selected rows (embedding lookup) without dequantizing the whole table."""
        if name in self.quant:
            data, qtype, shape = self.quant[name]
            return gguf.quants.dequantize(data[idx], qtype).astype(np.float32).reshape(len(idx), shape[-1])
        return self.raw[name][idx].astype(np.float32)


def rms_norm(x, g, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32) * g


def rope(x, pos, base, rot, style, scale=1.0):
    """x: [T, H, D]. style 'interleaved' (GGUF llama) or 'half' (NeoX / HF rotate_half)."""
    half = rot // 2
    inv = base ** (-2.0 * np.arange(half, dtype=np.float64) / rot)
    ang = (pos[:, None].astype(np.float64) / scale) * inv[None, :]
    c = np.cos(ang).astype(np.float32)[:, None, :]
    s = np.sin(ang).astype(np.float32)[:, None, :]
    x = x.copy()
    if style == "interleaved":
        a, b = x[..., 0:rot:2].copy(), x[..., 1:rot:2].copy()
        x[..., 0:rot:2], x[..., 1:rot:2] = a * c - b * s, a * s + b * c
    else:
        a, b = x[..., :half].copy(), x[..., half:rot].copy()
        x[..., :half], x[..., half:rot] = a * c - b * s, a * s + b * c
    return x


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1 + np.exp(-x))


def forward(kv, w, tokens):
    arch = kv["general.architecture"]
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)
    L, H, D = g("block_count"), g("attention.head_count"), g("embedding_length")
    KVH = g("attention.head_count_kv", H)
    hd = g("attention.key_length", D // H)
    eps = g("attention.layer_norm_rms_epsilon")
    base = g("rope.freq_base", 10000.0)
    rot = g("rope.dimension_count", hd)
    window = g("attention.sliding_window", 0) or 0
    attn_cap = g("attn_logit_softcapping", 0.0) or 0.0
    final_cap = g("final_logit_softcapping", 0.0) or 0.0

    gemma = arch.startswith("gemma")
    style = "interleaved" if arch == "llama" else "half"
    act = gelu_tanh if gemma else silu
    scale = hd ** -0.5
    if (arch == "gemma2" and L == 46) or (arch == "gemma3" and L == 62):
        scale = (D // H) ** -0.5

    T = len(tokens)
    pos = np.arange(T)
    x = w.rows("token_embd.weight", tokens)
    if gemma:
        x = x * np.float32(np.sqrt(D))
    qi, ki = np.arange(T)[:, None], np.arange(T)[None, :]
    causal = ki <= qi

    for l in range(L):
        p = f"blk.{l}."
        # Which layers are local (sliding window):
        #   gemma2: even layers; gemma3: all but every 6th; phi3: all (if set)
        if arch == "gemma2":
            local = window > 0 and l % 2 == 0
        elif arch == "gemma3":
            local = window > 0 and (l + 1) % 6 != 0
        else:
            local = window > 0
        mask = causal & ((qi - ki) < window) if local else causal
        layer_base = 10000.0 if (arch == "gemma3" and local) else base

        h = rms_norm(x, w[p + "attn_norm.weight"], eps)
        if p + "attn_qkv.weight" in w:
            qkv = h @ w[p + "attn_qkv.weight"].T
            q, k, v = np.split(qkv, [H * hd, H * hd + KVH * hd], axis=1)
        else:
            q, k, v = (h @ w[p + f"attn_{n}.weight"].T for n in "qkv")
            if p + "attn_q.bias" in w:
                q, k, v = q + w[p + "attn_q.bias"], k + w[p + "attn_k.bias"], v + w[p + "attn_v.bias"]
        q, k, v = q.reshape(T, H, hd), k.reshape(T, KVH, hd), v.reshape(T, KVH, hd)
        if p + "attn_q_norm.weight" in w:
            q = rms_norm(q, w[p + "attn_q_norm.weight"], eps)
            k = rms_norm(k, w[p + "attn_k_norm.weight"], eps)
        q, k = rope(q, pos, layer_base, rot, style), rope(k, pos, layer_base, rot, style)
        rep = H // KVH
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.einsum("thd,shd->hts", q, k) * scale
        if attn_cap:
            s = attn_cap * np.tanh(s / attn_cap)
        s = np.where(mask[None], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", s, v).reshape(T, H * hd) @ w[p + "attn_output.weight"].T
        if p + "post_attention_norm.weight" in w:
            o = rms_norm(o, w[p + "post_attention_norm.weight"], eps)
        x = x + o

        h = rms_norm(x, w[p + "ffn_norm.weight"], eps)
        if p + "ffn_gate.weight" in w:
            gate, up = h @ w[p + "ffn_gate.weight"].T, h @ w[p + "ffn_up.weight"].T
        else:  # phi3: fused gate|up stored as ffn_up
            gate, up = np.split(h @ w[p + "ffn_up.weight"].T, 2, axis=1)
        f = (act(gate) * up) @ w[p + "ffn_down.weight"].T
        if p + "post_ffw_norm.weight" in w:
            f = rms_norm(f, w[p + "post_ffw_norm.weight"], eps)
        x = x + f

    x = rms_norm(x[-1:], w["output_norm.weight"], eps)
    lm = "output.weight" if "output.weight" in w else "token_embd.weight"
    logits = (x @ w[lm].T)[0]
    if final_cap:
        logits = final_cap * np.tanh(logits / final_cap)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kv, w = load(a.model)
    tokens = list(a.tokens)
    logits0 = forward(kv, w, tokens)
    greedy, logits = [], logits0
    for _ in range(a.steps):
        nxt = int(np.argmax(logits))
        greedy.append(nxt)
        logits = forward(kv, w, tokens + greedy)
    top = np.argsort(-logits0)[:32]
    with open(a.out, "w", newline="\n") as f:
        f.write("tokens: " + " ".join(map(str, a.tokens)) + "\n")
        f.write("greedy: " + " ".join(map(str, greedy)) + "\n")
        f.write("top: " + " ".join(f"{i}:{logits0[i]:.6f}" for i in top) + "\n")
        l64 = logits0.astype(np.float64)
        f.write(f"stats: {float(np.sum(l64)):.6f} {float(np.sum(l64 ** 2)):.6f}\n")
    print(f"greedy {greedy}; top1 {top[0]} ({logits0[top[0]]:.4f})")


if __name__ == "__main__":
    main()
