#!/usr/bin/env python3
"""Independent NumPy reference forward pass for Llama-family GGUF models.

Used to generate golden logits for the C++ runtime tests. Deliberately
straightforward (fp32, no KV cache: full recompute per step) so it is easy to
audit against the model definition.

    PYTHONUTF8=1 python tools/ref_llama.py models/SmolLM2-135M-Instruct-f16.gguf \
        --tokens 1 2 3 --steps 8 --out tests/data/ref_smollm2.txt

Output file:
    tokens: <prompt ids>
    greedy: <ids generated greedily>
    top: <id>:<logit> ... (top 32 logits after the prompt)
    stats: <sum of logits> <sum of squares>
"""
import argparse

import numpy as np
from gguf import GGUFReader


def load(path):
    r = GGUFReader(path)
    kv = {}
    for f in r.fields.values():
        if f.types and f.types[0].name != "ARRAY":
            v = f.parts[f.data[0]]
            kv[f.name] = bytes(v).decode() if f.types[0].name == "STRING" else v.tolist()[0]
    w = {}
    for t in r.tensors:
        a = np.asarray(t.data)
        if a.dtype not in (np.float32, np.float16):
            raise SystemExit(f"{t.name}: only f32/f16 tensors supported by the reference")
        # GGUF stores ne[0] innermost; numpy shape is already [outer..., inner].
        w[t.name] = a.astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
    return kv, w


def rms_norm(x, g, eps):
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps).astype(np.float32) * g


def rope_interleaved(x, pos, base, rot):
    # x: [T, H, D]; rotate pairs (2i, 2i+1) for i < rot/2
    half = rot // 2
    inv = base ** (-2.0 * np.arange(half, dtype=np.float64) / rot)
    ang = pos[:, None].astype(np.float64) * inv[None, :]  # [T, half]
    c, s = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    x = x.copy()
    x0, x1 = x[..., 0:rot:2].copy(), x[..., 1:rot:2].copy()
    x[..., 0:rot:2] = x0 * c - x1 * s
    x[..., 1:rot:2] = x0 * s + x1 * c
    return x


def forward(kv, w, tokens):
    arch = kv["general.architecture"]
    assert arch == "llama", arch
    L = kv[f"{arch}.block_count"]
    H = kv[f"{arch}.attention.head_count"]
    KVH = kv.get(f"{arch}.attention.head_count_kv", H)
    D = kv[f"{arch}.embedding_length"]
    hd = D // H
    eps = kv[f"{arch}.attention.layer_norm_rms_epsilon"]
    base = kv.get(f"{arch}.rope.freq_base", 10000.0)
    rot = kv.get(f"{arch}.rope.dimension_count", hd)

    T = len(tokens)
    pos = np.arange(T)
    x = w["token_embd.weight"][tokens]  # [T, D]
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    for l in range(L):
        p = f"blk.{l}."
        h = rms_norm(x, w[p + "attn_norm.weight"], eps)
        q = (h @ w[p + "attn_q.weight"].T).reshape(T, H, hd)
        k = (h @ w[p + "attn_k.weight"].T).reshape(T, KVH, hd)
        v = (h @ w[p + "attn_v.weight"].T).reshape(T, KVH, hd)
        q, k = rope_interleaved(q, pos, base, rot), rope_interleaved(k, pos, base, rot)
        g = H // KVH
        k, v = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, H * hd)
        x = x + o @ w[p + "attn_output.weight"].T
        h = rms_norm(x, w[p + "ffn_norm.weight"], eps)
        gate = h @ w[p + "ffn_gate.weight"].T
        up = h @ w[p + "ffn_up.weight"].T
        x = x + ((gate / (1 + np.exp(-gate))) * up) @ w[p + "ffn_down.weight"].T
    x = rms_norm(x[-1:], w["output_norm.weight"], eps)
    lm = w.get("output.weight", w["token_embd.weight"])
    return (x @ lm.T)[0]


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
    greedy = []
    logits = logits0
    for _ in range(a.steps):
        nxt = int(np.argmax(logits))
        greedy.append(nxt)
        logits = forward(kv, w, tokens + greedy)
    top = np.argsort(-logits0)[:32]
    with open(a.out, "w", newline="\n") as f:
        f.write("tokens: " + " ".join(map(str, a.tokens)) + "\n")
        f.write("greedy: " + " ".join(map(str, greedy)) + "\n")
        f.write("top: " + " ".join(f"{i}:{logits0[i]:.6f}" for i in top) + "\n")
        f.write(f"stats: {float(np.sum(logits0.astype(np.float64))):.6f} "
                f"{float(np.sum(logits0.astype(np.float64) ** 2)):.6f}\n")
    print(f"greedy {greedy}; top1 {top[0]} ({logits0[top[0]]:.4f})")


if __name__ == "__main__":
    main()
