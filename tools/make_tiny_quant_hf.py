#!/usr/bin/env python3
"""GPTQ/AWQ fixtures: quantizes the linear layers of tests/data/hf_tiny_llama
(round-to-nearest per group), packs them exactly as AutoGPTQ / AutoAWQ do,
and writes for each variant

    tests/data/hf_tiny_llama_<variant>/      packed checkpoint (config.json with quantization_config)
    tests/data/hf_tiny_llama_<variant>_ref/  the same model with W = scale * (q - zero) as F32

The _ref directories are the independent oracle: they are computed here with
NumPy from the codes, not by unpacking.

    PYTHONUTF8=1 python tools/make_tiny_quant_hf.py tests/data
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

SRC = "hf_tiny_llama"
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]

VARIANTS = {
    # name: method, bits, group, sym, act_order, checkpoint_format
    "gptq_int4_sym_g32": ("gptq", 4, 32, True, False, "gptq"),
    "gptq_int4_asym_g64": ("gptq", 4, 64, False, False, "gptq_v2"),
    "gptq_int4_actorder_g32": ("gptq", 4, 32, False, True, "gptq"),
    "gptq_int8_sym_g32": ("gptq", 8, 32, True, False, "gptq"),
    "awq_int4_g32": ("awq", 4, 32, False, False, None),
}


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        dt = {"F16": np.float16, "F32": np.float32, "I32": np.int32}[v["dtype"]]
        b, e = v["data_offsets"]
        out[k] = np.frombuffer(data[b:e], dtype=dt).reshape(v["shape"])
    return out


def write_st(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, a in tensors.items():
        dt = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32", np.dtype(np.int32): "I32"}[a.dtype]
        b = np.ascontiguousarray(a).tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def quantize(w, bits, group, sym, g_idx):
    """RTN per (output row, group). Returns codes q [out, in], scales/zeros [groups, out]."""
    out_dim, in_dim = w.shape
    maxq = (1 << bits) - 1
    groups = (in_dim + group - 1) // group
    scales = np.zeros((groups, out_dim), np.float16)
    zeros = np.zeros((groups, out_dim), np.int64)
    q = np.zeros((out_dim, in_dim), np.int64)
    for g in range(groups):
        cols = np.nonzero(g_idx == g)[0]
        x = w[:, cols].astype(np.float32)
        if sym:
            xmax = np.maximum(np.abs(x).max(axis=1), 1e-6)
            s = (2 * xmax / maxq).astype(np.float16).astype(np.float32)
            z = np.full(out_dim, (maxq + 1) // 2)
        else:
            lo, hi = np.minimum(x.min(axis=1), 0), np.maximum(x.max(axis=1), 0)
            s = np.maximum((hi - lo) / maxq, 1e-6).astype(np.float16).astype(np.float32)
            z = np.clip(np.round(-lo / s), 1, maxq)  # >= 1: avoids GPTQ v1's zero-0 wraparound
        q[:, cols] = np.clip(np.round(x / s[:, None]) + z[:, None], 0, maxq)
        scales[g] = s.astype(np.float16)
        zeros[g] = z
    return q, scales, zeros


def pack_gptq(q, zeros, bits, v1):
    per = 32 // bits
    qt = q.T.astype(np.uint32)  # [in, out]
    qweight = np.zeros((qt.shape[0] // per, qt.shape[1]), np.uint32)
    for j in range(per):
        qweight |= qt[j::per] << (bits * j)
    z = (zeros - 1 if v1 else zeros).astype(np.int64) & ((1 << bits) - 1)
    qzeros = np.zeros((z.shape[0], z.shape[1] // per), np.uint32)
    for j in range(per):
        qzeros |= z[:, j::per].astype(np.uint32) << (bits * j)
    return qweight.view(np.int32), qzeros.view(np.int32)


def pack_awq(q, zeros):
    qt = q.T.astype(np.uint32)  # [in, out]
    qweight = np.zeros((qt.shape[0], qt.shape[1] // 8), np.uint32)
    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 8), np.uint32)
    for k in range(8):
        qweight |= qt[:, AWQ_ORDER[k]::8] << (4 * k)
        qzeros |= zeros[:, AWQ_ORDER[k]::8].astype(np.uint32) << (4 * k)
    return qweight.view(np.int32), qzeros.view(np.int32)


def main():
    data = sys.argv[1]
    src = os.path.join(data, SRC)
    base = read_st(os.path.join(src, "model.safetensors"))
    rng = np.random.default_rng(24)
    for name, (method, bits, group, sym, act, fmt) in VARIANTS.items():
        packed, ref = {}, {}
        for k, w in base.items():
            is_linear = any(k.endswith(f"{l}.weight") for l in LINEARS)
            if not is_linear:
                packed[k] = w
                ref[k] = w
                continue
            out_dim, in_dim = w.shape
            if act:
                perm = rng.permutation(in_dim)
                g_idx = np.empty(in_dim, np.int64)
                g_idx[perm] = np.arange(in_dim) // group
            else:
                g_idx = np.arange(in_dim) // group
            q, scales, zeros = quantize(w.astype(np.float32), bits, group, sym, g_idx)
            prefix = k[: -len("weight")]
            if method == "gptq":
                qweight, qzeros = pack_gptq(q, zeros, bits, fmt == "gptq")
                packed[prefix + "g_idx"] = g_idx.astype(np.int32)
                eff = zeros if fmt != "gptq" else ((zeros - 1) & ((1 << bits) - 1)) + 1
            else:
                qweight, qzeros = pack_awq(q, zeros)
                eff = zeros
            packed[prefix + "qweight"] = qweight
            packed[prefix + "qzeros"] = qzeros
            packed[prefix + "scales"] = scales
            s = scales.astype(np.float32)[g_idx].T  # [out, in]
            ref[k] = (s * (q - eff[g_idx].T)).astype(np.float32)

        for suffix, tensors in (("", packed), ("_ref", ref)):
            out = os.path.join(data, f"{SRC}_{name}{suffix}")
            os.makedirs(out, exist_ok=True)
            for f in ("tokenizer.json", "tokenizer_config.json"):
                shutil.copy(os.path.join(src, f), out)
            with open(os.path.join(src, "config.json")) as f:
                cfg = json.load(f)
            if not suffix:
                qc = {"quant_method": method, "bits": bits, "group_size": group}
                if method == "gptq":
                    qc.update(sym=sym, desc_act=act, checkpoint_format=fmt)
                else:
                    qc.update(zero_point=not sym, version="gemm")
                cfg["quantization_config"] = qc
            with open(os.path.join(out, "config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
            write_st(os.path.join(out, "model.safetensors"), tensors)
        print("wrote", name)


if __name__ == "__main__":
    main()
