#!/usr/bin/env python3
"""Dequantization fixtures from gguf-py's independent NumPy implementations.

For each GGML block type: random block bytes (with the fp16 scale fields set to
sane values so outputs are finite) and the expected fp32 values.

    PYTHONUTF8=1 python tools/make_quant_fixtures.py dynacore/tests/data/quant

Writes <type>.bin (raw blocks) and <type>.f32 (expected little-endian floats).
"""
import os
import sys

import numpy as np
import gguf
import gguf.quants as q
from gguf import GGMLQuantizationType as T

# Byte offsets of fp16 scale/min fields inside one block (GGML block structs).
FP16_FIELDS = {
    T.Q4_0: [0], T.Q4_1: [0, 2], T.Q5_0: [0], T.Q5_1: [0, 2], T.Q8_0: [0],
    T.Q2_K: [80, 82], T.Q3_K: [108], T.Q4_K: [0, 2], T.Q5_K: [0, 2], T.Q6_K: [208],
}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1234)
    for t, fields in FP16_FIELDS.items():
        block_elems, block_bytes = gguf.GGML_QUANT_SIZES[t]
        n_blocks = 8 if block_elems == 32 else 4
        raw = rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)
        for b in range(n_blocks):
            for off in fields:
                v = np.float16(rng.uniform(-0.05, 0.05))
                raw[b, off:off + 2] = np.frombuffer(v.tobytes(), dtype=np.uint8)
        expected = q.dequantize(raw.reshape(-1), t).astype(np.float32).reshape(-1)
        assert expected.size == n_blocks * block_elems and np.all(np.isfinite(expected))
        name = t.name.lower().replace("_k", "_K")
        raw.tofile(os.path.join(out, f"{name}.bin"))
        expected.astype("<f4").tofile(os.path.join(out, f"{name}.f32"))
        print(f"{name}: {n_blocks} blocks, range [{expected.min():.4f}, {expected.max():.4f}]")


if __name__ == "__main__":
    main()
