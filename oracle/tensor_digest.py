"""Digest every tensor in model.safetensors using an independent parse.

This is one half of the Phase 0 gate. NumPy parses the same container and
widens the same bfloat16 by the same definition, and the digests it writes are
then recomputed by the JavaScript reader in test/safetensors.test.js. Agreement
on all 290 tensors is a bit-level statement, not a spot check: SHA-256 over the
widened f32 bytes catches a wrong offset, a transposed shape, a truncated read
and a mis-widened exponent alike.

NumPy has no bfloat16, so the widening here is written out explicitly rather
than delegated -- which is the point. If it were delegated to the same code the
JavaScript calls, the comparison would prove nothing.
"""
import hashlib
import io
import json
import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / "weights" / "model.safetensors"
OUT = ROOT / "golden" / "tensor_digest.json"


def main() -> None:
    with open(MODEL, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        header.pop("__metadata__", None)

        entries = {}
        for name in sorted(header):
            spec = header[name]
            begin, end = spec["data_offsets"]
            assert spec["dtype"] == "BF16", f"{name}: unexpected dtype {spec['dtype']}"

            f.seek(data_start + begin)
            raw = f.read(end - begin)
            assert len(raw) == end - begin, f"{name}: short read"

            # bfloat16 -> float32 is the top 16 bits of the f32, so widening is
            # a shift into the high half of a u32 reinterpreted as f32.
            bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << np.uint32(16)
            wide = bits.view("<f4")

            entries[name] = {
                "dtype": spec["dtype"],
                "shape": spec["shape"],
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "f32_sha256": hashlib.sha256(wide.tobytes()).hexdigest(),
                # Kept human-readable so a mismatch says *how* it is wrong.
                "min": float(wide.min()),
                "max": float(wide.max()),
                "nonfinite": int((~np.isfinite(wide)).sum()),
            }

    doc = {
        "source": "Qwen/Qwen2.5-0.5B-Instruct model.safetensors",
        "produced_by": "oracle/tensor_digest.py (numpy, independent of src/core)",
        "file_sha256": _file_sha256(MODEL),
        "file_bytes": MODEL.stat().st_size,
        "tensor_count": len(entries),
        "tensors": entries,
    }
    OUT.parent.mkdir(exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as out:
        json.dump(doc, out, indent=1, sort_keys=True)
        out.write("\n")

    total = sum(t["shape"] and int(np.prod(t["shape"])) or 0 for t in entries.values())
    print(f"{len(entries)} tensors, {total:,} parameters -> {OUT.relative_to(ROOT)}")
    print(f"nonfinite values across the whole model: "
          f"{sum(t['nonfinite'] for t in entries.values())}")


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
