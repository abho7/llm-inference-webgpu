"""Record what ONNX Runtime computes, so the JavaScript engine can be checked.

This is the Phase 1 oracle. ONNX Runtime executing the same weights is an
independent implementation of the same architecture -- a genuinely different
codebase in a different language -- which makes it a real check rather than a
second copy of my own arithmetic.

The useful part is that the export publishes `present.N.key` and
`present.N.value` for all 24 layers as ordinary graph outputs. Those are the
post-RoPE keys and the values at every position of every layer, so matching
them checks the embedding, both norms, the Q/K/V projections, the rotary
embedding and the grouped-query head layout *per layer*, without any graph
surgery. If layer 7's keys agree, everything feeding layer 7 agreed.

Output per case:
  <n>.bin   f32 little-endian: present.0.key, present.0.value, ... , then the
            logits at the final position
  and one manifest.json describing all of it.

The model is ~2 GB of fp32 and this machine has less than that free, so the
session is built once, every case is run through it, and it is released. Run
this once; nothing in src/ ever needs it again.
"""
import io
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "golden" / "reference"

# The oracle model is not in the repository: it is ~2 GB, and *.onnx is
# gitignored. Point this at wherever it was downloaded.
MODEL = pathlib.Path(os.environ.get(
    "ONNX_ORACLE",
    r"C:\Users\satya\AppData\Local\Temp\claude\C--Users-satya"
    r"\ae58224e-4cb9-461c-bef2-1ab46f9420ae\scratchpad\oracle\model.onnx",
))

PROMPTS = [
    # Short, because every extra token multiplies 48 present tensors. Chosen to
    # differ in shape rather than in topic: a single token exercises the
    # degenerate sequence, the chat markup exercises the added-token path, and
    # the repeated word gives attention several identical keys to separate.
    "The capital of France is",
    "A",
    "def fibonacci(n):",
    "the the the the the",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]


def main():
    from tokenizers import Tokenizer
    import onnxruntime as ort

    if not MODEL.exists():
        sys.exit(f"oracle model not found at {MODEL}; set ONNX_ORACLE")

    tok = Tokenizer.from_file(str(ROOT / "weights" / "tokenizer.json"))
    config = json.loads((ROOT / "weights" / "config.json").read_text())
    layers = config["num_hidden_layers"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    print(f"loading {MODEL.name} ({MODEL.stat().st_size / 1e6:.0f} MB)...", flush=True)
    t0 = time.time()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cases = []

    for index, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt, add_special_tokens=False).ids
        seq = len(ids)
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, seq), dtype=np.int64),
            "position_ids": np.arange(seq, dtype=np.int64)[None, :],
        }
        # No past: this is a prefill over the whole prompt at once.
        for n in range(layers):
            empty = np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
            feed[f"past_key_values.{n}.key"] = empty
            feed[f"past_key_values.{n}.value"] = empty

        t0 = time.time()
        outputs = session.run(None, feed)
        elapsed = time.time() - t0
        named = dict(zip([o.name for o in session.get_outputs()], outputs))

        blocks = []
        for n in range(layers):
            for kind in ("key", "value"):
                tensor = named[f"present.{n}.{kind}"]
                expected = (1, kv_heads, seq, head_dim)
                assert tensor.shape == expected, f"present.{n}.{kind} {tensor.shape}"
                blocks.append(np.ascontiguousarray(tensor[0], dtype="<f4").ravel())

        logits = named["logits"]
        assert logits.shape == (1, seq, config["vocab_size"]), logits.shape
        last = np.ascontiguousarray(logits[0, -1], dtype="<f4")
        blocks.append(last)

        path = OUT / f"{index}.bin"
        with open(path, "wb") as f:
            for block in blocks:
                f.write(block.tobytes())

        order = np.argsort(last)[::-1][:10]
        cases.append({
            "index": index,
            "prompt": prompt,
            "input_ids": ids,
            "seq": seq,
            "bin": path.name,
            "present_floats_per_tensor": kv_heads * seq * head_dim,
            "logits_floats": int(last.size),
            # Recorded so a mismatch can be described in words, and so the
            # greedy continuation can be checked without re-running the oracle.
            "argmax": int(order[0]),
            "argmax_token": tok.decode([int(order[0])]),
            "top10": [[int(i), float(last[i])] for i in order],
            "logit_min": float(last.min()),
            "logit_max": float(last.max()),
            "seconds": round(elapsed, 3),
        })
        print(f"[{index}] {prompt[:40]!r} seq={seq} -> "
              f"{tok.decode([int(order[0])])!r} ({elapsed:.1f}s)", flush=True)

    manifest = {
        "source": "onnx-community/Qwen2.5-0.5B-Instruct model.onnx (fp32)",
        "produced_by": "oracle/dump_reference.py (onnxruntime, independent of src/)",
        "onnxruntime_version": ort.__version__,
        "layout": (
            "for each of %d layers: present.N.key then present.N.value, each "
            "[kv_heads=%d, seq, head_dim=%d] f32 little-endian; then the logits "
            "at the final position, [vocab] f32"
        ) % (layers, kv_heads, head_dim),
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": config["vocab_size"],
        "cases": cases,
    }
    with io.open(OUT / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")

    total = sum((OUT / c["bin"]).stat().st_size for c in cases)
    print(f"\n{len(cases)} cases, {total / 1e6:.1f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
