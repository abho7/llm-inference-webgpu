"""Measure ONNX Runtime on this machine, as the baseline to be judged against.

Comparing against a real inference runtime is the point. It is a mature,
heavily optimised implementation of the same model, so the number it produces
is what "fast" means here -- and if the engine in this repository loses, the
honest thing is to say by how much and why rather than to pick a softer
opponent.

Two numbers, measured separately because they are different problems:

  time to first token   prefill over the whole prompt in one pass, which is
                        compute-bound and parallel over positions

  decode throughput     one token at a time with the cache fed back in, which
                        is memory-bound and has no parallelism to exploit

The caveat, stated because it flatters nobody: this feeds the past keys and
values in and out as ordinary tensors on every step, so each token copies the
whole cache across the API boundary twice. ONNX Runtime supports binding those
buffers in place, which avoids the copies. This is therefore a *lower* bound on
what ONNX Runtime can do, and the engine in this repository has to beat a
handicapped opponent to mean anything.

  python oracle/bench_ort.py [decode_tokens]
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = pathlib.Path(os.environ.get(
    "ONNX_ORACLE",
    r"C:\Users\satya\AppData\Local\Temp\claude\C--Users-satya"
    r"\ae58224e-4cb9-461c-bef2-1ab46f9420ae\scratchpad\oracle\model.onnx",
))

PROMPT = (
    "The theory of computation asks what problems can be solved by machines, and how much "
    "time and memory the solutions need. A Turing machine is deliberately impoverished: a "
    "tape, a head that reads and writes one symbol at a time, and a finite table of rules."
)


def main():
    from tokenizers import Tokenizer
    import onnxruntime as ort

    decode_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    if not MODEL.exists():
        sys.exit(f"oracle model not found at {MODEL}; set ONNX_ORACLE")

    tok = Tokenizer.from_file(str(ROOT / "weights" / "tokenizer.json"))
    config = json.loads((ROOT / "weights" / "config.json").read_text())
    layers = config["num_hidden_layers"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - t0

    ids = tok.encode(PROMPT, add_special_tokens=False).ids
    print(f"onnxruntime {ort.__version__}, CPUExecutionProvider")
    print(f"model loaded in {load_seconds:.1f}s")
    print(f"prompt: {len(ids)} tokens, then {decode_tokens} decoded\n")

    def empty_past():
        return {
            f"past_key_values.{n}.{kind}":
                np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
            for n in range(layers) for kind in ("key", "value")
        }

    # A warm-up pass, so the first measured number is not paying for lazy
    # allocation and page faults that no later token pays for.
    warm = {"input_ids": np.array([ids[:4]], dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "position_ids": np.arange(4, dtype=np.int64)[None, :]}
    warm.update(empty_past())
    session.run(None, warm)

    # ---- time to first token: one prefill pass over the whole prompt ----
    feed = {"input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "position_ids": np.arange(len(ids), dtype=np.int64)[None, :]}
    feed.update(empty_past())

    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    ttft = time.perf_counter() - t0

    names = [o.name for o in session.get_outputs()]
    named = dict(zip(names, outputs))
    past = {f"past_key_values.{n}.{kind}": named[f"present.{n}.{kind}"]
            for n in range(layers) for kind in ("key", "value")}
    token = int(named["logits"][0, -1].argmax())
    generated = [token]

    print(f"time to first token   {ttft * 1000:8.1f} ms   "
          f"({len(ids)} tokens prefilled, {len(ids) / ttft:.1f} tok/s)")

    # ---- decode: one token per pass, cache fed back in ----
    step_seconds = []
    for i in range(decode_tokens):
        position = len(ids) + i
        feed = {"input_ids": np.array([[token]], dtype=np.int64),
                "attention_mask": np.ones((1, position + 1), dtype=np.int64),
                "position_ids": np.array([[position]], dtype=np.int64)}
        feed.update(past)

        t0 = time.perf_counter()
        outputs = session.run(None, feed)
        step_seconds.append(time.perf_counter() - t0)

        named = dict(zip(names, outputs))
        past = {f"past_key_values.{n}.{kind}": named[f"present.{n}.{kind}"]
                for n in range(layers) for kind in ("key", "value")}
        token = int(named["logits"][0, -1].argmax())
        generated.append(token)

    step_seconds = np.array(step_seconds)
    print(f"decode, median step   {np.median(step_seconds) * 1000:8.1f} ms   "
          f"({1 / np.median(step_seconds):.2f} tok/s)")
    print(f"decode, mean step     {step_seconds.mean() * 1000:8.1f} ms   "
          f"({1 / step_seconds.mean():.2f} tok/s)")
    print(f"decode, slowest step  {step_seconds.max() * 1000:8.1f} ms")
    print(f"\ncontinuation: {tok.decode(generated)!r}")
    print("\nnote: the past keys and values are passed as ordinary tensors, so every step")
    print("copies the whole cache in and out. Binding them in place would be faster, which")
    print("makes this a lower bound on ONNX Runtime rather than its best.")


if __name__ == "__main__":
    main()
