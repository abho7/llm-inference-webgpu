# llm-inference-webgpu

A transformer inference engine written from scratch, running
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), with
every correctness claim checked against an independent implementation and every
performance claim measured on named hardware.

No PyTorch, no `transformers`, no tokenizer library. The safetensors container,
the bfloat16 conversion, the BPE tokenizer, the attention kernels and the
quantization are all in this repository.

> **Status: Phase 0 of 6.** The model loads and the tokenizer is exact. The
> forward pass is not written yet, so this does not generate text. The table
> below marks what is actually done; nothing is claimed before it is measured.

## What this sets out to prove

Generating plausible text proves nothing -- the reference implementations
already do that. The claims worth making are narrower and checkable:

| # | Claim | Phase | Status |
|---|-------|-------|--------|
| 1 | Weights load bit-exactly, and the config matches them | 0 | **done** |
| 2 | Tokenization is identical to the reference on a corpus | 0 | **done** |
| 3 | Logits match ONNX Runtime to a stated tolerance, per layer | 1 | not started |
| 4 | Incremental decode with a KV cache equals full recomputation | 2 | not started |
| 5 | The WebGPU backend matches the CPU reference within f16 tolerance | 3 | not started |
| 6 | Quantization error is characterised per layer, not just end to end | 4 | not started |
| 7 | Throughput and time-to-first-token, measured, against a baseline | 5 | not started |

## Phase 0: what is established, and how

Two gates, both run by `node --test`.

**Every tensor loads bit-exactly.** `oracle/tensor_digest.py` parses the same
988 MB file with NumPy, widens the bfloat16 by the same definition, and records
a SHA-256 of both the raw bytes and the widened f32 for each of the 290
tensors. `test/safetensors.test.js` recomputes all 580 digests through the
JavaScript reader. A wrong offset, a transposed shape, a truncated read and a
mis-widened exponent all fail this, and it shares no code with the thing it
checks.

The float conversions underneath are tested exhaustively rather than sampled --
a 16-bit format has only 65536 inhabitants, so "every value" is a realistic
standard. `f16` is checked in both directions against `Float16Array`, which is
V8's own C++ implementation, on all 65536 round-trip values plus ~196k
midpoint probes that exercise the rounding logic itself.

**Tokenization is identical to the reference.** 3098 cases, 85,633 tokens, zero
differences from Hugging Face `tokenizers` (Rust). The corpus is built to hit
where byte-level BPE actually breaks: NFC normalisation, the contraction group,
whitespace runs, multi-byte characters split mid-character, the 22 added
tokens, and 3000 seeded random strings drawn across scripts.

**The config is checked against the weights, not trusted.** `config.json`
implies exactly which 290 tensors must exist and what shape each must have.
Extra tensors fail too, since an unread tensor means the engine is ignoring
part of the model. The parameter count derived from the config
(494,032,768) is compared against the count summed from the file.

**Loading is lazy.** This machine has under 2 GB of free physical memory, and
materialising the file then widening it to f32 would need about 3 GB. Tensors
are read through a file descriptor as they are needed: `validate/phase0.js`
reports 52 MB resident against a 988 MB model.

### Two real defects the gates caught

Both were found by tests, not by reading code, which is the point of writing
the tests first.

**The split pattern's `\s` does not mean the same thing in both languages.**
`tokenizer.json` was written for Rust, where `\s` is `\p{White_Space}`.
JavaScript's `\s` is a different set: it also matches U+FEFF, the byte order
mark. A faithful-looking transcription retokenized any text containing a BOM,
which showed up as 16 differing corpus cases. Spelling the property out fixes
it, and the disputed characters are now a permanent corpus group.

**`TextDecoder` silently eats a leading byte order mark.** Decoding text that
legitimately begins with U+FEFF returned something shorter than what was
encoded. `{ ignoreBOM: true }` is required, despite its name reading as the
opposite.

A third finding is documented rather than fixed, because it is the platform
rather than a bug: a signalling NaN cannot survive being returned as a JS
`Number`, so the scalar widening path quiets it while the bulk path, which
writes through an aliased `Uint32Array`, is bit-exact. The weight loader uses
the bulk path.

## Design decisions the environment forced

This machine has **no C, C++ or Rust toolchain** and **no PyTorch**, which
settled two things that would otherwise have been arguments:

**The fast backend is WebGPU, not C++.** WGSL kernels are as much from-scratch
as C++ ones, the CPU reference and the GPU engine can be compared inside a
single browser tab rather than across a language boundary, and the result
deploys as a demo that runs in the reader's own browser.

**The oracle is ONNX Runtime.** Without PyTorch there is no convenient
reference, but ONNX Runtime executing the same weights is an *independent
implementation*, which is a better oracle than a second copy of my own code
would have been. It is already confirmed working end to end: a greedy decode of
`"The capital of France is"` produces `" Paris. It is the largest city in"`.

## Layout

```
src/core/     backend-agnostic: dtypes, safetensors, config, tokenizer
src/cpu/      f32 reference forward pass                    (phase 1)
src/gpu/      WebGPU compute backend                        (phase 3)
oracle/       independent implementations, used only to generate golden files
golden/       committed outputs of the oracle scripts
validate/     gate reports; every number recomputed on the spot
test/         node --test
bench/        CPU vs GPU, one tab, one machine              (phase 5)
```

`oracle/` is deliberately quarantined. Nothing in `src/` imports from it, and
it is the only place NumPy or `tokenizers` appear.

## Running it

```bash
python oracle/fetch_weights.py     # ~1 GB from Hugging Face, into weights/
python oracle/tensor_digest.py     # regenerate golden/tensor_digest.json
python oracle/make_corpus.py       # regenerate test/corpus.json
python oracle/tokenize_corpus.py   # regenerate golden/tokenizer_cases.json
node --test                        # 36 tests
node validate/phase0.js            # the gate report
```

The weights are not committed. The golden files are, so the test suite is
meaningful on a fresh clone; the tests that need the weights skip rather than
fail when they are absent.

Requires Node 22 or newer for `Float16Array`.

## Hardware these numbers refer to

Intel Core i7-1355U, 10 cores / 12 threads, 16 GB RAM, Intel Iris Xe
(`gen-12lp`) with 2 GB of shared VRAM. WebGPU on this adapter reports
`shader-f16`, `subgroups`, `timestamp-query` and `float32-filterable`, a 2 GB
maximum storage buffer and 32 KB of workgroup shared memory. Decode on an
integrated GPU is memory-bandwidth-bound, so the throughput figures in phase 5
will be modest in absolute terms and will say so.

## Licence

MIT.
