# llm-inference-webgpu

A transformer inference engine written from scratch, running
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), with
every correctness claim checked against an independent implementation and every
performance claim measured on named hardware.

No PyTorch, no `transformers`, no tokenizer library. The safetensors container,
the bfloat16 conversion, the BPE tokenizer, the attention kernels and the
quantization are all in this repository.

> **Status: Phase 2 of 6.** The model runs, caches its keys and values, and
> generates text that matches an independent implementation token for token. It
> is still slow: there is no GPU backend yet, so everything below runs in scalar
> JavaScript. The table marks what is actually done, and nothing is claimed
> before it is measured.

## What this sets out to prove

Generating plausible text proves nothing -- the reference implementations
already do that. The claims worth making are narrower and checkable:

| # | Claim | Phase | Status |
|---|-------|-------|--------|
| 1 | Weights load bit-exactly, and the config matches them | 0 | **done** |
| 2 | Tokenization is identical to the reference on a corpus | 0 | **done** |
| 3 | Logits match ONNX Runtime to a stated tolerance, per layer | 1 | **done** |
| 4 | Incremental decode with a KV cache equals full recomputation | 2 | **done** |
| 5 | The WebGPU backend matches the CPU reference within f16 tolerance | 3 | not started |
| 6 | Quantization error is characterised per layer, not just end to end | 4 | not started |
| 7 | Throughput and time-to-first-token, measured, against a baseline | 5 | not started |

## Phase 2: the KV cache, checked exactly

Attention at position *t* needs every key and value up to *t*. Phase 1
recomputed them all on every step, which is why it managed 0.06 tokens per
second. The cache keeps them instead.

**The gate admits no tolerance.** Position *s*'s key depends only on positions
up to *s*, so the value computed while *s* was the newest token is the value a
full recomputation produces later -- the same bits, not merely close. So the
comparison is on f32 bit patterns, over all 48 per-layer tensors and all
151,936 logits, for three different ways of splitting the same sequence:

| split | passes | keys and values | logits |
|---|---|---|---|
| one token at a time | 7 | identical | identical |
| prefill then decode | 3 | identical | identical |
| uneven chunks | 3 | identical | identical |

What makes that achievable rather than lucky is that there is only one code
path. Passing no cache means "use a fresh one", not "take a different branch",
so cached and uncached decoding cannot drift apart -- they run the same
arithmetic over the same numbers in the same order.

The bug this is really testing for is the rotary position. Attention must rotate
by the absolute position in the conversation, not the offset within the current
call. Using the offset works perfectly on the first pass and silently corrupts
every one after it, and it is invisible to any check that only looks at prefill.

**Grouped-query attention gets its own gate.** Qwen2.5-0.5B has 14 query heads
and 2 key/value heads, so seven query heads share each. The definition
materialises 14 key/value heads by repeating each one seven times and runs
ordinary multi-head attention; `src/core/attention.js` skips the copy and
indexes instead. `test/attention.test.js` runs both and requires exact equality,
across prefill, cache-decode, mixed shapes, no grouping at all, and every head
sharing one. The grouping is blocked (heads 0-6 read kv head 0), and the
interleaved alternative -- `h % numKVHeads`, which looks just as natural --
is checked to be a genuinely different mapping so the test is not vacuous.

Attention is also checked for the properties a masking bug breaks: perturbing
the last position must leave every earlier output untouched, the first position
returns its own value verbatim, and every output lies inside the range of the
values it can see.

**What it buys**, generating 6 tokens from the same prompt:

| | tokens/s | time |
|---|---|---|
| with cache | 0.17 | 36.2s |
| without | 0.05 | 129.7s |

Same tokens either way, 3.6x faster, and the gap widens with length since the
uncached path is quadratic. A 2048-position cache costs 50 MB.

These are still terrible numbers in absolute terms. Everything is scalar
JavaScript on one core, and the GPU backend is phase 3.

### Weights in memory, or streamed

Streaming every weight from disk per pass moves about 1.5 GB, and with under
2 GB free the page cache cannot absorb it, so decoding becomes disk-bound
rather than compute-bound. `Weights` therefore takes a `resident` option that
holds all 24 layers in memory as bfloat16 -- 716 MB, widened into reusable
buffers on demand at about 700 ms for the whole model.

Streaming stays the default, because the point of it is that the engine runs
at all on a machine that cannot hold the model. `validate/phase2.js --stream`
runs it that way.

### A measurement bug worth recording

An early run of the phase 2 gate reported one 3-pass comparison as taking
52894.1 seconds -- 14.7 hours, in a run that finished in minutes. The cause was
`Date.now()`, which is wall-clock and jumps when the system clock is corrected.
All timings now use `performance.now()`, which is monotonic; the same comparison
then measured 21.6 seconds. A benchmark must never measure itself with a clock
that can move.

## Phase 1: the forward pass agrees with ONNX Runtime

The whole architecture is in `src/cpu/model.js`: RMSNorm, the Q/K/V
projections with Qwen2's asymmetric biases, rotary embeddings, causal
grouped-query attention at seven query heads per key/value head, SwiGLU, and a
tied output projection. About 200 lines, written to be read.

**The check is per layer, not just at the output.** The ONNX export publishes
`present.N.key` and `present.N.value` for all 24 layers as ordinary graph
outputs. Those are the post-RoPE keys and the values at every position of every
layer, so matching them exercises the embedding, both norms, all three
projections, the rotary embedding and the head grouping *per layer* -- and it
needs no graph surgery to get at them. If layer 7 agrees, everything feeding
layer 7 agreed.

Measured over five prompts, 48 tensors each:

| | worst relative | worst absolute | bound |
|---|---|---|---|
| per-layer keys and values | 9.1e-5 | 6.7e-4 | 5e-4 |
| logits at the final position | 6.6e-6 | 1.1e-4 | 1e-4 |
| argmax | identical on 5 of 5 | | exact |
| top-10 ordering | identical on 5 of 5 | | exact |

Both implementations store f32 and sum in different orders, so exact agreement
is impossible; the question is only whether the drift is explainable. A dot
product over 896 terms carries roughly `sqrt(896) * 2^-24`, about 2e-6
relative, and 24 layers compound it. The bounds sit a few times above the
measured worst case: float non-associativity cannot reach them, and a real
defect -- a swapped RoPE convention, a mis-grouped attention head, an epsilon
outside the square root -- overshoots them by orders of magnitude.

The argmax and top-10 checks are the ones with no tolerance at all. The argmax
is what actually decides the generated token, and the top-10 ordering
constrains ten values that often sit within tenths of each other, so a
systematic bias would break it even while every individual logit stayed inside
tolerance.

**End to end**, greedy decoding from `"The capital of France is"` produces
`" Paris. It is the largest city in"` -- the same eight tokens ONNX Runtime
produces, in the same order.

It does so at **0.06 tokens per second**. There is no KV cache, so generating
token *n* recomputes the whole sequence, and every step re-streams all 988 MB
of weights from disk. Both are phase 2 and phase 3 problems, and the number is
recorded here so the improvement has something to be measured against.

### Why the error jumps at layer 22

The per-layer agreement degrades smoothly through the stack and then jumps by
about an order of magnitude at layer 22, in every prompt rather than one. A
jump like that is either a bug in that layer or a property of the model, and
the error alone cannot tell them apart.

`validate/activation-scale.js` measures the statistic that does. Qwen2.5
concentrates its hidden state in a few channels: channel 490 grows steadily
from 3.5 at layer 1 to 66.5 at layer 22, reaching 27 times the RMS of the whole
vector. RMSNorm then divides everything by a number those few channels set, and
the projection that follows sums 896 terms in which one dominates the rest.
That is catastrophic cancellation, and it inflates absolute error without
anything being wrong. The error peaks exactly where the concentration does.

This is not just an explanation, it is a preview: outlier channels like 490 are
precisely what makes per-tensor quantization fail, which is phase 4's problem.

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
src/cpu/      the reference forward pass and KV cache
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
python oracle/dump_reference.py    # golden/reference/, needs the ~2 GB ONNX model
node --test                        # 82 tests
node validate/phase0.js            # loader and tokenizer gate
node validate/phase1.js            # forward-pass gate against ONNX Runtime
node validate/phase2.js            # KV cache gate, bit-exact
node validate/generate.js "The capital of France is" 8
node validate/layer-profile.js 3   # where the disagreement grows
node validate/activation-scale.js  # which channels are outliers
```

The weights are not committed. The golden files are, so the test suite is
meaningful on a fresh clone; the tests that need the weights skip rather than
fail when they are absent.

Requires Node 24 or newer: the f16 tests use `Float16Array` as their oracle,
and it is not available before then. On an older runtime those three tests
skip rather than fail.

## Hardware these numbers refer to

Intel Core i7-1355U, 10 cores / 12 threads, 16 GB RAM, Intel Iris Xe
(`gen-12lp`) with 2 GB of shared VRAM. WebGPU on this adapter reports
`shader-f16`, `subgroups`, `timestamp-query` and `float32-filterable`, a 2 GB
maximum storage buffer and 32 KB of workgroup shared memory. Decode on an
integrated GPU is memory-bandwidth-bound, so the throughput figures in phase 5
will be modest in absolute terms and will say so.

## Licence

MIT.
