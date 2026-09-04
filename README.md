# llm-inference-webgpu

A transformer inference engine written from scratch, running
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), with
every correctness claim checked against an independent implementation and every
performance claim measured on named hardware.

No PyTorch, no `transformers`, no tokenizer library. The safetensors container,
the bfloat16 conversion, the BPE tokenizer, the attention kernels and the
quantization are all in this repository.

> **Status: Phase 5 of 6.** The model runs on the GPU, agrees with ONNX Runtime
> per layer, quantizes to 8 and 4 bits with the damage measured, and has now
> been benchmarked against a real inference runtime -- which beats it. By how
> much, and where the time actually goes, is below.

## What this sets out to prove

Generating plausible text proves nothing -- the reference implementations
already do that. The claims worth making are narrower and checkable:

| # | Claim | Phase | Status |
|---|-------|-------|--------|
| 1 | Weights load bit-exactly, and the config matches them | 0 | **done** |
| 2 | Tokenization is identical to the reference on a corpus | 0 | **done** |
| 3 | Logits match ONNX Runtime to a stated tolerance, per layer | 1 | **done** |
| 4 | Incremental decode with a KV cache equals full recomputation | 2 | **done** |
| 5 | The WebGPU backend matches the CPU reference within f16 tolerance | 3 | **done** |
| 6 | Quantization error is characterised per layer, not just end to end | 4 | **done** |
| 7 | Throughput and time-to-first-token, measured, against a baseline | 5 | **done** |

## Phase 5: how fast it is, and losing to ONNX Runtime

The baseline is ONNX Runtime on the CPU of this same machine, running the same
model. It is a mature, heavily optimised implementation, which is the point: an
engine written from scratch should be judged against one, not against a softer
opponent.

| | this engine (WebGPU) | ONNX Runtime (CPU) | ratio |
|---|---|---|---|
| prefill, 52-token prompt | 1233 ms, **42 tok/s** | 218 ms, **238 tok/s** | ORT 5.6x faster |
| decode, median step | 86.8 ms, **11.5 tok/s** | 67.5 ms, **14.8 tok/s** | ORT 1.3x faster |

So it loses, and the prefill gap is the larger one. That is not mysterious:
prefill here runs a matrix-*vector* product per position, with the sequence in
the third dispatch dimension, rather than a real matrix multiply. One kernel
serving both prefill and decode was a deliberate phase 3 choice to keep a single
code path while correctness was being established, and this is what it costs.

The ONNX Runtime figure is itself a lower bound: it passes the past keys and
values as ordinary tensors, so every step copies the whole cache in and out.
Binding them in place would make it faster still.

### Where a decode step actually goes

Kernels are timed on the GPU's own clock. That clock is coarsened by the browser
to a measured quantum of **65.54 us**, far longer than any kernel here takes, so
each kernel is timed as 256 dispatches inside one pass and divided down. The
quantum is not assumed: it is recovered as the greatest common divisor of the
raw pass totals.

| kernel | per token | share of GPU time |
|---|---|---|
| gate/up (4864x896, twice per layer) | 14.72 ms | 40% |
| lm_head (151936x896, once per token) | 9.05 ms | 25% |
| down (896x4864) | 5.92 ms | 16% |
| attention | 1.98 ms | 5% |
| o_proj (896x896) | 1.94 ms | 5% |
| q_proj (896x896) | 1.87 ms | 5% |
| rmsnorm, k_proj, rope, swiglu, add | 1.01 ms | 3% |

| | |
|---|---|
| GPU time accounted for | 36.5 ms |
| logits readback, 608 KB | 5.8 ms |
| measured decode step | 86.8 ms |
| **unaccounted** | **44.5 ms (51%)** |

The MLP is 56% of GPU time across three matrices, and the output projection
alone is a quarter -- worth remembering that this model ties its embeddings, so
that 151936x896 matrix is 27% of the parameters and gets read once per token to
produce a single argmax.

### A hypothesis I tested and had to drop

Half the decode step was outside the kernels, and the obvious explanation was
submission overhead: the engine created a command encoder and submitted it for
every single dispatch, roughly 400 submissions per token. Batching the whole
forward pass into one command buffer took that to exactly 1.

It did not measurably speed anything up. Decode steps across five runs came out
at 175, 151, 141 and 87 ms, with the two fastest and the slowest both on the
batched code. The change is still in -- 400 submissions per token is
indefensible regardless -- but the honest conclusion is that submission count
was not the bottleneck, and I would have reported it as one if I had shipped the
optimisation without re-measuring.

### The numbers move, and pretending otherwise would be worse

The per-kernel table is a median of three rounds, and it reports every round,
because an early version reported a single reading and a later run disagreed
with it by a factor of four on the same kernel with unchanged code.

The instability is specific and explainable. Most kernels repeat tightly --
q_proj measured 272, 276, 272 us across rounds. The two exceptions are the
largest matrices: gate/up went 1080, 430, 296 us and lm_head went 12.4, 9.4,
9.2 ms, each settling downward. This is an integrated GPU allocating out of
system memory on a machine with about 1.3 GB free, so the first touch of a large
weight matrix pays for residency and later touches do not.

Prefill, by contrast, is stable to under 1% within a run (1226, 1233, 1235 ms),
because a 52-token prefill touches everything several times over.

Decode across whole runs spans 87 to 175 ms, a factor of two. The best figure is
quoted above and the range is quoted here; quoting only the first would be a
coin flip presented as a measurement.

## Phase 4: quantization, and a prediction that turned out wrong

Two weight-only schemes, both written here: INT8 symmetric per output channel,
and INT4 asymmetric per group of weights along the input dimension, packed two
codes to a byte. Activations stay f32.

### What it costs in weight space

`validate/quantization-error.js` measures every one of the 169 weight matrices
under every scheme. Relative Frobenius error, averaged by role:

| role | f16 | int8 | int4-g128 | int4-g64 | int4-g32 |
|---|---|---|---|---|---|
| embed | 2.6e-8 | 9.0e-3 | 1.05e-1 | 9.4e-2 | 8.2e-2 |
| q_proj | 1.7e-8 | 9.7e-3 | 1.10e-1 | 9.8e-2 | 8.5e-2 |
| k_proj | 1.4e-8 | 9.8e-3 | 1.13e-1 | 9.9e-2 | 8.6e-2 |
| v_proj | 2.4e-8 | 1.02e-2 | 1.18e-1 | 1.03e-1 | 8.8e-2 |
| o_proj | 2.4e-8 | 1.00e-2 | 1.08e-1 | 9.3e-2 | 8.2e-2 |
| gate_proj | 1.6e-8 | 8.9e-3 | 1.07e-1 | 9.5e-2 | 8.3e-2 |
| up_proj | 1.9e-8 | 8.9e-3 | 1.06e-1 | 9.5e-2 | 8.3e-2 |
| down_proj | 2.1e-8 | 1.23e-2 | 1.09e-1 | 9.6e-2 | 8.4e-2 |

The v_proj matrices are consistently the hardest, and the layer-16 one is the
worst in the model. Storage, counting the scales and zero points rather than
quoting the nominal bit width:

| scheme | bits/weight | model |
|---|---|---|
| f16 | 16.000 | 988 MB |
| int8 | 8.036 | 496 MB |
| int4-g128 | 4.500 | 278 MB |
| int4-g32 | 6.000 | 371 MB |

### What it costs in answers

Weight error only matters through its effect on output. Perplexity under a
protocol fixed before running -- same passage, same positions, full left
context, nothing scored twice -- with a paired comparison, because every scheme
scores identical positions and an aggregate difference of a fraction of a
percent means nothing on its own.

| weights | perplexity | mean NLL vs f16 | oracle argmax |
|---|---|---|---|
| f16 | 14.71 | baseline | 5/5 |
| int8 | 14.65 | -0.0038 +- 0.0136 (0.3 SE) | 4/5 |
| int4-g128 | 22.28 | **+0.4155 +- 0.1668 (2.5 SE)** | 3/5 |

int8 came out *lower* than f16, and the paired test is what stops that becoming
a claim: at 0.3 standard errors it is noise, and the honest statement is that
int8 does not measurably hurt this model, not that it helps it.

int4 is a different story. At 4.5 bits per weight the model loses 0.42 nats per
token, a 2.5-standard-error effect, and it stops agreeing with the oracle on
which token comes next for two of five prompts. A longer run in the browser
harness, over 183 positions, found the same shape: +38.8% perplexity at g128
and +27.0% at g32. Smaller groups help and do not rescue it. At 0.5B parameters
there is not enough redundancy to absorb 4-bit rounding, which is consistent
with the literature but worth having measured rather than cited.

### The prediction from phase 1 was wrong

Phase 1 found channel 490 growing to 27 times the RMS of the hidden state, and
I wrote that outlier channels like it are "precisely what makes per-tensor
quantization fail". Phase 4 says otherwise.

Weight-only quantization reaches the output as `dy[r] = sum over c of
dW[r,c] * x[c]`, so each channel's weight error is weighted by that channel's
activation. `validate/quantization-outliers.js` measures the resulting
per-channel damage. It is concentrated -- the worst single channel carries 1.0
to 2.3% of the total where an even split would give 0.112%, so 9 to 20 times
its share -- but channel 490 is almost never the culprit. Channel 570 tops the
list in 10 of 25 cases.

The reason is in the normalisation gain, and it is unambiguous:

| layer | gain[490] | mean abs gain | ratio | rank of 896 |
|---|---|---|---|---|
| 0 | 0.5039 | 0.0695 | 7.25 | 4 |
| 8 | 0.0486 | 1.0842 | 0.045 | **896** |
| 15 | -0.0630 | 1.5965 | 0.039 | **896** |
| 22 | 0.0845 | 1.5865 | 0.053 | **896** |
| 23 | 0.4980 | 1.7206 | 0.289 | 896 |

At every layer past the first, channel 490 has the *smallest* normalisation
gain of all 896. The model makes that channel massive in the residual stream
and then suppresses it before any projection reads it, so a projection never
sees the outlier at all.

That distinction is the real result. The massive activation lives in the
residual stream, and weight-only quantization does not touch the residual
stream. It would matter for quantizing *activations*, which is where the
outlier problem in the literature actually sits. It does not matter much for
quantizing weights, and phase 1 was too quick to assume it would.

## Phase 3: the WebGPU kernels

Seven compute kernels in WGSL, one per operation in the forward pass, each a
direct translation of the corresponding function in `src/core/ops.js`. No
fusion and no tiling yet: phase 3's claim is that the GPU agrees with the CPU,
and every optimisation added before that is established is a place for a
discrepancy to hide.

`web/kernels.html` runs each kernel on the GPU and the CPU function on the same
inputs, in the same tab. Two comparisons per kernel, which separate two
different questions:

| kernel | shape | vs same-f16 CPU | vs full-f32 CPU |
|---|---|---|---|
| matvec | 896x896 + bias | 1.08e-7 | 1.44e-4 |
| matvec | 151936x32, 2D dispatch | 9.43e-8 | 1.74e-4 |
| rmsnorm | 5x896 | 7.13e-8 | 2.51e-4 |
| rope | 6x14x64, from position 3 | 8.97e-8 | -- |
| attention | seq 3, past 5, 14/2 heads | 1.64e-7 | -- |
| swiglu | 4864 elements | 8.58e-8 | -- |
| add | 4480 elements | 0 | -- |

The middle column feeds the CPU the *same f16-rounded weights* the GPU holds,
so it isolates whether the kernel is correct: everything lands at 1e-7, which
is f32 accumulation order and nothing else.

The right column is easy to misread, and I nearly did. It says that rounding a
matrix of **random f32 values** to f16 moves the answer by about 2e-4 — which
is true, and is not what happens to this model, because these weights are not
f32. They are bfloat16: 7 mantissa bits against f16's 10. The mantissa is
*wider* in f16, so the conversion cannot lose precision at all; only f16's
narrower exponent range costs anything.

`validate/f16-fidelity.js` walks all 494,032,768 weights:

| | |
|---|---|
| changed by bf16 to f16 | 123,027 (0.0249%) |
| flushed to zero | 1,463 |
| overflowed | 0 |
| worst relative change, normal range | **exactly 0** |
| largest weight | 214, against an f16 ceiling of 65504 |

Every weight f16 can represent normally converts bit-exactly. The 0.0249% that
change are all below 6.1e-5 in magnitude, where f16 goes subnormal and starts
dropping bits. So for this model, f16 storage is very nearly free — which the
end-to-end run below then confirms independently.

Precision is stated rather than inherited: weights f16 in storage, activations
f32, every dot product accumulated in f32. Weights are the bandwidth bottleneck
at batch one and activations are not.

### The whole model on the GPU

`web/model.html` runs the full forward pass on the GPU and compares its
per-layer keys and values against the fp32 ONNX Runtime oracle, the same golden
files phase 1 used.

| prompt | tokens | seconds | worst layer | logits | argmax |
|---|---|---|---|---|---|
| "The capital of France is" | 5 | 1.21 | 9.54e-6 | 2.85e-6 | 12095 agree |
| "A" | 1 | 0.79 | 9.49e-6 | 4.74e-6 | 220 agree |
| "def fibonacci(n):" | 4 | 0.58 | 6.69e-6 | 1.52e-6 | 715 agree |
| "the the the the the" | 5 | 0.62 | 6.43e-5 | 3.96e-6 | 3491 agree |
| chat template, 2+2 | 15 | 0.80 | 1.84e-5 | 2.89e-6 | 17 agree |

Worst per-layer 6.4e-5 relative, worst logit 4.7e-6, and the same next token as
the oracle on all five.

The striking part is that those are *better* than the CPU reference's own
numbers against the same oracle (9.1e-5 per layer, 6.6e-6 on logits), which
looks impossible for a backend using half the precision until the fidelity
measurement above explains it: the f16 weights are the same numbers, and the
GPU accumulates in f32 with a tree reduction whose error grows like log(n)
rather than the sequential sum's n.

988 MB of f16 weights upload in 88 seconds and sit in 2 GB of shared VRAM. The
per-prompt times are prefill only and are not a throughput claim: prefill runs
a matrix-vector product per position rather than a real matrix multiply, the
kernels are unfused, and nothing has been tuned. That is phase 5.

### The rotary embedding could not be computed on the GPU

The rope kernel initially failed, at 2.2e-5 against a 1e-5 bound, while every
other kernel passed at 1e-7. `web/precision.html` measures why rather than
guessing:

| function | max absolute error |
|---|---|
| `cos`, angles up to ~2047 | 5.3e-5 |
| `sin`, angles up to ~2047 | 5.9e-5 |
| `cos`, angles within one turn | 3.0e-5 |
| `sin`, angles within one turn | 3.0e-5 |
| `pow(1e6, -2j/64)` | 1.4e-8 (6.5 ULP) |

WGSL permits relaxed precision on transcendentals, and this adapter uses the
allowance. The small-angle rows are what make the diagnosis: restricting to a
single turn barely helps, so this is the transcendental itself and not argument
reduction, and `pow` being fine at 6.5 ULP rules out the frequency computation.
About 3e-5 is a floor, 300 times worse than every other kernel manages.

So the cosines and sines are precomputed once on the CPU in f64 and uploaded as
a table, and the kernel does nothing but multiply and add. The rope kernel went
from 2.2e-5 to 9.0e-8, a factor of 250. It is also less work per token than
evaluating two transcendentals per element per layer, but that is a side
benefit; the reason is that the GPU cannot compute this accurately enough to be
checked.

### Running the browser side

```bash
node tools/serve.js                       # serves the repo with HTTP range support
# then open, in a WebGPU-capable browser:
#   /web/probe.html      adapter features, limits, and a compute shader with a known answer
#   /web/kernels.html    every kernel against the CPU reference
#   /web/precision.html  what this adapter's transcendentals actually cost
#   /web/model.html      the whole model on the GPU, against the ONNX oracle
#   /web/perplexity.html perplexity under each quantization scheme
#   /web/bench.html      throughput, time to first token, per-kernel breakdown
```

Range support is not incidental: the weights are 988 MB and the page needs
arbitrary slices, exactly as the Node loader reads them through a file
descriptor. `src/core/source-web.js` refuses to continue if the server answers
a range request with a 200, because silently downloading the whole model is
worse than failing.

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

I expected this to be a preview of phase 4 -- outlier channels being what makes
quantization fail. Phase 4 measured it and the prediction was wrong; the
correction is in that section.

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
src/core/     backend-agnostic: dtypes, safetensors, config, tokenizer, quantization
src/cpu/      the reference forward pass and KV cache
src/gpu/      WebGPU device plumbing and the WGSL kernels
web/          browser harnesses: probe, kernel comparison, precision
tools/        a static server with range support, for the browser harnesses
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
node --test                        # 101 tests
node validate/phase0.js            # loader and tokenizer gate
node validate/phase1.js            # forward-pass gate against ONNX Runtime
node validate/phase2.js            # KV cache gate, bit-exact
node validate/f16-fidelity.js     # what f16 storage costs these weights
node validate/quantization-error.js    # per-tensor error for every scheme
node validate/quantization-outliers.js # which channels carry the damage
node validate/perplexity.js       # paired perplexity, f16 vs int8 vs int4
python oracle/bench_ort.py        # the ONNX Runtime baseline to be judged against
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
maximum storage buffer and 32 KB of workgroup shared memory, and coarsens GPU
timestamps to a 65.54 us quantum.

The GPU is integrated, so its memory is system memory, and during these runs the
machine had roughly 1.3 GB free while holding 988 MB of weights resident. That
is the source of the run-to-run variance in phase 5, and it is why the
throughput figures come with a range rather than a single digit.

## Licence

MIT.
