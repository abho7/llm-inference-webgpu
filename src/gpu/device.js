// WebGPU device setup and the buffer plumbing every kernel needs.
//
// Deliberately thin. The interesting code is the WGSL in kernels.js; this file
// exists so that the kernels can be read without wading through bind group
// boilerplate, and so that buffer creation and readback happen in exactly one
// place.
//
// Precision, stated once: weights live on the GPU as f16 and activations as
// f32, with every dot product accumulated in f32. That is the usual
// mixed-precision arrangement, and it is chosen rather than inherited --
// weights are the bandwidth bottleneck at batch one, activations are not, and
// accumulating a 896-term sum in f16 would lose far more than storing its
// inputs in f16 does. No weight in this model exceeds 214 in magnitude against
// an f16 maximum of 65504, so the conversion cannot overflow.

export const REQUIRED_LIMITS = {
  maxStorageBufferBindingSize: 1 << 30,
  maxBufferSize: 1 << 30,
};

export class GpuContext {
  constructor(device, adapter, features) {
    this.device = device;
    this.adapter = adapter;
    this.features = features;
    this.pipelines = new Map();
    this.bytesUploaded = 0;
    // A shader that fails to compile still yields a pipeline, and dispatching
    // it quietly leaves the output buffer at zero -- which reads downstream as
    // a plausible-looking wrong answer rather than an error. Compilation
    // diagnostics are collected here and surfaced by assertShadersCompiled().
    this.shaderErrors = [];
    this.compilations = [];
    // When set, dispatch() records into this encoder instead of submitting.
    this.recording = null;
    this.submissions = 0;
  }

  static async create({ requireF16 = true } = {}) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');

    if (requireF16 && !adapter.features.has('shader-f16')) {
      // Refuse rather than silently falling back to f32 storage: the memory
      // budget and every reported number would be different.
      throw new Error('this adapter has no shader-f16, which the engine stores weights in');
    }
    const wanted = ['shader-f16', 'subgroups', 'timestamp-query'];
    const requiredFeatures = wanted.filter((f) => adapter.features.has(f));

    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize, REQUIRED_LIMITS.maxStorageBufferBindingSize,
        ),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, REQUIRED_LIMITS.maxBufferSize),
      },
    });

    // An uncaptured error is a shader or binding bug, and without this handler
    // it prints to the console and the computation quietly returns zeros.
    device.addEventListener('uncapturederror', (event) => {
      console.error('[webgpu] uncaptured error:', event.error.message);
    });

    return new GpuContext(device, adapter, new Set(requiredFeatures));
  }

  /** A storage buffer holding `data`, which may be any typed array. */
  upload(data, label, extraUsage = 0) {
    const buffer = this.device.createBuffer({
      label,
      size: Math.max(4, align4(data.byteLength)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        | extraUsage,
    });
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    this.bytesUploaded += data.byteLength;
    return buffer;
  }

  /** An empty storage buffer of `bytes`. */
  empty(bytes, label) {
    return this.device.createBuffer({
      label,
      size: Math.max(4, align4(bytes)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /** A small uniform buffer of u32 or f32 parameters. */
  uniform(values, label) {
    const data = values instanceof Uint32Array ? values : new Uint32Array(values);
    const buffer = this.device.createBuffer({
      label,
      size: Math.max(16, align4(data.byteLength)),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /**
   * A compute pipeline for `code`, cached by source text.
   *
   * Compiling WGSL is slow enough that recompiling the same kernel per layer
   * would dominate a decode step, and every layer runs the same shaders.
   */
  pipeline(code, label) {
    const hit = this.pipelines.get(code);
    if (hit) return hit;
    const module = this.device.createShaderModule({ code, label });
    this.compilations.push(
      module.getCompilationInfo().then((info) => {
        for (const message of info.messages) {
          if (message.type !== 'error') continue;
          const source = code.split(String.fromCharCode(10));
          const line = (source[message.lineNum - 1] ?? '').trim();
          this.shaderErrors.push(
            `${label ?? 'kernel'} line ${message.lineNum}: ${message.message} | ${line}`,
          );
        }
      }),
    );
    const pipeline = this.device.createComputePipeline({
      label, layout: 'auto', compute: { module, entryPoint: 'main' },
    });
    this.pipelines.set(code, pipeline);
    return pipeline;
  }

  /**
   * Start recording. Until flush(), dispatch() and copy() append to a single
   * command encoder rather than submitting one at a time.
   *
   * This matters more than it sounds. A decode step is about 360 dispatches,
   * and submitting each one separately means 360 round trips into the driver
   * for roughly 100 microseconds of actual work apiece. Measurement put 77% of
   * a decode step outside the kernels; this is where that went.
   */
  begin(label = 'forward') {
    if (this.recording) throw new Error('already recording');
    this.recording = this.device.createCommandEncoder({ label });
    return this.recording;
  }

  /** Submit everything recorded since begin(). */
  flush() {
    if (!this.recording) return;
    const encoder = this.recording;
    this.recording = null;
    this.device.queue.submit([encoder.finish()]);
    this.submissions++;
  }

  /** Copy between buffers, recorded if recording and submitted otherwise. */
  copy(src, srcOffset, dst, dstOffset, bytes) {
    if (this.recording) {
      this.recording.copyBufferToBuffer(src, srcOffset, dst, dstOffset, bytes);
      return;
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, bytes);
    this.device.queue.submit([encoder.finish()]);
    this.submissions++;
  }

  /** Run one kernel over `buffers`, dispatching a grid of workgroups. */
  dispatch(code, buffers, grid, label = 'kernel') {
    const pipeline = this.pipeline(code, label);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = this.recording ?? this.device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(grid[0], grid[1] ?? 1, grid[2] ?? 1);
    pass.end();
    if (!this.recording) {
      this.device.queue.submit([encoder.finish()]);
      this.submissions++;
    }
  }

  /** Copy a buffer back to the CPU as f32. */
  async readF32(buffer, floatCount) {
    const bytes = floatCount * 4;
    const staging = this.device.createBuffer({
      size: align4(bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, align4(bytes));
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    // slice() before unmap: the mapped range is detached on unmap.
    const out = new Float32Array(staging.getMappedRange(0, align4(bytes)).slice(0, bytes));
    staging.unmap();
    staging.destroy();
    return out.subarray(0, floatCount);
  }

  async done() { await this.device.queue.onSubmittedWorkDone(); }

  /**
   * Throw if any shader compiled so far reported an error.
   *
   * Worth calling before believing any result: a kernel that fails to compile
   * does not fail loudly, it writes nothing, and a buffer of zeros propagates
   * as a wrong answer rather than as a crash.
   */
  async assertShadersCompiled() {
    await Promise.all(this.compilations);
    if (this.shaderErrors.length) {
      const sep = String.fromCharCode(10) + '  ';
      throw new Error(`shader compilation failed:${sep}${this.shaderErrors.join(sep)}`);
    }
  }

  /**
   * Run `count` dispatches inside one submission, timing each on the GPU.
   *
   * `build` is called with an encoder and an index and should record one
   * compute pass per call. Timestamps come from the GPU's own clock rather than
   * from wall time on the CPU, which is the only way to separate how long a
   * kernel takes from how long the browser took to hand it over.
   *
   * Returns nanoseconds per dispatch. Browsers deliberately coarsen this clock,
   * so the caller is expected to check the granularity rather than trust the
   * digits: see bench/granularity below.
   */
  async timed(count, build) {
    if (!this.features.has('timestamp-query')) {
      throw new Error('this adapter has no timestamp-query');
    }
    const querySet = this.device.createQuerySet({ type: 'timestamp', count: count * 2 });
    const resolved = this.device.createBuffer({
      size: count * 2 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const staging = this.device.createBuffer({
      size: count * 2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder({ label: 'timed' });
    for (let i = 0; i < count; i++) {
      build(encoder, i, {
        querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1,
      });
    }
    encoder.resolveQuerySet(querySet, 0, count * 2, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, staging, 0, count * 2 * 8);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const stamps = new BigInt64Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    resolved.destroy();
    querySet.destroy();

    const out = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = Number(stamps[i * 2 + 1] - stamps[i * 2]);
    }
    return out;
  }

  /**
   * Record one compute pass containing `repeats` dispatches of the same kernel.
   *
   * This exists because the browser coarsens the GPU clock -- on this adapter to
   * multiples of about 65 microseconds -- which is far longer than any single
   * kernel here takes. Timing one dispatch therefore reads as either zero or one
   * whole quantum, and neither is the answer. Timing hundreds inside one pass
   * and dividing brings the quantisation error down to a few nanoseconds per
   * call.
   *
   * The dispatches all write the same buffer, so the implementation has to order
   * them; they cannot overlap and be counted once.
   */
  encodeRepeated(encoder, code, buffers, grid, repeats, timestampWrites, label = 'kernel') {
    const pipeline = this.pipeline(code, label);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass(timestampWrites ? { label, timestampWrites } : { label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < repeats; i++) {
      pass.dispatchWorkgroups(grid[0], grid[1] ?? 1, grid[2] ?? 1);
    }
    pass.end();
  }

  /** Record one timed compute pass for `code` over `buffers`. */
  encodeDispatch(encoder, code, buffers, grid, timestampWrites, label = 'kernel') {
    const pipeline = this.pipeline(code, label);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass(timestampWrites ? { label, timestampWrites } : { label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(grid[0], grid[1] ?? 1, grid[2] ?? 1);
    pass.end();
  }

  describe() {
    const l = this.adapter.limits;
    return {
      features: [...this.features],
      maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
    };
  }
}

function align4(n) { return (n + 3) & ~3; }

/**
 * Split a row count into a 2D workgroup grid.
 *
 * maxComputeWorkgroupsPerDimension is 65535 on this adapter, and the output
 * projection has 151936 rows, so a one-dimensional dispatch cannot address it.
 * The kernels recover the row as x + y * width.
 */
export function grid2d(count, width = 32768) {
  return { grid: [Math.min(count, width), Math.ceil(count / width)], width };
}
