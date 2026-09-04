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
    const pipeline = this.device.createComputePipeline({
      label, layout: 'auto', compute: { module, entryPoint: 'main' },
    });
    this.pipelines.set(code, pipeline);
    return pipeline;
  }

  /** Run one kernel over `buffers`, dispatching a grid of workgroups. */
  dispatch(code, buffers, grid, label = 'kernel') {
    const pipeline = this.pipeline(code, label);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = this.device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(grid[0], grid[1] ?? 1, grid[2] ?? 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
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
