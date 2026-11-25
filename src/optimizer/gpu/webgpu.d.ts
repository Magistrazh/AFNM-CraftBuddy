/**
 * Minimal local WebGPU declarations.
 *
 * Ported from the browser globals used by tools/gpu-probe/stage4.html and
 * stage7.html (validated prototypes). @webgpu/types is deliberately not a
 * dependency; this file declares only what src/optimizer/gpu actually uses.
 */

declare const GPUBufferUsage: {
  readonly MAP_READ: number;
  readonly MAP_WRITE: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly STORAGE: number;
  readonly UNIFORM: number;
};

declare const GPUMapMode: {
  readonly READ: number;
  readonly WRITE: number;
};

interface GpuAdapterInfo {
  readonly vendor?: string;
  readonly architecture?: string;
  readonly device?: string;
  readonly description?: string;
}

interface GpuSupportedLimits {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly maxComputeInvocationsPerWorkgroup?: number;
  readonly maxComputeWorkgroupSizeX?: number;
  readonly maxComputeWorkgroupsPerDimension?: number;
}

interface GpuAdapter {
  readonly info?: GpuAdapterInfo;
  readonly limits?: GpuSupportedLimits;
  requestDevice(): Promise<GpuDevice>;
}

interface GpuBuffer {
  destroy(): void;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
}

interface GpuBindGroup {}

interface GpuShaderModule {
  getCompilationInfo(): Promise<{
    readonly messages: ReadonlyArray<{
      readonly type: string;
      readonly lineNum: number;
      readonly message: string;
    }>;
  }>;
}

interface GpuComputePipeline {
  getBindGroupLayout(index: number): unknown;
}

interface GpuComputePassEncoder {
  setPipeline(pipeline: GpuComputePipeline): void;
  setBindGroup(index: number, bindGroup: GpuBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GpuCommandEncoder {
  beginComputePass(): GpuComputePassEncoder;
  copyBufferToBuffer(
    source: GpuBuffer,
    sourceOffset: number,
    destination: GpuBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

interface GpuQueue {
  submit(commandBuffers: readonly unknown[]): void;
  onSubmittedWorkDone(): Promise<void>;
  writeBuffer(
    buffer: GpuBuffer,
    bufferOffset: number,
    data: ArrayBufferView | ArrayBuffer,
    dataOffset?: number,
    size?: number,
  ): void;
}

interface GpuDeviceLostInfo {
  readonly reason: string;
  readonly message: string;
}

interface GpuDevice {
  readonly limits?: GpuSupportedLimits;
  readonly lost: Promise<GpuDeviceLostInfo>;
  addEventListener(type: string, listener: (event: unknown) => void): void;
  destroy(): void;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GpuBuffer;
  createShaderModule(descriptor: { code: string }): GpuShaderModule;
  createComputePipelineAsync(descriptor: {
    layout: 'auto';
    compute: { module: GpuShaderModule; entryPoint: string };
  }): Promise<GpuComputePipeline>;
  createBindGroup(descriptor: {
    layout: unknown;
    entries: ReadonlyArray<{ binding: number; resource: { buffer: GpuBuffer } }>;
  }): GpuBindGroup;
  createCommandEncoder(): GpuCommandEncoder;
  readonly queue: GpuQueue;
}

interface GpuApi {
  requestAdapter(options?: {
    powerPreference?: 'low-power' | 'high-performance';
    forceFallbackAdapter?: boolean;
  }): Promise<GpuAdapter | null>;
}

interface Navigator {
  readonly gpu?: GpuApi;
}
