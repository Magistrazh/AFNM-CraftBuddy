/**
 * WebGPU availability probe.
 *
 * Ported from the adapter selection in tools/gpu-probe/stage7.html, with the
 * software-adapter rejection and limit floors required by docs/GPU_BACKEND.md.
 * No device is created here: the probe only asks for an adapter and its limits.
 */

import { gpuDebugLog } from './debug';

export interface GpuAdapterDetails {
  readonly vendor?: string;
  readonly architecture?: string;
  readonly device?: string;
  readonly description?: string;
}

export interface GpuLimits {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeWorkgroupsPerDimension: number;
}

export interface GpuCapability {
  readonly available: boolean;
  readonly reason: string;
  /** Adapter identity exactly as reported by `adapter.info`, when available. */
  readonly adapter?: GpuAdapterDetails;
  readonly limits?: GpuLimits;
}

/** Floors needed by the scoring/beam kernel (workgroup 64, B*K candidates). */
export const REQUIRED_GPU_LIMITS: GpuLimits = {
  maxBufferSize: 16 * 1024 * 1024,
  maxStorageBufferBindingSize: 16 * 1024 * 1024,
  maxComputeInvocationsPerWorkgroup: 64,
  maxComputeWorkgroupSizeX: 64,
  maxComputeWorkgroupsPerDimension: 64,
};

const SOFTWARE_ADAPTER_PATTERN =
  /swiftshader|llvmpipe|softpipe|software rasterizer|microsoft basic render driver|warp/i;

function limitValue(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/**
 * Probe the adapter once per call. `reason` is always human-readable and is
 * suitable for the panel status line once the backend is wired in.
 */
async function probeCapabilityInner(): Promise<GpuCapability> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return {
      available: false,
      reason: 'WebGPU недоступен: navigator.gpu отсутствует',
    };
  }

  let adapter: GpuAdapter | null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (error) {
    return {
      available: false,
      reason: `WebGPU недоступен: запрос адаптера завершился ошибкой (${String(error)})`,
    };
  }
  if (!adapter) {
    return { available: false, reason: 'WebGPU недоступен: адаптер не найден' };
  }

  const info = adapter.info ?? {};
  const adapterInfo: GpuAdapterDetails = {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
  };
  const identity = [info.vendor, info.architecture, info.device, info.description]
    .filter((part): part is string => typeof part === 'string' && part.length > 0)
    .join(' ');
  if (SOFTWARE_ADAPTER_PATTERN.test(identity)) {
    return {
      available: false,
      reason: `WebGPU: программный адаптер (${identity}) — backend отключён`,
      adapter: adapterInfo,
    };
  }

  const raw = adapter.limits ?? {};
  const limits: GpuLimits = {
    maxBufferSize: limitValue(raw.maxBufferSize, 0),
    maxStorageBufferBindingSize: limitValue(
      raw.maxStorageBufferBindingSize,
      0,
    ),
    maxComputeInvocationsPerWorkgroup: limitValue(
      raw.maxComputeInvocationsPerWorkgroup,
      0,
    ),
    maxComputeWorkgroupSizeX: limitValue(raw.maxComputeWorkgroupSizeX, 0),
    maxComputeWorkgroupsPerDimension: limitValue(
      raw.maxComputeWorkgroupsPerDimension,
      0,
    ),
  };
  const missing: string[] = [];
  const keys = Object.keys(REQUIRED_GPU_LIMITS) as Array<keyof GpuLimits>;
  for (const key of keys) {
    if (limits[key] < REQUIRED_GPU_LIMITS[key]) {
      missing.push(`${key}=${limits[key]}<${REQUIRED_GPU_LIMITS[key]}`);
    }
  }
  if (missing.length > 0) {
    return {
      available: false,
      reason: `WebGPU: лимиты адаптера недостаточны (${missing.join(', ')})`,
      adapter: adapterInfo,
      limits,
    };
  }

  return {
    available: true,
    reason: identity || 'WebGPU-адаптер доступен',
    adapter: adapterInfo,
    limits,
  };
}

/** Logged wrapper: one capability line, then adapter and limits when present. */
export async function probeCapability(): Promise<GpuCapability> {
  const capability = await probeCapabilityInner();
  gpuDebugLog(
    `capability available=${capability.available} reason=${capability.reason}`,
  );
  if (capability.adapter) {
    const a = capability.adapter;
    gpuDebugLog(
      `adapter vendor=${a.vendor ?? '-'} architecture=${a.architecture ?? '-'} description=${a.description ?? '-'}`,
    );
  }
  if (capability.limits) {
    const l = capability.limits;
    gpuDebugLog(
      `limits maxStorageBufferBindingSize=${l.maxStorageBufferBindingSize} ` +
        `maxComputeWorkgroupsPerDimension=${l.maxComputeWorkgroupsPerDimension} ` +
        `maxComputeInvocationsPerWorkgroup=${l.maxComputeInvocationsPerWorkgroup} ` +
        `maxBufferSize=${l.maxBufferSize} maxComputeWorkgroupSizeX=${l.maxComputeWorkgroupSizeX}`,
    );
  }
  return capability;
}
