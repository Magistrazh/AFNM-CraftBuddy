/**
 * GPU backend public surface (S1).
 *
 * The debug object is intentionally not attached to `window` here: the caller
 * (a later slice) decides when to expose it as `window.__craftbuddyGpu`.
 * Kernel/records parity follows tools/gpu-probe/stage4.html + stage7.html.
 */

import { probeCapability, type GpuCapability } from './capability';
import { gpuDebugLog } from './debug';
import type { GpuProfile } from './profile';
import { buildKernelSource, GPU_KERNEL_WGSL } from './kernel';
import {
  CAND,
  EFF_OFF,
  FMOD_OFF,
  HEAD,
  MENTRY_OFF,
  NODE,
  NS,
  OUT,
  REC,
  buildGpuRecords,
} from './records';
import {
  disableGpuBackend,
  gpuBackendStatus,
  resetGpuBackend,
  search,
} from './search';
import type { CraftingState } from '../state';
import type { OptimizerConfig, SkillDefinition } from '../skills';
import type {
  GpuRootRank,
  GpuSearchOptions,
  GpuSearchResult,
  GpuTiming,
} from './search';

export { probeCapability, REQUIRED_GPU_LIMITS } from './capability';
export type {
  GpuAdapterDetails,
  GpuCapability,
  GpuLimits,
} from './capability';
export { describeProfile, GpuProfiler } from './profile';
export type { GpuProfile, GpuProfileMark } from './profile';
export { buildGpuRecords } from './records';
export type { GpuRecordOptions, GpuRecordSet } from './records';
export { buildKernelSource, GPU_KERNEL_WGSL } from './kernel';
export type { KernelDimensions } from './kernel';
export {
  DEFAULT_PATH_LIMIT,
  disableGpuBackend,
  gpuBackendStatus,
  reconstructPaths,
  resetGpuBackend,
  search,
} from './search';
export type {
  GpuBeamSummary,
  GpuPath,
  GpuRootRank,
  GpuSearchOptions,
  GpuSearchResult,
  GpuSkillDelta,
  GpuTiming,
} from './search';

export interface GpuDebugApi {
  capability(): Promise<GpuCapability>;
  status(): ReturnType<typeof gpuBackendStatus>;
  records(
    state: CraftingState,
    config: OptimizerConfig,
    skills: SkillDefinition[],
    options?: GpuSearchOptions,
  ): ReturnType<typeof buildGpuRecords>;
  search(
    state: CraftingState,
    config: OptimizerConfig,
    skills: SkillDefinition[],
    options?: GpuSearchOptions,
  ): Promise<GpuSearchResult>;
  disable(reason: string): void;
  reset(): void;
  readonly layout: {
    readonly HEAD: number;
    readonly EFF_OFF: number;
    readonly MENTRY_OFF: number;
    readonly FMOD_OFF: number;
    readonly REC: number;
    readonly NODE: number;
    readonly NS: number;
    readonly OUT: number;
    readonly CAND: number;
  };
  readonly wgsl: string;
  makeWgsl(dim: {
    beamWidth: number;
    skillCount: number;
    topK: number;
  }): string;
}

/** Attach later with `window.__craftbuddyGpu = craftbuddyGpuDebug`. */
export const craftbuddyGpuDebug: GpuDebugApi = {
  capability: probeCapability,
  status: gpuBackendStatus,
  records: buildGpuRecords,
  search,
  disable: disableGpuBackend,
  reset: resetGpuBackend,
  layout: { HEAD, EFF_OFF, MENTRY_OFF, FMOD_OFF, REC, NODE, NS, OUT, CAND },
  wgsl: GPU_KERNEL_WGSL,
  makeWgsl: buildKernelSource,
};

const ROOT_DEFAULT_DEPTH = 4;
/** Hard ceiling: ranking depth is a heuristic, never the caller's search depth. */
const ROOT_MAX_DEPTH = 8;
const ROOT_DEFAULT_BEAM_WIDTH = 1000;
const ROOT_DEFAULT_TIMEOUT_MS = 800;
const ROOT_LOG_LIMIT = 5;

export interface GpuRootRanking {
  readonly available: boolean;
  readonly reason?: string;
  readonly roots: GpuRootRank[];
  readonly timing?: GpuTiming;
  readonly profile?: GpuProfile;
}

/**
 * Rank opening skills from one GPU beam search. The JS engine can use the
 * result as a candidate filter; the score is the beam heuristic, not its own
 * scoreState. Never throws: every failure is an `available: false` reason.
 */
export async function rankRoots(
  state: CraftingState,
  config: OptimizerConfig,
  condition: string | undefined,
  options: {
    depth?: number;
    beamWidth?: number;
    timeoutMs?: number;
  } = {},
): Promise<GpuRootRanking> {
  try {
    const capability = await withParityTimeout(probeCapability());
    if (!capability.available) {
      return { available: false, roots: [], reason: capability.reason };
    }
    // Ranking is a heuristic, not a search: cap the depth so a caller's deep
    // lookahead setting (96 is legal) cannot turn this into a minute-long run.
    const depth = Math.min(
      ROOT_MAX_DEPTH,
      Math.max(1, Math.floor(options.depth ?? ROOT_DEFAULT_DEPTH)),
    );
    const beamWidth = Math.max(1, Math.floor(options.beamWidth ?? ROOT_DEFAULT_BEAM_WIDTH));
    const timeoutMs = Math.max(1, Math.floor(options.timeoutMs ?? ROOT_DEFAULT_TIMEOUT_MS));
    const gpu = await search(state, config, config.skills, {
      currentCondition: condition,
      depth,
      beamWidth,
      timeoutMs,
      // A slow ranking must not disable the backend until a restart.
      disableOnTimeout: false,
    });
    const noteKeys = Object.keys(gpu.notes ?? {});
    if (noteKeys.length > 0) {
      const summary = noteKeys
        .slice(0, 3)
        .map((key) => `${key}=${gpu.notes[key]}`)
        .join('; ');
      return { available: false, roots: [], reason: `неполные записи: ${summary}` };
    }
    const roots = gpu.beam?.roots ?? [];
    if (roots.length === 0) {
      return { available: false, roots: [], reason: 'beam не вернул ни одного корня' };
    }
    gpuDebugLog(
      `roots count=${roots.length} top=` +
        roots
          .slice(0, ROOT_LOG_LIMIT)
          .map((root) => `${root.name}:${root.score}`)
          .join(','),
    );
    return { available: true, roots, timing: gpu.timing, profile: gpu.profile };
  } catch (error) {
    return { available: false, roots: [], reason: parityReason(error) };
  }
}

function parityReason(error: unknown): string {
  return error instanceof Error
    ? `${error.name}: ${error.message}`
    : String(error);
}

const RANK_HARD_TIMEOUT_MS = 3000;

/** Mirrors the hard-timeout pattern in search.ts without a new dependency. */
function withParityTimeout<T>(work: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`WebGPU-таймаут ${RANK_HARD_TIMEOUT_MS} мс`));
    }, RANK_HARD_TIMEOUT_MS);
  });
  return Promise.race([work, timeout]).finally(() => {
    if (timer !== undefined) clearTimeout(timer);
  });
}
