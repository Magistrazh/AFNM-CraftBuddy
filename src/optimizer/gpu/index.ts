/**
 * GPU backend public surface (S1).
 *
 * The debug object is intentionally not attached to `window` here: the caller
 * (a later slice) decides when to expose it as `window.__craftbuddyGpu`.
 * Kernel/records parity follows tools/gpu-probe/stage4.html + stage7.html.
 */

import {
  probeCapability,
  type GpuAdapterDetails,
  type GpuCapability,
  type GpuLimits,
} from './capability';
import { gpuDebugLog } from './debug';
import { describeProfile, type GpuProfile } from './profile';
import {
  calculateEffectiveActionCosts,
  calculateSkillGains,
  getConditionEffectsForConfig,
} from '../skills';
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
const ROOT_PARITY_LIMIT = 8;

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

/**
 * Parity report carried by the replay export (S2). The JS engine is the
 * reference: the GPU deltas are compared against its own numbers for the
 * exact state/config/condition that the exported snapshot describes.
 *
 * dMaxStability is recorded from the GPU side only: neither
 * `calculateEffectiveActionCosts` nor `calculateSkillGains` exposes a JS
 * expected max-stability delta, so inventing one here would be a false check.
 */
export interface GpuParityDeltas {
  readonly dQi: number;
  readonly dStability: number;
  readonly dPerfection: number;
  readonly dCompletion: number;
}

export interface GpuParityGpuDeltas extends GpuParityDeltas {
  readonly dMaxStability: number;
}

export interface GpuParitySample {
  readonly skill: string;
  readonly gpu: GpuParityGpuDeltas;
  readonly js: GpuParityDeltas;
}

export interface GpuParityMismatch {
  readonly skill: string;
  readonly field: keyof GpuParityDeltas;
  readonly gpu: number;
  readonly js: number;
  readonly delta: number;
}

interface GpuParitySuccess {
  readonly available: true;
  /** Vendor / architecture / description exactly as reported by the adapter. */
  readonly adapter: GpuAdapterDetails;
  /** Adapter limits that were actually exposed (0 where not reported). */
  readonly limits?: GpuLimits;
  readonly timing: GpuTiming;
  readonly profile: GpuProfile;
  readonly compared: number;
  readonly mismatches: GpuParityMismatch[];
  readonly samples: GpuParitySample[];
  /** Top root-ranking entries from one beam search, when available. */
  readonly roots?: GpuRootRank[];
  /** Device loss seen on this export; one re-create was attempted. */
  readonly deviceLoss?: string;
}

export type GpuParityReport =
  | {
      readonly available: false;
      readonly reason: string;
      /** Present when the device was reached but the records were incomplete. */
      readonly adapter?: Record<string, string>;
      readonly profile?: GpuProfile;
      readonly notes?: Record<string, number>;
    }
  | GpuParitySuccess;

/** Records lacked data; the device itself worked, so keep its diagnostics. */
class ParityIncompleteError extends Error {
  constructor(
    message: string,
    readonly adapter: GpuAdapterDetails,
    readonly profile: GpuProfile,
    readonly notes: Record<string, number>,
  ) {
    super(message);
    this.name = 'ParityIncompleteError';
    // ES5 target: subclassing a built-in breaks the prototype chain, so
    // `instanceof` would be false and the adapter/profile would be dropped.
    Object.setPrototypeOf(this, ParityIncompleteError.prototype);
  }
}

const PARITY_SEARCH_TIMEOUT_MS = 2000;
/** Slightly longer than the search timeout so its own safety-net can fire first. */
const PARITY_HARD_TIMEOUT_MS = 3000;
const PARITY_MAX_SAMPLES = 5;
const PARITY_ABS_TOLERANCE = 0.02;
const PARITY_REL_TOLERANCE = 1e-5;
const PARITY_FIELDS: readonly (keyof GpuParityDeltas)[] = [
  'dQi',
  'dStability',
  'dPerfection',
  'dCompletion',
];

function parityReason(error: unknown): string {
  return error instanceof Error
    ? `${error.name}: ${error.message}`
    : String(error);
}

/** Mirrors the hard-timeout pattern in search.ts without a new dependency. */
function withParityTimeout<T>(work: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`WebGPU-таймаут ${PARITY_HARD_TIMEOUT_MS} мс`));
    }, PARITY_HARD_TIMEOUT_MS);
  });
  return Promise.race([work, timeout]).finally(() => {
    if (timer !== undefined) clearTimeout(timer);
  });
}

async function runParity(
  state: CraftingState,
  config: OptimizerConfig,
  condition: string | undefined,
  capability: GpuCapability,
): Promise<GpuParitySuccess> {
  const gpu = await withParityTimeout(
    search(state, config, config.skills, {
      currentCondition: condition,
      timeoutMs: PARITY_SEARCH_TIMEOUT_MS,
    }),
  );
  // Notes mean the kernel lacked data (record overflow or an unknown effect
  // kind). Plausible-looking deltas are worse than an honest refusal, but the
  // device clearly worked - keep that fact in the report.
  const noteKeys = Object.keys(gpu.notes ?? {});
  if (noteKeys.length > 0) {
    const summary = noteKeys
      .slice(0, 3)
      .map((key) => `${key}=${gpu.notes[key]}`)
      .join('; ');
    throw new ParityIncompleteError(
      `неполные записи: ${summary}`,
      capability.adapter ?? {},
      gpu.profile,
      gpu.notes,
    );
  }
  const conditionEffects = getConditionEffectsForConfig(config, condition);
  const samplesBySkill = new Map<string, GpuParitySample>();
  const mismatches: GpuParityMismatch[] = [];
  let compared = 0;

  for (const delta of gpu.deltas) {
    const skill = config.skills.find((entry) => entry.name === delta.name);
    if (!skill) continue;
    const costs = calculateEffectiveActionCosts(
      state,
      skill,
      config.minStability,
      conditionEffects,
      config,
    );
    const gains = calculateSkillGains(
      state,
      skill,
      config,
      conditionEffects,
      { includeExpectedValue: true, currentCondition: condition },
    );
    // records.ts adds qiRestore only when the technique has no pool effect.
    const hasPoolEffect = (skill.effects ?? []).some(
      (effect) => effect.kind === 'pool',
    );
    const restore =
      !hasPoolEffect && skill.restoresQi ? (skill.qiRestore ?? 0) : 0;
    const js: GpuParityDeltas = {
      dQi:
        Math.max(
          0,
          Math.min(config.maxQi, state.qi - costs.qiCost + restore),
        ) - state.qi,
      dStability: gains.stability - costs.stabilityCost,
      dPerfection: gains.perfection,
      dCompletion: gains.completion,
    };
    const gpuDeltas: GpuParityDeltas = {
      dQi: delta.dQi,
      dStability: delta.dStability,
      dPerfection: delta.dPerfection,
      dCompletion: delta.dCompletion,
    };
    compared++;
    samplesBySkill.set(delta.name, {
      skill: delta.name,
      gpu: { ...gpuDeltas, dMaxStability: delta.dMaxStability },
      js,
    });
    for (const field of PARITY_FIELDS) {
      const gpuValue = gpuDeltas[field];
      const jsValue = js[field];
      const tolerance = Math.max(
        PARITY_ABS_TOLERANCE,
        Math.abs(gpuValue) * PARITY_REL_TOLERANCE,
        Math.abs(jsValue) * PARITY_REL_TOLERANCE,
      );
      if (
        !Number.isFinite(gpuValue) ||
        !Number.isFinite(jsValue) ||
        Math.abs(gpuValue - jsValue) > tolerance
      ) {
        mismatches.push({
          skill: delta.name,
          field,
          gpu: gpuValue,
          js: jsValue,
          delta: gpuValue - jsValue,
        });
      }
    }
  }

  const samples: GpuParitySample[] = [];
  const seen = new Set<string>();
  for (const name of [
    ...mismatches.map((mismatch) => mismatch.skill),
    ...Array.from(samplesBySkill.keys()),
  ]) {
    if (seen.has(name)) continue;
    seen.add(name);
    const sample = samplesBySkill.get(name);
    if (!sample) continue;
    samples.push(sample);
    if (samples.length >= PARITY_MAX_SAMPLES) break;
  }

  return {
    available: true,
    adapter: capability.adapter ?? {},
    ...(capability.limits ? { limits: capability.limits } : {}),
    timing: gpu.timing,
    profile: gpu.profile,
    compared,
    mismatches,
    samples,
  };
}

/**
 * Build the report for one replay export. Never throws: WebGPU absence,
 * timeout, shader/device errors and device loss all become `available: false`.
 * The device itself stays cached in search.ts between exports; on loss it is
 * re-created once before the failure is reported.
 */
export async function buildParityReport(
  state: CraftingState,
  config: OptimizerConfig,
  condition: string | undefined,
): Promise<GpuParityReport> {
  try {
    const capability = await withParityTimeout(probeCapability());
    if (!capability.available) {
      gpuDebugLog(`parity summary available=false reason=${capability.reason}`);
      return { available: false, reason: capability.reason };
    }
    const lossBefore = gpuBackendStatus().lastDeviceLoss;
    let report: GpuParitySuccess;
    try {
      report = await runParity(state, config, condition, capability);
    } catch (error) {
      gpuDebugLog(`error phase=parity message=${parityReason(error)}`);
      if (error instanceof ParityIncompleteError) {
        return {
          available: false,
          reason: error.message,
          adapter: error.adapter as unknown as Record<string, string>,
          profile: error.profile,
          notes: error.notes,
        };
      }
      const loss = gpuBackendStatus().lastDeviceLoss ?? lossBefore;
      if (!loss) {
        return { available: false, reason: parityReason(error) };
      }
      gpuDebugLog(`device-loss detected=${loss}; retrying parity once`);
      resetGpuBackend();
      try {
        report = await runParity(state, config, condition, capability);
      } catch (retryError) {
        gpuDebugLog(`error phase=parity-retry message=${parityReason(retryError)}`);
        return {
          available: false,
          reason: `WebGPU-устройство потеряно (${loss}); повторная инициализация не удалась: ${parityReason(retryError)}`,
        };
      }
      report = { ...report, deviceLoss: loss };
    }
    if (lossBefore && report.deviceLoss === undefined) {
      report = { ...report, deviceLoss: lossBefore };
    }
    // One extra beam search feeds the exported root ranking. Its own refusal
    // rules apply, so a missing ranking never invalidates the parity result.
    const ranking = await rankRoots(state, config, condition);
    if (ranking.available) {
      report = { ...report, roots: ranking.roots.slice(0, ROOT_PARITY_LIMIT) };
    }
    gpuDebugLog(
      `parity summary available=true compared=${report.compared} ` +
        `mismatches=${report.mismatches.length} samples=${report.samples.length} ` +
        `roots=${report.roots?.length ?? 0} ` +
        `profile=${describeProfile(report.profile)}`,
    );
    // Per-skill details stay bounded to PARITY_MAX_SAMPLES lines.
    for (const sample of report.samples.slice(0, PARITY_MAX_SAMPLES)) {
      gpuDebugLog(
        `parity sample skill=${sample.skill} dQi gpu=${sample.gpu.dQi} js=${sample.js.dQi} ` +
          `dStability gpu=${sample.gpu.dStability} js=${sample.js.dStability}`,
      );
    }
    return report;
  } catch (error) {
    gpuDebugLog(`error phase=parity message=${parityReason(error)}`);
    return { available: false, reason: parityReason(error) };
  }
}
