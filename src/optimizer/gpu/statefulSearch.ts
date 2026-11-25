/**
 * Self-contained stateful GPU search (S7a).
 *
 * Owns the complete depth-2 pipeline previously assembled by
 * scripts/depth2-gpu.ts + scripts/depth2-harness.ts:
 *   catalog + state skill records + state nodes, condition transitions,
 *   projection/verdict, survival blend, availability, score and ranking.
 *
 * The WGSL patches the stand used to apply with `.replace()` at call time are
 * applied ONCE here, at module load. Nothing imports `scripts/`, the compact
 * kernel (kernel.ts) and skills.ts stay untouched. On every failure the module
 * returns a loud refusal (see `StatefulSearchResult.refusal`) instead of
 * throwing into the UI.
 */

import { probeCapability, type GpuCapability } from './capability';
import { gpuDebugLog } from './debug';
import { GpuProfiler } from './profile';
import {
  CONDITION_KIND,
  HARMONY_REACTION_KEYS,
  NS_STATE,
  buildBuffCatalog,
  buildStateNodes,
  buildStateSkillRecords,
  conditionIdOf,
} from './records';
import { CONDITION_KERNEL_WGSL } from './kernelConditions';
import { STATE_KERNEL_WGSL } from './kernelState';
import {
  SCORE_INPUT,
  SCORE_PARAM,
  TERMINALITY_WORDS,
  buildScoreParams,
  createScoreKernelPipeline,
} from './kernelScore';
import {
  AVAIL_OUT,
  AVAIL_PARAM,
  AVAIL_SKILL,
  AVAIL_SKILL_FLAGS,
  conditionKindCode,
  conditionNameOfKind,
  createAvailabilityKernelPipeline,
} from './availability';
import {
  VERDICT_IN,
  VERDICT_OUT,
  createVerdictPipeline,
  runVerdictBatch,
} from './verdict';
import { STYLE, createScoreInputBindGroup, createScoreInputPipeline } from './scoreInputKernel';
import { getHarmonyStatModifiers } from '../harmony';
import { willAutoFinish, type OutcomeBands } from '../outcome';
import type { CraftingState } from '../state';
import type { OptimizerConfig, SkillDefinition } from '../skills';
import {
  applySkill,
  calculateActionSurvivabilityFloor,
  calculateEffectiveActionCosts,
  calculateSkillGains,
  getBuffDiscordantConditions,
  getConditionEffectsForConfig,
  isTerminalState,
} from '../skills';
import type { ConditionEffect } from '../gameTypes';
import { __testing } from '../search';
import { normalizeForecastConditionQueue } from '../search/conditions';
import { DEFAULT_SEARCH_CONFIG } from '../search/config';
import { calculateImmediateProgressTowardGoals, deriveBandGoals, goalsMet } from '../search/goals';
import { buildScoringContext } from '../search/scoringContext';
import { SCORING } from '../search/scoring';
import { compareTieBreakMetrics, computeScoreTieWindow } from '../search/ranking';
import type { SkillRecommendation } from '../search/types';
import {
  applySurvivabilityFloorToState,
  conditionContinuationOnSurvival,
  hasFatalTechniqueFailure,
  hasGuaranteedSafeStabilizeAction,
  orderRecommendationsBySafety,
  shouldDeprioritizeAgainstGuaranteedSafeStabilize,
  shouldUseSurvivabilityFloorForContinuation,
} from '../search/safety';
import type { SearchConfig } from '../search/types';

// ---------------------------------------------------------------------------
// Final WGSL — the stand's textual patches, built once at module load.
// ---------------------------------------------------------------------------

export const STATE_PARAM_WORDS = 36;
export const P_COND_POOL_RED = 32;
export const P_COND_POOL_PEN = 33;
export const P_COND_STAB_RED = 34;
export const P_COND_STAB_PEN = 35;
export const COND_PARAM_WORDS = 6;
export const PATCH_MARKER = 'let appliedSuccess = derivedSuccess;';
export const BLEND_IN_WORDS = 4;

/** Matches scripts/depth2-gpu.ts BLEND_WGSL byte for byte. */
export const BLEND_WGSL = `
@group(0) @binding(0) var<storage, read> blendIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> blendOut: array<f32>;

@compute @workgroup_size(64)
fn blendKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&blendOut)) { return; }
  let b = i * ${BLEND_IN_WORDS}u;
  let risk = blendIn[b];
  let p = blendIn[b + 1u];
  let vCont = blendIn[b + 2u];
  let vDeath = blendIn[b + 3u];
  blendOut[i] = select(vCont, p * vCont + (1.0 - p) * vDeath, risk > 0.5);
}
`;

function requireNeedle(source: string, needle: string, label: string): void {
  if (!source.includes(needle)) {
    throw new Error(`${label}: WGSL needle not found (${needle.slice(0, 40)})`);
  }
}

/** The depth-2 EV state kernel, final source, compiled once per device. */
function buildFinalStateWgsl(): string {
  const appliedNeedle = 'let appliedSuccess = 1.0;';
  requireNeedle(STATE_KERNEL_WGSL, appliedNeedle, 'STATE_KERNEL_WGSL');
  let wgsl = STATE_KERNEL_WGSL.replace(appliedNeedle, PATCH_MARKER);
  const evNeedle = 'const EV_MODE: bool = false;';
  requireNeedle(wgsl, evNeedle, 'STATE_KERNEL_WGSL EV_MODE');
  wgsl = wgsl.replace(evNeedle, 'const EV_MODE: bool = true;');
  const derivedNeedle =
    'let derivedSuccess = successRoll(\n' +
    '    skills[header + SKILL_SUCCESS],\n' +
    '    inState[base + NODE_SUCCESS_BONUS] + preMasterySuccessBonus +\n' +
    '      params[P_H_SUCCESS] + params[P_COND_CHANCE]);';
  requireNeedle(wgsl, derivedNeedle, 'STATE_KERNEL_WGSL derivedSuccess');
  wgsl = wgsl.replace(
    '@group(0) @binding(5) var<storage, read_write> trace: array<f32>;',
    '@group(0) @binding(5) var<storage, read_write> trace: array<f32>;\n' +
      '@group(0) @binding(6) var<storage, read> nodeParams: array<f32>;\n' +
      `fn np(nodeBase: u32, p: u32) -> f32 { return nodeParams[(nodeBase / ${NS_STATE.WORDS}u) * ${STATE_PARAM_WORDS}u + p]; }\n`,
  );
  wgsl = wgsl.replace('if (params[P_TRACE] > 0.5', 'if (traceEnabled > 0.5');
  wgsl = wgsl.replace(
    'const TRACE_STRIDE: u32 = 7u;',
    'const TRACE_STRIDE: u32 = 7u;\nvar<private> traceEnabled: f32 = 0.0;',
  );
  wgsl = wgsl.replace(/params\[([^\]]*)\]/g, 'np(nodeBase, $1)');

  const harmonyCostNeedle =
    '  if (np(nodeBase, P_H_QI_RED) != 0.0) {\n' +
    '    f.qiReduction = combineCostStat(true, f.qiReduction, np(nodeBase, P_H_QI_RED));\n' +
    '  }\n' +
    '  if (np(nodeBase, P_H_STAB_RED) != 0.0) {\n' +
    '    f.stabReduction = combineCostStat(true, f.stabReduction, np(nodeBase, P_H_STAB_RED));\n' +
    '  }';
  if (wgsl.split(harmonyCostNeedle).length - 1 !== 2) {
    throw new Error('STATE_KERNEL_WGSL: harmony cost block not found twice');
  }
  const conditionCostNeedle =
    harmonyCostNeedle + '\n' +
    `  if (np(nodeBase, ${P_COND_POOL_RED}u) != 0.0) {\n` +
    `    f.qiReduction = combineCostStat(true, f.qiReduction, np(nodeBase, ${P_COND_POOL_RED}u));\n` +
    '  }\n' +
    `  if (np(nodeBase, ${P_COND_POOL_PEN}u) != 0.0) {\n` +
    `    f.qiPenalty = combineCostStat(false, f.qiPenalty, np(nodeBase, ${P_COND_POOL_PEN}u));\n` +
    '  }\n' +
    `  if (np(nodeBase, ${P_COND_STAB_RED}u) != 0.0) {\n` +
    `    f.stabReduction = combineCostStat(true, f.stabReduction, np(nodeBase, ${P_COND_STAB_RED}u));\n` +
    '  }\n' +
    `  if (np(nodeBase, ${P_COND_STAB_PEN}u) != 0.0) {\n` +
    `    f.stabPenalty = combineCostStat(false, f.stabPenalty, np(nodeBase, ${P_COND_STAB_PEN}u));\n` +
    '  }';
  wgsl = wgsl.split(harmonyCostNeedle).join(conditionCostNeedle);
  requireNeedle(wgsl, 'let base = node * NS_STATE_WORDS;', 'STATE_KERNEL_WGSL entry');
  wgsl = wgsl.replace(
    'let base = node * NS_STATE_WORDS;',
    'let base = node * NS_STATE_WORDS;\n  let nodeBase = base;\n  traceEnabled = params[P_TRACE];',
  );
  return wgsl;
}

/** The depth-2 condition kernel, final source, compiled once per device. */
function buildFinalConditionWgsl(): string {
  const binding = '@group(0) @binding(2) var<storage, read> params: array<f32>;';
  requireNeedle(CONDITION_KERNEL_WGSL, binding, 'CONDITION_KERNEL_WGSL');
  let wgsl = CONDITION_KERNEL_WGSL.replace(
    binding,
    '@group(0) @binding(2) var<storage, read> nodeParams: array<f32>;',
  );
  wgsl = wgsl.replace(
    /params\[/g,
    `nodeParams[node * ${COND_PARAM_WORDS}u + `,
  );
  return wgsl;
}

const FINAL_STATE_WGSL = buildFinalStateWgsl();
const FINAL_CONDITION_WGSL = buildFinalConditionWgsl();

/** Exposed for the stand's patch-applied proof; no call-time patching. */
export function finalKernelSources(): {
  readonly state: string;
  readonly condition: string;
  readonly blend: string;
  readonly patchApplied: boolean;
} {
  return {
    state: FINAL_STATE_WGSL,
    condition: FINAL_CONDITION_WGSL,
    blend: BLEND_WGSL,
    patchApplied:
      FINAL_STATE_WGSL.includes(PATCH_MARKER) &&
      FINAL_STATE_WGSL.includes('const EV_MODE: bool = true;'),
  };
}

// ---------------------------------------------------------------------------
// Device / cache / timeout / lost handling — the search.ts pattern, isolated.
// ---------------------------------------------------------------------------

interface Pipelines {
  readonly state: GpuComputePipeline;
  readonly condition: GpuComputePipeline;
  readonly score: GpuComputePipeline;
  readonly scoreTerminality: GpuComputePipeline;
  readonly availability: GpuComputePipeline;
  readonly verdict: GpuComputePipeline;
  readonly blend: GpuComputePipeline;
  readonly scoreInput: GpuComputePipeline;
}

interface DeviceContext {
  readonly device: GpuDevice;
  readonly capability: GpuCapability;
  readonly pipelines: Pipelines;
}

let context: DeviceContext | null = null;
let pending: Promise<DeviceContext> | null = null;
let disabledReason: string | null = null;
let lastDeviceLoss: string | null = null;

const now = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();

/** Mirrors search.ts: disable on protocol errors and device loss. */
export function disableStatefulGpuBackend(reason: string): void {
  disabledReason = reason;
  context = null;
  pending = null;
}

export function resetStatefulGpuBackend(): void {
  disabledReason = null;
  lastDeviceLoss = null;
  context = null;
  pending = null;
}

export function statefulGpuBackendStatus(): {
  active: boolean;
  disabledReason: string | null;
  lastDeviceLoss: string | null;
} {
  return { active: context !== null, disabledReason, lastDeviceLoss };
}

async function compilePipeline(
  device: GpuDevice,
  wgsl: string,
  entryPoint: string,
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: wgsl });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `${entryPoint} WGSL: ${errors.map((e) => `${e.lineNum}:${e.message}`).join(' | ')}`,
    );
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
  });
}

function objEventMessage(error: unknown): string {
  if (!error || typeof error !== 'object') return String(error);
  const holder = error as { error?: { message?: unknown }; message?: unknown };
  return String(holder.error?.message ?? holder.message ?? error);
}

async function createContext(): Promise<DeviceContext> {
  const capability = await probeCapability();
  if (!capability.available) throw new Error(capability.reason);
  if (!navigator.gpu) throw new Error('WebGPU недоступен: navigator.gpu исчез');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU недоступен: адаптер не найден');
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (event: unknown) => {
    const raw = objEventMessage(event);
    gpuDebugLog(`stateful error phase=device message=${raw}`);
    disableStatefulGpuBackend(`WebGPU-ошибка: ${raw}`);
  });
  const fresh: DeviceContext = {
    device,
    capability,
    pipelines: {
      state: await compilePipeline(device, FINAL_STATE_WGSL, 'applyStateKernel'),
      condition: await compilePipeline(device, FINAL_CONDITION_WGSL, 'conditionTransitions'),
      score: await createScoreKernelPipeline(device),
      scoreTerminality: await createScoreKernelPipeline(device, 'scoreWithTerminalityKernel'),
      availability: await createAvailabilityKernelPipeline(device),
      verdict: await createVerdictPipeline(device),
      blend: await compilePipeline(device, BLEND_WGSL, 'blendKernel'),
      scoreInput: await createScoreInputPipeline(device),
    },
  };
  device.lost.then((info) => {
    if (context === fresh) {
      context = null;
      lastDeviceLoss = `${info.reason}: ${info.message}`;
      gpuDebugLog(`stateful device-loss reason=${info.reason} message=${info.message}`);
    }
  });
  context = fresh;
  return fresh;
}

async function getContext(): Promise<DeviceContext> {
  if (disabledReason !== null) throw new Error(disabledReason);
  if (context !== null) return context;
  if (pending !== null) return pending;
  pending = createContext().then(
    (created) => {
      pending = null;
      return created;
    },
    (error: unknown) => {
      pending = null;
      throw error;
    },
  );
  return pending;
}

// ---------------------------------------------------------------------------
// Result / options API.
// ---------------------------------------------------------------------------

export interface StatefulSearchResult {
  /** null on refusal. Skill `key`, as the engine's recommendation keys use. */
  readonly chosen: string | null;
  readonly score: number;
  /** Replayable opening moves (root first, best depth-2 continuation second). */
  readonly path: string[];
  /** Full order under the engine's safety/tie/diversity ranking. */
  readonly order: string[];
  readonly scores: Readonly<Record<string, number>>;
  readonly timing: {
    readonly encodeMs: number;
    readonly gpuMs: number;
    readonly readbackMs: number;
    readonly totalMs: number;
  };
  /** Device-side counters and parity evidence for the stand. */
  readonly notes: Record<string, number>;
  /** Human-readable reason when the search refused; absent on success. */
  readonly refusal?: string;
}

export interface StatefulSearchOptions {
  /** Forecast condition; defaults to 'neutral'. */
  readonly currentCondition?: string;
  /** First three condition-queue words; defaults to the normalized forecast. */
  readonly conditionQueue?: readonly string[];
  /** Overrides config.targetCompletion / config.targetPerfection. */
  readonly targetCompletion?: number;
  readonly targetPerfection?: number;
  /**
   * Engine lookahead depth, passed unchanged from the mod settings.
   * Omitted only by depth-2 legacy callers (stand/tests): defaults to 2.
   * Depths above 2 are refused until the device-resident beam is wired in —
   * never silently answered with a depth-2 plan.
   */
  readonly lookaheadDepth?: number;
  /** Engine search budget, passed unchanged from `searchConfig.timeBudgetMs`. */
  readonly timeBudgetMs?: number;
  /** Engine node ceiling, passed unchanged from `searchConfig.maxNodes`. */
  readonly maxNodes?: number;
  /** Engine branch/beam width, passed unchanged from `searchConfig.beamWidth`. */
  readonly beamWidth?: number;
  /** Whole engine search config, passed unchanged; no parallel GPU constants. */
  readonly searchConfig?: Partial<SearchConfig>;
  /**
   * Hard watchdog. When `timeBudgetMs` is present it is the search deadline and
   * this is only the legacy fallback (cold-compilation ceiling in the mod).
   */
  readonly timeoutMs?: number;
  /** Timeout disables the backend for the session unless false. */
  readonly disableOnTimeout?: boolean;
}

// ---------------------------------------------------------------------------
// Host preparation — mirrors scripts/depth2-gpu.ts builders.
// ---------------------------------------------------------------------------

function normalizeSkill(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, '_');
}

/** conditionEffectsData is keyed by condition; keep one narrow cast site. */
function effectsOf(config: OptimizerConfig, condition: string): ConditionEffect[] {
  const table = config.conditionEffectsData as unknown as
    | Record<string, ConditionEffect[]>
    | undefined;
  return table?.[condition] ?? [];
}

/** ES5 target: no spread over Float32Array. */
function pushFloats(into: number[], values: Float32Array): void {
  for (let i = 0; i < values.length; i++) into.push(values[i]);
}

const HARMONY_OPS: Readonly<Record<string, number>> = {
  forge: 1,
  alchemical: 2,
  resonance: 3,
  eccentricDecree: 4,
  inscription: 5,
  captivatingCadence: 6,
};

function harmonyParamsFromState(state: CraftingState, config: OptimizerConfig): Record<string, number> {
  const type = config.craftingType ?? '';
  const hd = (state.harmonyData ?? {}) as Record<string, unknown>;
  const out: Record<string, number> = {
    type: HARMONY_OPS[type] ?? 0,
    controlMult: 1,
    intensityMult: 1,
    critChance: 0,
    successChance: 0,
    qiReduction: 0,
    stabilityReduction: 0,
    derivedMask: 0,
  };
  if (type === 'forge') {
    const fw = (hd.forgeWorks ?? {}) as Record<string, unknown>;
    const clamp = (value: number): number => Math.max(0, Math.min(10, Math.floor(value)));
    let heat = clamp(Number(fw.heat ?? 0));
    if (heat === 1) heat = clamp(Number(fw.lastBuffedHeat ?? heat));
    if (heat >= 4 && heat <= 6) {
      out.controlMult = 1.5;
      out.intensityMult = 1.5;
    } else if (heat >= 2 && heat <= 3) out.controlMult = 0.5;
    else if (heat >= 7 && heat <= 9) out.intensityMult = 0.5;
    else if (heat === 0) out.controlMult = -9;
    else if (heat === 10) out.intensityMult = -9;
    out.derivedMask = hd.forgeWorks ? 1 : 0;
    return out;
  }
  if (type === 'alchemical') {
    const extra = (hd.additionalData ?? {}) as Record<string, unknown>;
    const mods = (extra.alchemicalReactionModifiers ?? {}) as Record<string, number>;
    const poolPct = mods.poolCostPercentage;
    const stabPct = mods.stabilityCostPercentage;
    out.controlMult = mods.controlMultiplier ?? 1;
    out.intensityMult = mods.intensityMultiplier ?? 1;
    out.critChance = mods.critChanceBonus ?? 0;
    out.successChance = mods.successChanceBonus ?? 0;
    out.qiReduction = mods.qiCostReduction ?? (typeof poolPct === 'number' ? 100 - poolPct : 0);
    out.stabilityReduction =
      mods.stabilityCostReduction ?? (typeof stabPct === 'number' ? 100 - stabPct : 0);
    out.derivedMask = extra.alchemicalReactionModifiers !== undefined ? 8 : 0;
    return out;
  }
  if (type === 'resonance') {
    const res = (hd.resonance ?? {}) as Record<string, unknown>;
    out.critChance = Number(res.strength ?? 0) * 3;
    out.successChance = Number(res.strength ?? 0) * 0.03;
    out.derivedMask = hd.resonance ? 2 : 0;
    return out;
  }
  if (type === 'eccentricDecree') {
    const decree = (hd.eccentricDecree ?? {}) as Record<string, unknown>;
    const focused = decree.focusedBar ?? 'completion';
    out.controlMult = focused === 'perfection' ? 1.5 : 1;
    out.intensityMult = focused === 'completion' ? 1.5 : 1;
    out.derivedMask = hd.eccentricDecree ? 4 : 0;
    return out;
  }
  if (type === 'inscription') {
    const inscribed = (hd.inscribedPatterns ?? {}) as Record<string, unknown>;
    const bonus = Number(inscribed.stacks ?? 0) * 0.02;
    out.controlMult = 1 + bonus;
    out.intensityMult = 1 + bonus;
    return out;
  }
  if (type === 'captivatingCadence') {
    const cadence = (hd.captivatingCadence ?? {}) as Record<string, unknown>;
    const bonus = Number(cadence.chain ?? 0) * 0.02;
    out.controlMult = 1 + bonus;
    out.intensityMult = 1 + bonus;
    return out;
  }
  return out;
}

function conditionChanceBonus(config: OptimizerConfig, condition: string): number {
  const effects = effectsOf(config, condition) as unknown as Array<{
    kind?: string;
    bonus?: number;
  }>;
  let bonus = 0;
  for (const effect of effects) {
    if (effect?.kind === 'chance' && effect.bonus !== undefined) {
      bonus += Number(effect.bonus);
    }
  }
  return bonus;
}

function conditionDelta(config: OptimizerConfig, condition: string): { control: number; intensity: number } {
  let control = 0;
  let intensity = 0;
  const effects = getConditionEffectsForConfig(config, condition) as Array<{
    kind?: string;
    multiplier?: number;
  }>;
  for (const effect of effects) {
    if (effect?.kind === 'control') control += Number(effect.multiplier ?? 0);
    if (effect?.kind === 'intensity') intensity += Number(effect.multiplier ?? 0);
  }
  return { control, intensity };
}

interface Records {
  readonly catalog: ReturnType<typeof buildBuffCatalog>;
  readonly skillSet: ReturnType<typeof buildStateSkillRecords>;
  readonly rootNode: Float32Array;
  readonly paramsBase: Float32Array;
}

function buildRecords(state: CraftingState, config: OptimizerConfig, skills: SkillDefinition[]): Records {
  const catalog = buildBuffCatalog(state, config, skills);
  const skillSet = buildStateSkillRecords(state, config, skills, catalog);
  const rootNode = Float32Array.from(buildStateNodes(state, config, skills, 1, catalog).nodes);
  const paramsBase = new Float32Array(STATE_PARAM_WORDS);
  const harmony = harmonyParamsFromState(state, config);
  paramsBase[4] = Number(config.maxQi ?? 0);
  paramsBase[5] = Number(state.initialMaxStability);
  paramsBase[6] = harmony.type;
  paramsBase[7] = harmony.controlMult;
  paramsBase[8] = harmony.intensityMult;
  paramsBase[9] = harmony.critChance;
  paramsBase[10] = harmony.qiReduction;
  paramsBase[11] = harmony.stabilityReduction;
  paramsBase[29] = harmony.successChance;
  paramsBase[30] = conditionChanceBonus(
    config,
    (config as OptimizerConfig & { currentCondition?: string }).currentCondition ?? 'neutral',
  );
  paramsBase[12] = harmony.derivedMask;
  ['heat', 'resonance', 'decree_focus', 'reaction'].forEach((name, i) => {
    paramsBase[13 + i] = catalog.ids.get(name) ?? -1;
  });
  [
    'completion_bonus',
    'inscription',
    'captivating_cadence',
    'captivating_cadence_chain',
  ].forEach((name, i) => {
    paramsBase[17 + i] = catalog.ids.get(name) ?? -1;
  });
  paramsBase[23] = Number(config.targetCompletion ?? 0);
  paramsBase[24] = Number(config.targetPerfection ?? 0);
  paramsBase[25] = Number(config.maxCompletion ?? Number.POSITIVE_INFINITY);
  paramsBase[26] = Number(config.maxPerfection ?? Number.POSITIVE_INFINITY);
  paramsBase[28] = config.isSublimeCraft === true ? 1 : 0;
  return { catalog, skillSet, rootNode, paramsBase };
}

function makeStateParams(
  base: Float32Array,
  skillIndex: number,
  condition: string,
  node: Float32Array,
  nodeOffset: number,
  config: OptimizerConfig,
  forceSuccess = 0,
): Float32Array {
  const params = Float32Array.from(base);
  params[31] = forceSuccess;
  params[0] = skillIndex;
  params[1] = conditionIdOf(condition);
  const effects = getConditionEffectsForConfig(config, condition) as Array<{
    kind?: string;
    multiplier?: number;
  }>;
  let control = 0;
  let intensity = 0;
  let poolReduction = 0;
  let poolPenalty = 0;
  let stabReduction = 0;
  let stabPenalty = 0;
  for (const effect of effects) {
    if (effect?.kind === 'control') control += Number(effect.multiplier ?? 0);
    if (effect?.kind === 'intensity') intensity += Number(effect.multiplier ?? 0);
    const multiplier = Number(effect?.multiplier);
    if (!Number.isFinite(multiplier) || multiplier === 1) continue;
    if (effect?.kind === 'pool') {
      if (multiplier < 1) poolReduction = combine(1, poolReduction, (1 - multiplier) * 100);
      else poolPenalty = combine(false, poolPenalty, (multiplier - 1) * 100);
    } else if (effect?.kind === 'stability') {
      if (multiplier < 1) stabReduction = combine(1, stabReduction, (1 - multiplier) * 100);
      else stabPenalty = combine(false, stabPenalty, (multiplier - 1) * 100);
    }
  }
  params[2] = control;
  params[3] = intensity;
  params[P_COND_POOL_RED] = poolReduction;
  params[P_COND_POOL_PEN] = poolPenalty;
  params[P_COND_STAB_RED] = stabReduction;
  params[P_COND_STAB_PEN] = stabPenalty;
  params[12] = node[nodeOffset + NS_STATE.harmonyDerivedMask];
  const type = config.craftingType ?? '';
  if (type === 'forge') {
    let heat = node[nodeOffset + NS_STATE.harmonyForgeHeat];
    if (heat === 1) {
      const last = node[nodeOffset + NS_STATE.harmonyForgeLastBuffedHeat];
      if (last >= 0) heat = last;
    }
    heat = Math.max(0, Math.min(10, Math.floor(heat)));
    let controlMult = 1;
    let intensityMult = 1;
    if (heat >= 4 && heat <= 6) {
      controlMult = 1.5;
      intensityMult = 1.5;
    } else if (heat >= 2 && heat <= 3) controlMult = 0.5;
    else if (heat >= 7 && heat <= 9) intensityMult = 0.5;
    else if (heat === 0) controlMult = -9;
    else if (heat === 10) intensityMult = -9;
    params[7] = controlMult;
    params[8] = intensityMult;
  } else if (type === 'alchemical') {
    const mods: Record<string, number> = {};
    HARMONY_REACTION_KEYS.forEach((key, index) => {
      mods[key] = node[nodeOffset + NS_STATE.harmonyReactionMods + index];
    });
    const poolPct = mods.poolCostPercentage;
    const stabPct = mods.stabilityCostPercentage;
    params[7] = mods.controlMultiplier ?? 1;
    params[8] = mods.intensityMultiplier ?? 1;
    params[9] = mods.critChanceBonus ?? 0;
    params[10] = mods.qiCostReduction ?? (poolPct !== undefined ? 100 - poolPct : 0);
    params[11] = mods.stabilityCostReduction ?? (stabPct !== undefined ? 100 - stabPct : 0);
    params[29] = mods.successChanceBonus ?? 0;
  } else if (type === 'resonance') {
    params[9] = node[nodeOffset + NS_STATE.harmonyResonanceStrength] * 3;
    params[29] = node[nodeOffset + NS_STATE.harmonyResonanceStrength] * 0.03;
  }
  return params;
}

function combine(reduction: number | boolean, current: number, contribution: number): number {
  return reduction
    ? 100 - ((100 - Math.min(100, current)) * (100 - Math.min(100, contribution))) / 100
    : ((100 + current) * (100 + contribution)) / 100 - 100;
}

function makeCondParams(
  harmony: number,
  discordance: number,
  config: SearchConfig,
  condition: string,
): Float32Array {
  const params = new Float32Array(COND_PARAM_WORDS);
  params[0] = harmony;
  params[1] = discordance;
  params[2] = config.enableConditionBranchingAfterForecast ? 1 : 0;
  params[3] = Number(config.conditionBranchLimit ?? 2);
  params[4] = Number(config.conditionBranchMinProbability ?? 0.15);
  params[5] = conditionIdOf(condition);
  return params;
}

function buffer(device: GpuDevice, data: Float32Array, usage: number): GpuBuffer {
  const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

async function runStateBatch(
  ctx: DeviceContext,
  records: Records,
  nodes: Float32Array,
  params: Float32Array,
): Promise<Float32Array> {
  const device = ctx.device;
  const pipeline = ctx.pipelines.state;
  const count = nodes.length / NS_STATE.WORDS;
  if (params.length !== count * STATE_PARAM_WORDS) throw new Error('state params length mismatch');
  const input = buffer(device, nodes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = device.createBuffer({
    size: Math.max(16, nodes.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const catalog = buffer(device, records.catalog.records, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const skills = buffer(device, records.skillSet.records, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const paramBuf = buffer(device, params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const nodeParamBuf = buffer(device, params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const trace = device.createBuffer({ size: 128 * 7 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const traceReadback = device.createBuffer({ size: 128 * 7 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readback = device.createBuffer({
    size: Math.max(16, nodes.byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: catalog } },
        { binding: 3, resource: { buffer: skills } },
        { binding: 4, resource: { buffer: paramBuf } },
        { binding: 5, resource: { buffer: trace } },
        { binding: 6, resource: { buffer: nodeParamBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, nodes.byteLength);
    encoder.copyBufferToBuffer(trace, 0, traceReadback, 0, 128 * 7 * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    await traceReadback.mapAsync(GPUMapMode.READ);
    mapped = true;
    return new Float32Array(readback.getMappedRange().slice(0, nodes.byteLength));
  } finally {
    if (mapped) {
      readback.unmap();
      traceReadback.unmap();
    }
    readback.destroy();
    traceReadback.destroy();
    input.destroy();
    output.destroy();
    catalog.destroy();
    skills.destroy();
    paramBuf.destroy();
    nodeParamBuf.destroy();
    trace.destroy();
  }
}

const COND_RECORD_WORDS = 1 + 5 * 5;

async function runConditionBatch(
  ctx: DeviceContext,
  nodes: Float32Array,
  params: Float32Array,
): Promise<Float32Array> {
  const device = ctx.device;
  const count = nodes.length / NS_STATE.WORDS;
  if (params.length !== count * COND_PARAM_WORDS) throw new Error('cond params length mismatch');
  const nodeBuf = buffer(device, nodes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const branchBuf = device.createBuffer({
    size: Math.max(16, count * COND_RECORD_WORDS * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const paramBuf = buffer(device, params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const readback = device.createBuffer({
    size: Math.max(16, count * COND_RECORD_WORDS * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const group = device.createBindGroup({
      layout: ctx.pipelines.condition.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuf } },
        { binding: 1, resource: { buffer: branchBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(ctx.pipelines.condition);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(branchBuf, 0, readback, 0, count * COND_RECORD_WORDS * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    return new Float32Array(readback.getMappedRange().slice(0, count * COND_RECORD_WORDS * 4));
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
    nodeBuf.destroy();
    branchBuf.destroy();
    paramBuf.destroy();
  }
}

function decodeBranches(records: Float32Array, index: number): Array<{
  probability: number;
  conditionId: number;
  queue: number[];
}> {
  const base = index * COND_RECORD_WORDS;
  const count = Math.round(records[base]);
  const out: Array<{ probability: number; conditionId: number; queue: number[] }> = [];
  for (let b = 0; b < count; b++) {
    const o = base + 1 + b * 5;
    out.push({
      probability: records[o],
      conditionId: Math.round(records[o + 1]),
      queue: [records[o + 2], records[o + 3], records[o + 4]].map((v) => Math.round(v)),
    });
  }
  return out;
}

async function runScoreBatch(
  ctx: DeviceContext,
  inputs: Float32Array,
  params: Float32Array,
): Promise<{ scores: Float32Array; terminality: Float32Array }> {
  const device = ctx.device;
  const pipeline = ctx.pipelines.scoreTerminality;
  const count = params.length / SCORE_PARAM.WORDS;
  if (inputs.length !== count * SCORE_INPUT.WORDS) throw new Error('score inputs length mismatch');
  const inputBuf = buffer(device, inputs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const paramBuf = buffer(device, params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scoreBuf = device.createBuffer({
    size: Math.max(16, count * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const termBuf = device.createBuffer({
    size: Math.max(16, count * TERMINALITY_WORDS * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const termReadback = device.createBuffer({
    size: Math.max(16, count * TERMINALITY_WORDS * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readback = device.createBuffer({
    size: Math.max(16, count * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  let termMapped = false;
  try {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: scoreBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
        { binding: 3, resource: { buffer: termBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(scoreBuf, 0, readback, 0, count * 4);
    encoder.copyBufferToBuffer(termBuf, 0, termReadback, 0, count * TERMINALITY_WORDS * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    await termReadback.mapAsync(GPUMapMode.READ);
    termMapped = true;
    return {
      scores: new Float32Array(readback.getMappedRange().slice(0, count * 4)),
      terminality: new Float32Array(
        termReadback.getMappedRange().slice(0, count * TERMINALITY_WORDS * 4),
      ),
    };
  } finally {
    if (mapped) readback.unmap();
    if (termMapped) termReadback.unmap();
    readback.destroy();
    termReadback.destroy();
    inputBuf.destroy();
    paramBuf.destroy();
    scoreBuf.destroy();
    termBuf.destroy();
  }
}

/** One batched dispatch for all survival-edge blends (BLEND_WGSL). */
async function runBlendBatch(
  ctx: DeviceContext,
  cases: Float32Array,
): Promise<Float32Array> {
  const count = cases.length / BLEND_IN_WORDS;
  const device = ctx.device;
  const inBuf = buffer(device, cases, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outBuf = device.createBuffer({
    size: Math.max(16, count * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: Math.max(16, count * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const group = device.createBindGroup({
      layout: ctx.pipelines.blend.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(ctx.pipelines.blend);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    mapped = true;
    return new Float32Array(readBuf.getMappedRange().slice(0, count * 4));
  } finally {
    if (mapped) readBuf.unmap();
    inBuf.destroy();
    outBuf.destroy();
    readBuf.destroy();
  }
}

// ---------------------------------------------------------------------------
// Availability + verdict + candidates.
// ---------------------------------------------------------------------------

interface AvailabilityInput {
  readonly node: Float32Array;
  readonly skill: SkillDefinition;
  readonly skillIndex: number;
  readonly state: CraftingState;
  readonly condition: string;
  readonly config: OptimizerConfig;
  readonly catalogIds: ReadonlyMap<string, number>;
}

function buildAvailabilityInputs(candidates: readonly AvailabilityInput[]): {
  nodes: Float32Array;
  skills: Float32Array;
  params: Float32Array;
} {
  const nodes = new Float32Array(candidates.length * NS_STATE.WORDS);
  const skills = new Float32Array(candidates.length * AVAIL_SKILL.WORDS);
  const params = new Float32Array(candidates.length * AVAIL_PARAM.WORDS);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    nodes.set(c.node.subarray(0, NS_STATE.WORDS), i * NS_STATE.WORDS);
    const b = i * AVAIL_SKILL.WORDS;
    const p = i * AVAIL_PARAM.WORDS;
    const skill = c.skill;
    const buffRequirement = skill.buffRequirement;
    const buffCost = skill.buffCost;
    const reqName = buffRequirement ? normalizeIdentifier(buffRequirement.buffName) : '';
    const costName = buffCost ? normalizeIdentifier(buffCost.buffName) : '';
    const reqId = reqName ? (c.catalogIds.get(reqName) ?? -1) : -1;
    const costId = costName ? (c.catalogIds.get(costName) ?? -1) : -1;
    skills[b + AVAIL_SKILL.dropletCost] = Number(skill.dropletCost ?? 0);
    skills[b + AVAIL_SKILL.toxicityCost] = Number(skill.toxicityCost ?? 0);
    skills[b + AVAIL_SKILL.conditionKind] = skill.conditionRequirement
      ? conditionKindCode(skill.conditionRequirement)
      : 0;
    skills[b + AVAIL_SKILL.cooldownSkillIndex] =
      skill.actionKind === 'item' ? -1 : c.skillIndex;
    skills[b + AVAIL_SKILL.buffRequirementId] = reqId;
    skills[b + AVAIL_SKILL.buffRequirementAmount] = Number(buffRequirement?.amount ?? 0);
    skills[b + AVAIL_SKILL.buffCostId] = costId;
    skills[b + AVAIL_SKILL.buffCostRequired] = buffCost
      ? buffCost.consumeAll === true ? 1 : Number(buffCost.amount ?? 0)
      : 0;
    skills[b + AVAIL_SKILL.isItem] = skill.actionKind === 'item' ? 1 : 0;
    skills[b + AVAIL_SKILL.reagentOnlyAtStepZero] = skill.reagentOnlyAtStepZero === true ? 1 : 0;
    let flags = skill.conditionRequirement ? AVAIL_SKILL_FLAGS.hasConditionRequirement : 0;
    skills[b + AVAIL_SKILL.flags] = flags;

    const effects = getConditionEffectsForConfig(c.config, c.condition);
    const effectiveQiCost = calculateEffectiveActionCosts(
      c.state, skill, c.config.minStability, effects, c.config,
    ).qiCost;
    const itemKey = normalizeIdentifier(skill.itemName || skill.key);
    params[p + AVAIL_PARAM.qiDroplets] = Number(c.state.qiDroplets ?? 0);
    params[p + AVAIL_PARAM.itemRemaining] = Number(c.state.items.get(itemKey) ?? 0);
    params[p + AVAIL_PARAM.consumedPillsThisTurn] = Number(c.state.consumedPillsThisTurn ?? 0);
    params[p + AVAIL_PARAM.pillsPerRound] = Math.max(1, Math.floor(c.config.pillsPerRound || 1));
    // No native provider inside the mod: neutral precheck, same as the stand.
    params[p + AVAIL_PARAM.nativePrecheck] = 1;
    params[p + AVAIL_PARAM.nativePrecheckPresent] = 0;
    params[p + AVAIL_PARAM.effectiveQiCost] = effectiveQiCost;
    params[p + AVAIL_PARAM.maxToxicity] = Number(c.config.maxToxicity || 0);
    params[p + AVAIL_PARAM.currentConditionKind] = conditionKindCode(c.condition);
  }
  return { nodes, skills, params };
}

function normalizeIdentifier(value: unknown): string {
  return String(value ?? '').trim().toLowerCase().replace(/\s+/g, '_');
}

async function deviceAvailableSkills(
  ctx: DeviceContext,
  records: Records,
  node: Float32Array,
  state: CraftingState,
  condition: string,
  config: OptimizerConfig,
  skills: readonly SkillDefinition[],
  notes: Record<string, number>,
): Promise<SkillDefinition[]> {
  if (skills.length === 0) return [];
  const candidates = skills.map((skill) => ({
    node,
    skill,
    skillIndex: records.skillSet.names.findIndex(
      (n) => normalizeSkill(n) === normalizeSkill(skill.name),
    ),
    state,
    condition,
    config,
    catalogIds: records.catalog.ids,
  }));
  const inputs = buildAvailabilityInputs(candidates);
  const count = inputs.params.length / AVAIL_PARAM.WORDS;
  const device = ctx.device;
  const nodeBuf = buffer(device, inputs.nodes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const skillBuf = buffer(device, inputs.skills, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const paramBuf = buffer(device, inputs.params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outBuf = device.createBuffer({
    size: Math.max(16, count * AVAIL_OUT.WORDS * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: Math.max(16, count * AVAIL_OUT.WORDS * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  let decisions: boolean[];
  try {
    const group = device.createBindGroup({
      layout: ctx.pipelines.availability.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuf } },
        { binding: 1, resource: { buffer: skillBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(ctx.pipelines.availability);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readback, 0, count * AVAIL_OUT.WORDS * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const words = new Float32Array(readback.getMappedRange().slice(0, count * AVAIL_OUT.WORDS * 4));
    decisions = [];
    for (let i = 0; i < count; i++) {
      decisions.push(words[i * AVAIL_OUT.WORDS + AVAIL_OUT.available] > 0.5);
    }
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
    nodeBuf.destroy();
    skillBuf.destroy();
    paramBuf.destroy();
    outBuf.destroy();
  }
  notes.availabilityNodes = (notes.availabilityNodes ?? 0) + 1;
  notes.availabilityCandidates = (notes.availabilityCandidates ?? 0) + count;
  return skills.filter((_skill, i) => decisions[i]);
}

interface SurvivalContext {
  readonly hasGuaranteedSafeStabilize: boolean;
  readonly minimumGuaranteedContinuationStability: number;
}

interface CandidateInfo {
  readonly skill: SkillDefinition;
  readonly display: CraftingState;
  readonly searchState: CraftingState;
  readonly expectedGains: { completion: number; perfection: number };
  readonly successGains: { completion: number; perfection: number };
  readonly requiresProbabilisticSurvival: boolean;
  readonly unsafe: boolean;
  readonly fatal: boolean;
  readonly displayEndsCraft: boolean;
  readonly endsCraft: boolean;
}

interface SearchEnv {
  readonly config: OptimizerConfig;
  readonly configState: CraftingState;
  readonly searchCfg: SearchConfig;
  readonly ctx: ReturnType<typeof buildScoringContext>;
  readonly targets: { completion: number; perfection: number };
  readonly modeCompGoal: number;
  readonly modePerfGoal: number;
  readonly bands: OutcomeBands;
}

/**
 * Engine-owned gain projections consumed by the verdict kernel.
 *
 * Cites: `src/optimizer/skills.ts:2957-2971` (`calculateSkillGains`),
 * `src/optimizer/search/safety.ts:209-221` (the expected vs forced-success
 * pair used by `hasFatalTechniqueFailure`). Both numbers are *raw deltas*
 * before the bar clamp; the failure word is the separate successChance:0
 * state transition (`statefulSearch.ts:1262`, `verdict.ts:85-100`).
 * S7d: this is the one numeric input that is still host-derived; the verdict
 * WGSL itself is unchanged (`verdict.ts:54-124`).
 */
function prepareVerdictGains(
  env: SearchEnv,
  current: CraftingState,
  skill: SkillDefinition,
  condition: string,
): { expected: { completion: number; perfection: number }; success: { completion: number; perfection: number } } {
  const effects = effectsOf(env.config, condition);
  const expectedProjection = calculateSkillGains(current, skill, env.config, effects, {
    currentCondition: condition,
  });
  const successProjection = calculateSkillGains(current, skill, env.config, effects, {
    successChance: 1,
    currentCondition: condition,
  });
  return {
    expected: { completion: expectedProjection.completion, perfection: expectedProjection.perfection },
    success: { completion: successProjection.completion, perfection: successProjection.perfection },
  };
}

function safeContext(env: SearchEnv, state: CraftingState, skills: SkillDefinition[], condition: string): SurvivalContext {
  const gateThreshold = Math.max(10, Math.ceil(env.ctx.avgStabilityCostPerTurn));
  if (
    state.stability > gateThreshold ||
    (state !== env.configState &&
      env.config.isSublimeCraft === true &&
      goalsMet(state, env.targets.completion, env.targets.perfection))
  ) {
    return { hasGuaranteedSafeStabilize: false, minimumGuaranteedContinuationStability: 0 };
  }
  const effects = effectsOf(env.config, condition) as Parameters<
    typeof hasGuaranteedSafeStabilizeAction
  >[3];
  const has = hasGuaranteedSafeStabilizeAction(state, skills, env.config, effects, condition);
  return {
    hasGuaranteedSafeStabilize: has,
    minimumGuaranteedContinuationStability: has ? Math.max(1, env.ctx.avgStabilityCostPerTurn) : 0,
  };
}

function buildCandidate(
  env: SearchEnv,
  current: CraftingState,
  skill: SkillDefinition,
  display: CraftingState,
  condition: string,
  safe: SurvivalContext,
): CandidateInfo {
  const effects = effectsOf(env.config, condition);
  const floor = calculateActionSurvivabilityFloor(current, skill, env.config, effects, condition, display);
  const floorStability = floor?.stability ?? display.stability;
  const gains = prepareVerdictGains(env, current, skill, condition);
  const fatal = hasFatalTechniqueFailure(
    current, skill, env.config, effects, condition,
    env.targets.completion, env.targets.perfection,
  );
  const requiresProbabilisticSurvival =
    fatal ||
    shouldUseSurvivabilityFloorForContinuation({
      currentStability: current.stability,
      displayStability: display.stability,
      floorStability,
      goalsMetAfterAction: goalsMet(display, env.modeCompGoal, env.modePerfGoal),
      baseSuccessSecuredAfterAction: goalsMet(display, env.targets.completion, env.targets.perfection),
      hasGuaranteedSafeStabilize: safe.hasGuaranteedSafeStabilize,
      minimumGuaranteedContinuationStability: safe.minimumGuaranteedContinuationStability,
    });
  const unsafe =
    fatal ||
    shouldDeprioritizeAgainstGuaranteedSafeStabilize({
      currentStability: current.stability,
      displayStability: display.stability,
      floorStability,
      goalsMetAfterAction: goalsMet(display, env.modeCompGoal, env.modePerfGoal),
      baseSuccessSecuredAfterAction: goalsMet(display, env.targets.completion, env.targets.perfection),
      hasGuaranteedSafeStabilize: safe.hasGuaranteedSafeStabilize,
      minimumGuaranteedContinuationStability: safe.minimumGuaranteedContinuationStability,
    });
  const displayEndsCraft = willAutoFinish(display, env.bands);
  const presentation = requiresProbabilisticSurvival
    ? applySurvivabilityFloorToState(display, floor)
    : display;
  const searchState = conditionContinuationOnSurvival(
    presentation, floor, requiresProbabilisticSurvival, displayEndsCraft,
  );
  return {
    skill,
    display,
    searchState,
    expectedGains: gains.expected,
    successGains: gains.success,
    requiresProbabilisticSurvival,
    unsafe,
    fatal,
    displayEndsCraft,
    endsCraft: willAutoFinish(searchState, env.bands),
  };
}

const uncollapsedRisk = (cand: CandidateInfo, floor: ReturnType<typeof calculateActionSurvivabilityFloor>): boolean =>
  !cand.requiresProbabilisticSurvival &&
  !cand.displayEndsCraft &&
  floor !== null &&
  floor.stability <= 0 &&
  floor.survivalProbability > 0 &&
  floor.survivalProbability < 1;

function toNodeFloat(value: number): number {
  const rounded = Math.round(value);
  if (rounded === value || Math.fround(value) !== Math.fround(rounded)) return value;
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, rounded, false);
  const bits = view.getUint32(0, false) + (value < rounded ? -1 : 1);
  view.setUint32(0, bits, false);
  return view.getFloat32(0, false);
}

function continuationNode(node: Float32Array, cand: CandidateInfo): Float32Array {
  const out = Float32Array.from(node);
  out[NS_STATE.stability] = toNodeFloat(cand.searchState.stability);
  out[NS_STATE.maxStability] = toNodeFloat(cand.searchState.maxStability);
  out[NS_STATE.stabilityPenalty] = toNodeFloat(cand.searchState.stabilityPenalty);
  return out;
}

function barsMet(completion: number, perfection: number, cGoal: number, pGoal: number): boolean {
  const hasC = Number.isFinite(cGoal) && cGoal > 0;
  const hasP = Number.isFinite(pGoal) && pGoal > 0;
  if (!hasC && !hasP) return false;
  return (!hasC || completion >= cGoal) && (!hasP || perfection >= pGoal);
}

/**
 * Host-side verdict input assembly. Every word is either a device node word or
 * an engine-named input; the expected/success gain pair comes from
 * `prepareVerdictGains` above (skills.ts:2957-2971, safety.ts:209-221) and the
 * failure bars come from the forced-failure state dispatch (verdict.ts:85-100).
 */
function prepareVerdictInputs(
  env: SearchEnv,
  inputNodes: Float32Array,
  evNodes: Float32Array,
  failureRun: Float32Array,
  params: Float32Array,
  candidates: readonly CandidateInfo[],
  safes: readonly SurvivalContext[],
): Float32Array {
  const count = candidates.length;
  const inputs = new Float32Array(count * VERDICT_IN.WORDS);
  for (let i = 0; i < count; i++) {
    const off = i * VERDICT_IN.WORDS;
    const ev = i * NS_STATE.WORDS;
    const cand = candidates[i];
    const safe = safes[i];
    inputs[off + VERDICT_IN.currentStability] = inputNodes[ev + NS_STATE.stability];
    inputs[off + VERDICT_IN.displayStability] = evNodes[ev + NS_STATE.stability];
    inputs[off + VERDICT_IN.floorStability] = evNodes[ev + NS_STATE.floorStability];
    inputs[off + VERDICT_IN.floorMaxStability] = evNodes[ev + NS_STATE.floorMaxStability];
    inputs[off + VERDICT_IN.survivalProbability] = evNodes[ev + NS_STATE.survivalProbability];
    const comp = evNodes[ev + NS_STATE.completion];
    const perf = evNodes[ev + NS_STATE.perfection];
    inputs[off + VERDICT_IN.goalsMetAfterAction] = barsMet(comp, perf, env.modeCompGoal, env.modePerfGoal) ? 1 : 0;
    inputs[off + VERDICT_IN.baseSuccessSecuredAfterAction] = barsMet(comp, perf, env.targets.completion, env.targets.perfection) ? 1 : 0;
    inputs[off + VERDICT_IN.hasGuaranteedSafeStabilize] = safe.hasGuaranteedSafeStabilize ? 1 : 0;
    inputs[off + VERDICT_IN.minimumGuaranteedContinuationStability] = safe.minimumGuaranteedContinuationStability;
    inputs[off + VERDICT_IN.nearDeathStability] = SCORING.NEAR_DEATH_STABILITY;
    // Engine gain deltas, not the post-clamp EV bars (safety.ts:209-225).
    inputs[off + VERDICT_IN.expectedCompletion] = cand.expectedGains.completion;
    inputs[off + VERDICT_IN.expectedPerfection] = cand.expectedGains.perfection;
    inputs[off + VERDICT_IN.successCompletion] = cand.successGains.completion;
    inputs[off + VERDICT_IN.successPerfection] = cand.successGains.perfection;
    inputs[off + VERDICT_IN.failureCompletion] = failureRun[ev + NS_STATE.completion];
    inputs[off + VERDICT_IN.failurePerfection] = failureRun[ev + NS_STATE.perfection];
    inputs[off + VERDICT_IN.failureStability] = failureRun[ev + NS_STATE.stability];
    inputs[off + VERDICT_IN.isItem] = cand.skill.actionKind === 'item' ? 1 : 0;
    inputs[off + VERDICT_IN.targetCompletion] = env.targets.completion;
    inputs[off + VERDICT_IN.targetPerfection] = env.targets.perfection;
    inputs[off + VERDICT_IN.displayEndsCraft] = cand.displayEndsCraft ? 1 : 0;
    inputs[off + VERDICT_IN.initialMaxStability] = params[i * STATE_PARAM_WORDS + 5];
  }
  return inputs;
}

async function runVerdictGroup(
  ctx: DeviceContext,
  env: SearchEnv,
  records: Records,
  inputNodes: Float32Array,
  evNodes: Float32Array,
  params: Float32Array,
  candidates: readonly CandidateInfo[],
  safes: readonly SurvivalContext[],
): Promise<Array<{ requires: number; fatal: number; uncollapsedRisk: number; blendProbability: number }>> {
  if (candidates.length === 0) return [];
  const count = candidates.length;
  const successParams = Float32Array.from(params);
  const failureParams = Float32Array.from(params);
  for (let i = 0; i < count; i++) {
    successParams[i * STATE_PARAM_WORDS + 31] = 1;
    failureParams[i * STATE_PARAM_WORDS + 31] = 2;
  }
  const successRun = await runStateBatch(ctx, records, inputNodes, successParams);
  void successRun; // forced-success transition still runs; only failure bars feed the verdict
  const failureRun = await runStateBatch(ctx, records, inputNodes, failureParams);
  const inputs = prepareVerdictInputs(
    env, inputNodes, evNodes, failureRun, params, candidates, safes,
  );
  const out = await runVerdictBatch(ctx.device, ctx.pipelines.verdict, inputs);
  const result: Array<{ requires: number; fatal: number; uncollapsedRisk: number; blendProbability: number }> = [];
  for (let i = 0; i < count; i++) {
    const base = i * VERDICT_OUT.WORDS;
    result.push({
      requires: out[base + VERDICT_OUT.requires],
      fatal: out[base + VERDICT_OUT.fatal],
      uncollapsedRisk: out[base + VERDICT_OUT.uncollapsedRisk],
      blendProbability: out[base + VERDICT_OUT.blendProbability],
    });
  }
  return result;
}

/** Immutable `harmonyData` snapshot, uploaded once per search (S7d style buffer). */
function buildStyleWords(state: CraftingState): Float32Array {
  const words = new Float32Array(STYLE.WORDS);
  const hd = (state.harmonyData ?? {}) as Record<string, unknown>;
  const obj = (value: unknown): Record<string, unknown> =>
    value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  const alignment = (value: unknown): number =>
    value === 'fusion' ? 0 : value === 'refine' ? 1 : 2;
  const num = (value: unknown, fallback: number): number => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  if (hd.forgeWorks) {
    words[STYLE.kind] = 1;
    const fw = obj(hd.forgeWorks);
    const heat = num(fw.heat, 0);
    words[STYLE.forgeHeat] = heat;
    words[STYLE.forgeLastBuffedHeat] = num(fw.lastBuffedHeat, heat);
  }
  else if (hd.inscribedPatterns) {
    words[STYLE.kind] = 2;
    const ins = obj(hd.inscribedPatterns);
    words[STYLE.inscriptionStacks] = Number(ins.stacks ?? 0);
    words[STYLE.inscribedRemaining] = Array.isArray(ins.currentBlock) ? ins.currentBlock.length : 5;
  } else if (hd.alchemicalArts) {
    words[STYLE.kind] = 3;
    words[STYLE.alchemicalCharges] = Array.isArray(obj(hd.alchemicalArts).charges)
      ? (obj(hd.alchemicalArts).charges as unknown[]).length
      : 0;
    const mods: Record<string, number> =
      state.harmonyData && 'alchemicalArts' in state.harmonyData
        ? ({ ...getHarmonyStatModifiers(state.harmonyData, 'alchemical') } as Record<string, number>)
        : {};
    words[STYLE.alchemicalModsFinite] = Object.values(mods).every((v) => Number.isFinite(v)) ? 1 : 0;
    words[STYLE.alchemicalControlMult] = mods.controlMultiplier ?? 1;
    words[STYLE.alchemicalIntensityMult] = mods.intensityMultiplier ?? 1;
    words[STYLE.alchemicalCritBonus] = mods.critChanceBonus ?? 0;
    words[STYLE.alchemicalSuccessBonus] = mods.successChanceBonus ?? 0;
    words[STYLE.alchemicalQiReduction] = mods.qiCostReduction ?? 0;
    words[STYLE.alchemicalStabReduction] = mods.stabilityCostReduction ?? 0;
    const recommended = Array.isArray(hd.recommendedTechniqueTypes) ? hd.recommendedTechniqueTypes : [];
    words[STYLE.recommendedCount] = Math.min(recommended.length, 8);
    for (let i = 0; i < words[STYLE.recommendedCount]; i++) {
      words[STYLE.recommended + i] = alignment(recommended[i]);
    }
  } else if (hd.resonance) {
    words[STYLE.kind] = 4;
    const res = obj(hd.resonance);
    words[STYLE.resonanceStrength] = num(res.strength, 0);
    words[STYLE.resonanceType] = res.resonance === undefined ? -1 : alignment(res.resonance);
    words[STYLE.pendingResonanceType] = res.pendingResonance ? alignment(res.pendingResonance) : -1;
    words[STYLE.pendingCount] = Number(res.pendingCount ?? 0);
  } else if (hd.captivatingCadence) {
    words[STYLE.kind] = 5;
    words[STYLE.cadenceChain] = Number(obj(hd.captivatingCadence).chain ?? 0);
  }
  return words;
}

/** One dispatch replaces the host's per-case `buildScoreWords` loop. */
async function deviceScoreInputs(
  ctx: DeviceContext,
  cases: readonly ScoreCase[],
): Promise<Float32Array> {
  const count = cases.length;
  const nodes = new Float32Array(count * NS_STATE.WORDS);
  const style = new Float32Array(count * STYLE.WORDS);
  const stability = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    nodes.set(cases[i].node.subarray(0, NS_STATE.WORDS), i * NS_STATE.WORDS);
    // Until the engine's per-action harmony projection is ported, the
    // immutable snapshot is supplied per case from the case's harmony state.
    style.set(buildStyleWords(cases[i].harmonyState), i * STYLE.WORDS);
    stability[i] = cases[i].stability;
  }
  const device = ctx.device;
  const nodeBuf = buffer(device, nodes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const styleBuf = buffer(device, style, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const stabBuf = buffer(device, stability, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBuf = device.createBuffer({
    size: Math.max(16, count * SCORE_INPUT.WORDS * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: Math.max(16, count * SCORE_INPUT.WORDS * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const group = createScoreInputBindGroup(device, ctx.pipelines.scoreInput, nodeBuf, styleBuf, stabBuf, inputBuf);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(ctx.pipelines.scoreInput);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(inputBuf, 0, readback, 0, count * SCORE_INPUT.WORDS * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    return new Float32Array(readback.getMappedRange().slice(0, count * SCORE_INPUT.WORDS * 4));
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
    nodeBuf.destroy();
    styleBuf.destroy();
    stabBuf.destroy();
    inputBuf.destroy();
  }
}

// ---------------------------------------------------------------------------
// The search.
// ---------------------------------------------------------------------------

interface ScoreCase {
  readonly node: Float32Array;
  readonly stability: number;
  readonly harmonyState: CraftingState;
  readonly condition: string;
  readonly bits: number;
  readonly hostTerminal: boolean;
  readonly expected: readonly [number, number, number, number, number];
}

function pushScoreCase(
  cases: ScoreCase[],
  params: number[],
  env: SearchEnv,
  node: Float32Array,
  stability: number,
  harmonyState: CraftingState,
  condition: string,
  bits: number,
): number {
  const deltas = conditionDelta(env.config, condition);
  const scoreParamBlock = buildScoreParams({
    targetCompletion: env.targets.completion,
    targetPerfection: env.targets.perfection,
    isSublime: env.config.isSublimeCraft === true,
    trainingMode: env.config.trainingMode === true,
    targetMultiplier: env.config.targetMultiplier || 2,
    maxCompletion: env.config.maxCompletion,
    maxPerfection: env.config.maxPerfection,
    goalPriorityBias: env.searchCfg.goalPriorityBias ?? 0,
    overcraftAmbition: env.searchCfg.overcraftAmbition !== false,
    perfectionBandGoal: env.searchCfg.perfectionBandGoal ?? 0,
    completionBandCeiling: env.searchCfg.completionBandCeiling ?? 0,
    ctx: env.ctx,
    conditionControlDelta: deltas.control,
    conditionIntensityDelta: deltas.intensity,
    terminalBits: bits,
  });
  pushFloats(params, scoreParamBlock);
  const autoFinish = willAutoFinish(harmonyState, env.bands) ? 1 : 0;
  const isTerminal = (bits & 1) !== 0 ? 1 : 0;
  const terminalMode = autoFinish || isTerminal ? 1 : 0;
  const belowTarget = terminalMode && !goalsMet(harmonyState, env.targets.completion, env.targets.perfection) ? 1 : 0;
  const expected: [number, number, number, number, number] = bits & 2
    ? [0, 0, 0, 1, 0]
    : [autoFinish, isTerminal, autoFinish, terminalMode, belowTarget];
  cases.push({ node, stability, harmonyState, condition, bits, hostTerminal: (bits & 1) !== 0, expected });
  return cases.length - 1;
}

async function runPipeline(
  ctx: DeviceContext,
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  options: StatefulSearchOptions,
  profiler: GpuProfiler,
): Promise<StatefulSearchResult> {
  const condition = options.currentCondition ?? 'neutral';
  const configWithCondition = Object.assign({}, config, { currentCondition: condition }) as OptimizerConfig & {
    currentCondition: string;
  };
  const targets = {
    completion: options.targetCompletion ?? Number(config.targetCompletion ?? 0),
    perfection: options.targetPerfection ?? Number(config.targetPerfection ?? 0),
  };
  const notes: Record<string, number> = {};
  const requestedDepth = Number.isFinite(options.lookaheadDepth)
    ? Math.max(0, Math.min(96, Math.floor(options.lookaheadDepth as number)))
    : 2;
  notes.requestedDepth = requestedDepth;
  // Only the validated depth-2 pipeline exists today. A larger setting must be
  // refused rather than answered with a depth-2 plan.
  if (requestedDepth > 2) {
    notes.implementedDepth = 2;
    return refusal(
      `глубина ${requestedDepth} недоступна: device-resident beam ещё не подключён`,
      notes, profiler, now(),
    );
  }
  const records = buildRecords(state, configWithCondition, skills);
  const discordance = getBuffDiscordantConditions(state.buffs);
  // No parallel constants: every limit comes from the mod settings (or the
  // engine default when the caller omits it, as the legacy stand does).
  const searchCfg: SearchConfig = {
    ...DEFAULT_SEARCH_CONFIG,
    ...options.searchConfig,
    timeBudgetMs: Number.isFinite(options.timeBudgetMs)
      ? Math.max(1, Math.floor(options.timeBudgetMs as number))
      : DEFAULT_SEARCH_CONFIG.timeBudgetMs,
    maxNodes: Number.isFinite(options.maxNodes)
      ? Math.max(1, Math.floor(options.maxNodes as number))
      : DEFAULT_SEARCH_CONFIG.maxNodes,
    beamWidth: Number.isFinite(options.beamWidth)
      ? Math.max(1, Math.floor(options.beamWidth as number))
      : DEFAULT_SEARCH_CONFIG.beamWidth,
    conditionDiscordance: discordance,
  };
  notes.timeBudgetMs = searchCfg.timeBudgetMs;
  notes.maxNodes = searchCfg.maxNodes;
  notes.beamWidth = searchCfg.beamWidth;
  const queue = options.conditionQueue
    ? [...options.conditionQueue]
    : normalizeForecastConditionQueue(condition, [], state.harmony, 3, discordance);
  const scoringCtx = __testing.buildScoringContext(configWithCondition, state, condition);
  const bands = deriveBandGoals(
    targets.completion,
    targets.perfection,
    config.isSublimeCraft === true,
    config.maxCompletion,
    config.maxPerfection,
    searchCfg.perfectionBandGoal,
    searchCfg.completionBandCeiling,
  );
  const env: SearchEnv = {
    config: configWithCondition,
    configState: state,
    searchCfg,
    ctx: scoringCtx,
    targets,
    modeCompGoal: bands.effectiveCompGoal,
    modePerfGoal: bands.effectivePerfGoal,
    bands: bands.bands,
  };
  profiler.mark('records');

  const encodeStart = now();
  // ---- root availability (device decides) -----------------------------------
  let available1 = await deviceAvailableSkills(
    ctx, records, records.rootNode, state, condition, configWithCondition, skills, notes,
  );
  if (available1.length === 0) {
    return refusal('NET: устройство не нашло доступных навыков', notes, profiler, encodeStart);
  }

  // ---- root skills -----------------------------------------------------------
  const rootNodes = new Float32Array(available1.length * NS_STATE.WORDS);
  const rootParams = new Float32Array(available1.length * STATE_PARAM_WORDS);
  const rootCandidates: CandidateInfo[] = [];
  const safeRoot = safeContext(env, state, available1, condition);
  for (let i = 0; i < available1.length; i++) {
    const skill = available1[i];
    const display = applySkill(
      state, skill, configWithCondition,
      effectsOf(config, condition) as Parameters<typeof applySkill>[3],
      targets.completion, condition, targets.perfection,
    );
    if (!display) continue;
    rootCandidates.push(buildCandidate(env, state, skill, display, condition, safeRoot));
    rootNodes.set(records.rootNode, i * NS_STATE.WORDS);
    const base = i * NS_STATE.WORDS;
    rootNodes[base + NS_STATE.conditionId] = conditionIdOf(condition);
    for (let q = 0; q < 3; q++) {
      rootNodes[base + NS_STATE.conditionQueue + q] =
        CONDITION_KIND[queue[q] as keyof typeof CONDITION_KIND] ?? CONDITION_KIND.neutral;
    }
    const skillIndex = records.skillSet.names.findIndex(
      (n) => normalizeSkill(n) === normalizeSkill(skill.name),
    );
    rootParams.set(
      makeStateParams(records.paramsBase, skillIndex, condition, records.rootNode, 0, configWithCondition),
      i * STATE_PARAM_WORDS,
    );
  }
  if (rootCandidates.length === 0) {
    return refusal('NET: ни один корневой навык не дал переход', notes, profiler, encodeStart);
  }
  const rootRun = await runStateBatch(ctx, records, rootNodes, rootParams);
  const rootVerdicts = await runVerdictGroup(
    ctx, env, records, rootNodes, rootRun, rootParams, rootCandidates,
    rootCandidates.map(() => safeRoot),
  );
  notes.stateNodes = (notes.stateNodes ?? 0) + rootCandidates.length;

  const scoreParams: number[] = [];
  const scoreCases: ScoreCase[] = [];
  const rootDeath: number[] = [];
  const rootFinished: number[] = [];
  for (let i = 0; i < rootCandidates.length; i++) {
    const cand = rootCandidates[i];
    const node = rootRun.slice(i * NS_STATE.WORDS, (i + 1) * NS_STATE.WORDS);
    const floor = calculateActionSurvivabilityFloor(
      state, cand.skill, configWithCondition,
      effectsOf(config, condition) as Parameters<typeof calculateActionSurvivabilityFloor>[3],
      condition, cand.display,
    );
    rootDeath[i] = uncollapsedRisk(cand, floor)
      ? pushScoreCase(
          scoreCases, scoreParams, env, node,
          applySurvivabilityFloorToState(cand.display, floor).stability,
          cand.display, condition, 2,
        )
      : -1;
    rootFinished[i] = cand.endsCraft
      ? pushScoreCase(
          scoreCases, scoreParams, env, node,
          cand.searchState.stability, cand.searchState, condition, 2,
        )
      : -1;
  }

  // ---- condition transitions after root --------------------------------------
  const condParams1 = new Float32Array(rootCandidates.length * COND_PARAM_WORDS);
  for (let i = 0; i < rootCandidates.length; i++) {
    condParams1.set(
      makeCondParams(
        rootRun[i * NS_STATE.WORDS + NS_STATE.conditionGenerationHarmony],
        discordance, searchCfg, condition,
      ),
      i * COND_PARAM_WORDS,
    );
  }
  const condRun1 = await runConditionBatch(ctx, rootRun, condParams1);
  notes.branchCases = (notes.branchCases ?? 0) + rootCandidates.length;

  interface Level1 {
    readonly root: number;
    readonly branch: number;
    readonly condition: string;
    readonly queue: string[];
    readonly probability: number;
    readonly state: CraftingState;
    readonly cand: CandidateInfo;
    readonly node: Float32Array;
    readonly nodeCase: number;
  }
  const level1: Level1[] = [];
  const level1Nodes: number[] = [];
  for (let i = 0; i < rootCandidates.length; i++) {
    const cand = rootCandidates[i];
    const branches = decodeBranches(condRun1, i);
    for (let b = 0; b < branches.length; b++) {
      const branch = branches[b];
      const nextCondition = conditionNameOfKind(branch.conditionId);
      const nextQueue = branch.queue.map(conditionNameOfKind);
      const evNode = Float32Array.from(rootRun.slice(i * NS_STATE.WORDS, (i + 1) * NS_STATE.WORDS));
      evNode[NS_STATE.conditionId] = conditionIdOf(nextCondition);
      for (let q = 0; q < 3; q++) {
        evNode[NS_STATE.conditionQueue + q] =
          CONDITION_KIND[nextQueue[q] as keyof typeof CONDITION_KIND] ?? CONDITION_KIND.neutral;
      }
      const node = continuationNode(evNode, cand);
      pushFloats(level1Nodes, node);
      const nodeCase = pushScoreCase(
        scoreCases, scoreParams, env, node,
        cand.searchState.stability, cand.searchState, nextCondition, 2,
      );
      level1.push({
        root: i, branch: b, condition: nextCondition, queue: nextQueue,
        probability: branch.probability, state: cand.searchState, cand, node, nodeCase,
      });
      notes.branches = (notes.branches ?? 0) + 1;
    }
  }
  if (level1.length === 0) {
    return refusal('NET: устройство не вернуло переходы условий', notes, profiler, encodeStart);
  }

  // ---- second skills ---------------------------------------------------------
  interface Level2 {
    readonly meta: Level1;
    readonly info: CandidateInfo;
    readonly node: Float32Array;
    readonly index: number;
    readonly safe: SurvivalContext;
  }
  const level2: Level2[] = [];
  const l2Input: number[] = [];
  const l2Params: number[] = [];
  const l1Flat = Float32Array.from(level1Nodes);
  for (let n = 0; n < level1.length; n++) {
    const meta = level1[n];
    const node = l1Flat.slice(n * NS_STATE.WORDS, (n + 1) * NS_STATE.WORDS);
    const skills2 = await deviceAvailableSkills(
      ctx, records, node, meta.state, meta.condition, configWithCondition, skills, notes,
    );
    const safe2 = safeContext(env, meta.state, skills2, meta.condition);
    for (const skill of skills2) {
      const display = applySkill(
        meta.state, skill, configWithCondition,
        effectsOf(config, meta.condition) as Parameters<typeof applySkill>[3],
        targets.completion, meta.condition, targets.perfection,
      );
      if (!display) continue;
      const info = buildCandidate(env, meta.state, skill, display, meta.condition, safe2);
      const index = level2.length;
      level2.push({ meta, info, node, index, safe: safe2 });
      pushFloats(l2Input, node);
      const skillIndex = records.skillSet.names.findIndex(
        (name) => normalizeSkill(name) === normalizeSkill(skill.name),
      );
      pushFloats(l2Params, makeStateParams(
        records.paramsBase, skillIndex, meta.condition, node, 0, configWithCondition,
      ));
    }
  }
  if (level2.length === 0) {
    return refusal('NET: устройство не нашло вторых навыков', notes, profiler, encodeStart);
  }
  const l2Run = await runStateBatch(ctx, records, Float32Array.from(l2Input), Float32Array.from(l2Params));
  const l2Verdicts = await runVerdictGroup(
    ctx, env, records, Float32Array.from(l2Input), l2Run,
    Float32Array.from(l2Params), level2.map((c) => c.info), level2.map((c) => c.safe),
  );
  notes.stateNodes += level2.length;

  const level2Death: number[] = [];
  const level2Finished: number[] = [];
  for (let i = 0; i < level2.length; i++) {
    const c = level2[i];
    const node = l2Run.slice(i * NS_STATE.WORDS, (i + 1) * NS_STATE.WORDS);
    const floor = calculateActionSurvivabilityFloor(
      c.meta.state, c.info.skill, configWithCondition,
      effectsOf(config, c.meta.condition) as Parameters<typeof calculateActionSurvivabilityFloor>[3],
      c.meta.condition, c.info.display,
    );
    level2Death[i] = uncollapsedRisk(c.info, floor)
      ? pushScoreCase(
          scoreCases, scoreParams, env, node,
          applySurvivabilityFloorToState(c.info.display, floor).stability,
          c.info.display, c.meta.condition, 2,
        )
      : -1;
    level2Finished[i] = c.info.endsCraft
      ? pushScoreCase(
          scoreCases, scoreParams, env, node,
          c.info.searchState.stability, c.info.searchState, c.meta.condition, 2,
        )
      : -1;
  }

  // ---- condition transitions after second skill ------------------------------
  const condParams2 = new Float32Array(level2.length * COND_PARAM_WORDS);
  for (let i = 0; i < level2.length; i++) {
    condParams2.set(
      makeCondParams(
        l2Run[i * NS_STATE.WORDS + NS_STATE.conditionGenerationHarmony],
        discordance, searchCfg, level2[i].meta.condition,
      ),
      i * COND_PARAM_WORDS,
    );
  }
  const condRun2 = await runConditionBatch(ctx, l2Run, condParams2);
  notes.branchCases += level2.length;

  const leaves: Array<{
    readonly l2: number;
    readonly root: number;
    readonly branch: number;
    readonly probability: number;
    readonly caseIndex: number;
  }> = [];
  for (let i = 0; i < level2.length; i++) {
    const c = level2[i];
    const node = l2Run.slice(i * NS_STATE.WORDS, (i + 1) * NS_STATE.WORDS);
    const branches = decodeBranches(condRun2, i);
    for (let b = 0; b < branches.length; b++) {
      const branch = branches[b];
      const nextCondition = conditionNameOfKind(branch.conditionId);
      const hostBits = isTerminalState(c.info.searchState, configWithCondition, nextCondition) ? 1 : 0;
      const caseIndex = pushScoreCase(
        scoreCases, scoreParams, env, node,
        c.info.searchState.stability, c.info.searchState, nextCondition, hostBits,
      );
      leaves.push({
        l2: i, root: c.meta.root, branch: c.meta.branch,
        probability: branch.probability, caseIndex,
      });
      notes.branches++;
    }
  }

  // ---- one batched score pass; device decides terminality ---------------------
  const scoreRun = await runScoreBatch(
    ctx, await deviceScoreInputs(ctx, scoreCases), Float32Array.from(scoreParams),
  );
  notes.scoreInputKernelCases = scoreCases.length;
  notes.scoreCases = scoreCases.length;
  for (let i = 0; i < scoreCases.length; i++) {
    const want = scoreCases[i].expected;
    const got = scoreRun.terminality.slice(i * TERMINALITY_WORDS, (i + 1) * TERMINALITY_WORDS);
    if (got[3] === want[3]) notes.terminalModeMatched = (notes.terminalModeMatched ?? 0) + 1;
    else notes.terminalModeMismatched = (notes.terminalModeMismatched ?? 0) + 1;
    if (scoreCases[i].bits & 2) continue;
    let all = true;
    for (let w = 0; w < 5; w++) if (got[w] !== want[w]) all = false;
    if (all) notes.terminalityMatched = (notes.terminalityMatched ?? 0) + 1;
    else notes.terminalityMismatched = (notes.terminalityMismatched ?? 0) + 1;
  }

  // ---- reduction: device blend, tie window, final ranking --------------------
  const tieWindow = computeScoreTieWindow(Math.max(1, env.modeCompGoal + env.modePerfGoal));
  const rootScores = new Map<string, number>();
  const rootBestSecond = new Map<string, string>();
  const metricOf = (current: CraftingState, cand: CandidateInfo) => ({
    immediateProgress: calculateImmediateProgressTowardGoals(
      current, cand.display, env.modeCompGoal, env.modePerfGoal,
      targets.completion, targets.perfection,
    ),
    isStabilize: cand.skill.type === 'stabilize',
    requiresProbabilisticSurvival: cand.requiresProbabilisticSurvival,
    qiSpent: Math.max(0, current.qi - cand.display.qi),
    skillKey: cand.skill.key,
  });

  const makeBlendCases = (edges: BlendInput[]): Float32Array => {
    const out = new Float32Array(edges.length * BLEND_IN_WORDS);
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      out[i * BLEND_IN_WORDS] = e.risk;
      out[i * BLEND_IN_WORDS + 1] = e.p;
      out[i * BLEND_IN_WORDS + 2] = e.value;
      out[i * BLEND_IN_WORDS + 3] = e.death;
    }
    return out;
  };
  interface BlendInput {
    readonly owner: string;
    readonly slot: number;
    readonly probability: number;
    readonly risk: number;
    readonly p: number;
    readonly value: number;
    readonly death: number;
  }
  const survivalProbabilityOf = (
    currentState: CraftingState,
    cand: CandidateInfo,
    atCondition: string,
  ): number => {
    const floor = calculateActionSurvivabilityFloor(
      currentState, cand.skill, configWithCondition,
      effectsOf(config, atCondition) as Parameters<typeof calculateActionSurvivabilityFloor>[3],
      atCondition, cand.display,
    );
    return floor?.survivalProbability ?? 0;
  };

  // Phase 1: every depth-2 edge value, with survival mixes taken from BLEND_WGSL.
  const l2Inputs: BlendInput[] = [];
  const l2SecondValue = new Map<number, number>();
  for (let i = 0; i < level2.length; i++) {
    const c = level2[i];
    let value = 0;
    for (const leaf of leaves) {
      if (leaf.l2 === i) value += leaf.probability * scoreRun.scores[leaf.caseIndex];
    }
    if (c.info.endsCraft) {
      const fin = level2Finished[i];
      if (fin >= 0) value = scoreRun.scores[fin];
    } else {
      const risk = l2Verdicts[i]?.uncollapsedRisk ?? 0;
      const death = level2Death[i] ?? -1;
      if (risk >= 0.5 && death >= 0) {
        l2Inputs.push({
          owner: `l2:${i}`, probability: c.meta.probability, slot: l2Inputs.length,
          risk, p: survivalProbabilityOf(c.meta.state, c.info, c.meta.condition), value, death: scoreRun.scores[death],
        });
      } else if (risk >= 0.5 && death < 0) {
        notes.missingDeathCases = (notes.missingDeathCases ?? 0) + 1;
        value = Number.NaN;
      }
    }
    l2SecondValue.set(i, value);
  }
  const l2Blended = l2Inputs.length > 0
    ? await runBlendBatch(ctx, makeBlendCases(l2Inputs))
    : new Float32Array(0);
  notes.blendCases = l2Inputs.length;
  for (const edge of l2Inputs) l2SecondValue.set(Number(edge.owner.slice(3)), l2Blended[edge.slot]);

  // Phase 2: root values from the (already blended) depth-2 values.
  const rootInputs: BlendInput[] = [];
  for (let r = 0; r < rootCandidates.length; r++) {
    const cand = rootCandidates[r];
    if (cand.endsCraft) {
      const fin = rootFinished[r];
      rootScores.set(cand.skill.key, fin >= 0 ? scoreRun.scores[fin] : Number.NaN);
      continue;
    }
    let expected = 0;
    for (const meta of level1) {
      if (meta.root !== r) continue;
      let branchValue: number;
      if (isTerminalState(cand.searchState, configWithCondition, meta.condition)) {
        branchValue = scoreRun.scores[meta.nodeCase];
      } else {
        const skillIds: number[] = [];
        for (const c of level2) {
          if (c.meta.root === r && c.meta.branch === meta.branch && skillIds.indexOf(c.index) < 0) {
            skillIds.push(c.index);
          }
        }
        let bestScore = -Infinity;
        let bestInfo: CandidateInfo | null = null;
        let bestKey = '';
        if (skillIds.length === 0) bestScore = scoreRun.scores[meta.nodeCase];
        for (const skillId of skillIds) {
          const info = level2[skillId].info;
          const value = l2SecondValue.get(skillId) ?? Number.NaN;
          const delta = value - bestScore;
          const tie =
            Math.abs(delta) <= tieWindow && bestInfo !== null
              ? compareTieBreakMetrics(
                  metricOf(cand.searchState, info),
                  metricOf(cand.searchState, bestInfo),
                ) > 0
              : false;
          if (delta > tieWindow || tie || bestInfo === null) {
            bestScore = value;
            bestInfo = info;
            bestKey = info.skill.key;
          }
        }
        branchValue = bestScore;
        if (bestInfo && bestKey) rootBestSecond.set(cand.skill.key, bestKey);
      }
      if (!Number.isNaN(branchValue)) expected += meta.probability * branchValue;
      else notes.nonFiniteScores = (notes.nonFiniteScores ?? 0) + 1;
    }
    const risk = rootVerdicts[r]?.uncollapsedRisk ?? 0;
    const death = rootDeath[r] ?? -1;
    if (risk >= 0.5 && death >= 0) {
      rootInputs.push({
        owner: `root:${r}`, probability: 1, slot: rootInputs.length,
        risk, p: survivalProbabilityOf(state, cand, condition), value: expected, death: scoreRun.scores[death],
      });
    } else if (risk >= 0.5 && death < 0) {
      notes.missingDeathCases = (notes.missingDeathCases ?? 0) + 1;
      expected = Number.NaN;
    }
    rootScores.set(cand.skill.key, expected);
  }
  const rootBlended = rootInputs.length > 0
    ? await runBlendBatch(ctx, makeBlendCases(rootInputs))
    : new Float32Array(0);
  notes.blendCases += rootInputs.length;
  for (const edge of rootInputs) {
    const r = Number(edge.owner.slice(5));
    const key = rootCandidates[r].skill.key;
    rootScores.set(key, rootBlended[edge.slot]);
  }

  // ---- final ranking (engine rule: safety, tie metrics, diversity) -----------
  interface RankRow {
    key: string;
    score: number;
    unsafe: boolean;
    yields: boolean;
    below: boolean;
    requires: boolean;
    type: string;
  }
  const unsafeKeys = new Set<string>(rootCandidates.filter((c) => c.unsafe).map((c) => c.skill.key));
  const rows: RankRow[] = rootCandidates.map((cand) => {
    const below = cand.endsCraft && !goalsMet(cand.searchState, targets.completion, targets.perfection);
    return {
      key: cand.skill.key,
      score: rootScores.get(cand.skill.key) ?? Number.NaN,
      unsafe: unsafeKeys.has(cand.skill.key),
      yields: unsafeKeys.has(cand.skill.key),
      below,
      requires: cand.requiresProbabilisticSurvival,
      type: cand.skill.type,
    };
  });
  const candidateByKey = new Map(rootCandidates.map((c) => [c.skill.key, c] as const));
  const ranked = orderRecommendationsBySafety(
    rows as unknown as SkillRecommendation[],
    (a, b) => {
      const ra = a as unknown as RankRow;
      const rb = b as unknown as RankRow;
      const diff = rb.score - ra.score;
      if (Math.abs(diff) > tieWindow) return diff;
      const ca = candidateByKey.get(ra.key);
      const cb = candidateByKey.get(rb.key);
      if (ca && cb) {
        const t = compareTieBreakMetrics(metricOf(state, ca), metricOf(state, cb));
        if (t !== 0) return t > 0 ? -1 : 1;
      }
      return ra.key.localeCompare(rb.key);
    },
    (row) => unsafeKeys.has((row as unknown as RankRow).key),
  ) as unknown as RankRow[];
  const safetyGroup = (row: RankRow): boolean =>
    unsafeKeys.has(row.key) || row.requires || row.yields || row.below;
  const finalRows: RankRow[] = ranked.length > 0 ? [ranked[0]] : [];
  const usedTypes = new Set<string>(ranked.length > 0 ? [ranked[0].type] : []);
  const remaining = ranked.slice(1);
  while (remaining.length > 0) {
    const topScore = remaining[0].score;
    const topSafety = safetyGroup(remaining[0]);
    const end = remaining.findIndex(
      (row) => safetyGroup(row) !== topSafety || Math.abs(topScore - row.score) > 1,
    );
    const groupLen = end === -1 ? remaining.length : end;
    let pick = 0;
    for (let i = 0; i < groupLen; i++) {
      if (!usedTypes.has(remaining[i].type)) {
        pick = i;
        break;
      }
    }
    const next = remaining.splice(pick, 1)[0];
    finalRows.push(next);
    usedTypes.add(next.type);
  }

  const order = finalRows.map((row) => row.key);
  const chosen = order[0] ?? null;
  const scores: Record<string, number> = {};
  rootScores.forEach((score, key) => {
    scores[key] = score;
  });
  const path: string[] = [];
  if (chosen) {
    path.push(chosen);
    const second = rootBestSecond.get(chosen);
    if (second) path.push(second);
  }
  return {
    chosen,
    score: chosen ? (rootScores.get(chosen) ?? Number.NaN) : Number.NaN,
    path,
    order,
    scores,
    timing: {
      encodeMs: now() - encodeStart,
      gpuMs: 0,
      readbackMs: 0,
      totalMs: profiler.finish().totalMs,
    },
    notes,
  };
}

function refusal(
  reason: string,
  notes: Record<string, number>,
  profiler: GpuProfiler,
  encodeStart: number,
): StatefulSearchResult {
  notes.refusals = (notes.refusals ?? 0) + 1;
  return {
    chosen: null,
    score: Number.NaN,
    path: [],
    order: [],
    scores: {},
    timing: {
      encodeMs: now() - encodeStart,
      gpuMs: 0,
      readbackMs: 0,
      totalMs: profiler.elapsedSince('start'),
    },
    notes,
    refusal: reason,
  };
}

/**
 * One full depth-2 stateful search. Every failure — capability, pipeline
 * compilation, device loss, timeout or kernel error — is converted into a
 * loud refusal in the returned result; no exception reaches the caller.
 */
export async function runStatefulSearch(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  options: StatefulSearchOptions = {},
): Promise<StatefulSearchResult> {
  const profiler = new GpuProfiler(now);
  // The engine time budget from the mod settings is the deadline. `timeoutMs`
  // remains a watchdog only for legacy callers that do not pass a budget.
  const timeoutMs = Math.max(
    1,
    Math.min(
      60_000,
      Math.floor(options.timeBudgetMs ?? options.timeoutMs ?? 20_000),
    ),
  );
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const message = `WebGPU-таймаут ${timeoutMs} мс (прошло ${profiler.elapsedSince('start').toFixed(0)} мс)`;
      gpuDebugLog(`stateful timeout limit=${timeoutMs}`);
      if (options.disableOnTimeout !== false) disableStatefulGpuBackend(message);
      reject(new Error(message));
    }, timeoutMs);
  });
  try {
    const ctx = await getContext();
    const result = await Promise.race([runPipeline(ctx, state, config, skills, options, profiler), timeout]);
    gpuDebugLog(`stateful done chosen=${result.chosen ?? '-'} notes=${JSON.stringify(result.notes)}`);
    return result;
  } catch (error: unknown) {
    const message = `WebGPU-поиск прерван: ${objEventMessage(error)}`;
    gpuDebugLog(`stateful error phase=search message=${message}`);
    disableStatefulGpuBackend(message);
    return refusal(message, {}, profiler, now());
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
