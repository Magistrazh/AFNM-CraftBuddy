/**
 * S7b: the self-contained stateful GPU search, wired into the recommendation
 * path without touching the engine, the compact kernel or the settings schema.
 *
 * The GPU only proposes an opening. Every number the panel sees still comes
 * from one ordinary engine search restricted to that opening, so gains, costs
 * and the outcome projection are real engine fields, never GPU heuristics.
 *
 * This module never throws to the caller: every refusal is a typed `ok: false`
 * outcome and the caller keeps its existing JS ladder untouched.
 */

import {
  findBestSkill,
  type CraftingConditionType,
  type OptimizerConfig,
  type SearchConfig,
  type SearchResult,
  type SkillDefinition,
} from '../optimizer';
import type { CraftingState } from '../optimizer/state';
import {
  runStatefulSearch,
  statefulGpuBackendStatus,
  type StatefulSearchResult,
} from '../optimizer/gpu/statefulSearch';
import { gpuDebugLog } from '../optimizer/gpu/debug';
import type { ResidentSearchResult } from '../optimizer/gpu/residentSearch';
import { emitDebugEvent } from '../utils/debugSink';

/**
 * Strict `engineOrder` gate accepted for the stateful port: 183 of 184 cases.
 * This is the accepted limit recorded in data/gpu-full-port-plan.md (P1 S4),
 * not a per-craft measurement; it is printed as the run's parity contract.
 */
export const GPU_STATEFUL_PARITY_LIMIT = '183/184';

/** Verification budget for the one engine search that fills the panel fields. */
export const GPU_STATEFUL_VERIFY_BUDGET_MS = 500;
/** Warm search ceiling: a slow batch refuses, not hangs. */
export const GPU_STATEFUL_TIMEOUT_MS = 2500;
/**
 * First-call ceiling. The first search compiles every compute pipeline (state,
 * conditions, availability, verdict, score, blend) and measured 347 ms in a
 * warm browser but exceeded 2.5 s in the game's Electron adapter. The goal is
 * sub-second AFTER warm-up, so cold compilation gets its own budget.
 */
export const GPU_STATEFUL_COLD_TIMEOUT_MS = 20000;

/** Notes that mean the device ran on incomplete records, not normal counters. */
const INCOMPLETE_NOTE_KEYS: readonly string[] = [
  'refusals',
  'terminalModeMismatched',
  'terminalityMismatched',
  'nonFiniteScores',
  'missingDeathCases',
];

export interface StatefulRecommendationInput {
  readonly state: CraftingState;
  readonly config: OptimizerConfig;
  readonly skills: readonly SkillDefinition[];
  readonly completion: number;
  readonly perfection: number;
  readonly lookaheadDepth: number;
  readonly condition: CraftingConditionType | undefined;
  readonly forecast: readonly CraftingConditionType[];
  readonly searchConfig: Partial<SearchConfig> & { timeBudgetMs: number };
  readonly timeoutMs?: number;
}

interface StatefulOutcomeBase {
  /** Wall time around the stateful call, including device acquisition. */
  readonly wallMs: number;
  /** Device-side timing from `StatefulSearchResult`; no values are inferred. */
  readonly timing: StatefulSearchResult['timing'];
  readonly notes: Record<string, number>;
}

export type StatefulRecommendationOutcome =
  | (StatefulOutcomeBase & {
      readonly ok: true;
      readonly result: SearchResult;
      readonly chosenKey: string;
      readonly chosenName: string;
      readonly gpuScore: number;
      readonly path: readonly string[];
      readonly order: readonly string[];
      readonly status: string;
    })
  | (StatefulOutcomeBase & {
      readonly ok: false;
      readonly reason: string;
      readonly refusal: string | null;
    });

export interface StatefulRecommendationDeps {
  readonly runSearch?: typeof runStatefulSearch;
  readonly find?: typeof findBestSkill;
  readonly now?: () => number;
  readonly log?: (...parts: unknown[]) => void;
  readonly emit?: (event: Record<string, unknown>) => void;
  readonly backendStatus?: () => { active: boolean };
}

type ResidentSearchFn = typeof import('../optimizer/gpu/residentSearch').runResidentSearch;

/**
 * The resident module reads WebGPU globals at load time, so it is required
 * lazily: unit tests inject a fake and never load the real kernels, while the
 * webpack bundle still resolves this literal require into the main chunk.
 */
function defaultResidentSearch(): ResidentSearchFn {
  const module = require('../optimizer/gpu/residentSearch') as {
    runResidentSearch: ResidentSearchFn;
  };
  return module.runResidentSearch;
}

/** Dependency surface of the depth router; the old facade stays untouched. */
export interface GpuRecommendationDeps extends StatefulRecommendationDeps {
  readonly runResidentSearch?: ResidentSearchFn;
}

function defaultNow(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function incompleteNotes(notes: Record<string, number>): string | null {
  for (const key of INCOMPLETE_NOTE_KEYS) {
    if ((notes[key] ?? 0) > 0) return `неполные записи stateful GPU: ${key}=${notes[key]}`;
  }
  return null;
}

/**
 * Run one stateful search. Returns a refusal outcome instead of throwing for
 * no WebGPU, a software adapter, incomplete records, timeout or device loss.
 * `deps` is only for isolated tests; production uses the real backend/engine.
 */
export async function tryStatefulRecommendation(
  input: StatefulRecommendationInput,
  deps: StatefulRecommendationDeps = {},
): Promise<StatefulRecommendationOutcome> {
  const runSearch = deps.runSearch ?? runStatefulSearch;
  const find = deps.find ?? findBestSkill;
  const now = deps.now ?? defaultNow;
  const log = deps.log ?? gpuDebugLog;
  const emit = deps.emit ?? emitDebugEvent;

  const started = now();
  let gpu: StatefulSearchResult;
  // Cold compilation of every pipeline gets its own budget; warm searches stay
  // under the tight ceiling. The sub-second goal applies after warm-up.
  const cold = !(deps.backendStatus ?? statefulGpuBackendStatus)().active;
  const timeoutMs =
    input.timeoutMs ??
    (cold ? GPU_STATEFUL_COLD_TIMEOUT_MS : GPU_STATEFUL_TIMEOUT_MS);
  log(`stateful ${cold ? 'cold' : 'warm'} timeout=${timeoutMs}`);
  try {
    gpu = await runSearch(
      input.state,
      input.config,
      [...input.skills],
      {
        currentCondition: input.condition,
        conditionQueue: input.forecast.slice(0, 3),
        targetCompletion: input.completion,
        targetPerfection: input.perfection,
        // Settings contract: pass the engine values unchanged. No parallel
        // depth/budget/node/beam constants on the GPU side.
        lookaheadDepth: input.lookaheadDepth,
        timeBudgetMs: input.searchConfig.timeBudgetMs,
        maxNodes: input.searchConfig.maxNodes,
        beamWidth: input.searchConfig.beamWidth,
        searchConfig: input.searchConfig,
        // Watchdog only; the search deadline itself is `timeBudgetMs` above.
        timeoutMs,
        // A single slow craft must not disable the backend for the session.
        disableOnTimeout: false,
      },
    );
  } catch (error) {
    gpu = {
      chosen: null,
      score: Number.NaN,
      path: [],
      order: [],
      scores: {},
      timing: { encodeMs: 0, gpuMs: 0, readbackMs: 0, totalMs: 0 },
      notes: {},
      refusal: `исключение stateful GPU: ${String(error)}`,
    };
  }
  const wallMs = now() - started;

  const refusalReason = (): string => {
    if (gpu.refusal) return `stateful-поиск отказал: ${gpu.refusal}`;
    const incomplete = incompleteNotes(gpu.notes);
    if (incomplete) return incomplete;
    if (!gpu.chosen) return 'stateful-поиск не выбрал ход';
    return '';
  };
  const reason = refusalReason();
  const chosen = reason ? null : input.skills.find((skill) => skill.key === gpu.chosen);

  const base = {
    wallMs,
    timing: gpu.timing,
    notes: gpu.notes,
  };
  const refuse = (finalReason: string): StatefulRecommendationOutcome => {
    emit({
      kind: 'gpu-stateful',
      ok: false,
      reason: finalReason,
      refusal: gpu.refusal ?? null,
      wallMs,
      timing: { ...gpu.timing },
      notes: { ...gpu.notes },
      parityLimit: GPU_STATEFUL_PARITY_LIMIT,
    });
    log(
      `stateful-wiring ok=false reason=${finalReason} wallMs=${wallMs.toFixed(1)} ` +
        `totalMs=${gpu.timing.totalMs.toFixed(1)} parity=${GPU_STATEFUL_PARITY_LIMIT}`,
    );
    return { ok: false, ...base, reason: finalReason, refusal: gpu.refusal ?? null };
  };
  if (reason || !chosen) {
    return refuse(
      reason || `ход stateful GPU ${String(gpu.chosen)} не найден в наборе приёмов`,
    );
  }

  // One ordinary engine search, restricted to the GPU's opening. The panel
  // receives the engine's own gains, costs and projection from this result.
  const verifyBudgetMs = Math.min(
    input.searchConfig.timeBudgetMs,
    GPU_STATEFUL_VERIFY_BUDGET_MS,
  );
  let result: SearchResult;
  try {
    result = find(
      input.state,
      input.config,
      input.completion,
      input.perfection,
      false,
      input.lookaheadDepth,
      input.condition,
      [...input.forecast],
      { ...input.searchConfig, rootSkillKeys: [chosen.key], timeBudgetMs: verifyBudgetMs },
    );
  } catch (error) {
    return refuse(`проверка хода stateful GPU прервалась: ${String(error)}`);
  }

  const status =
    `GPU stateful → engine: ход ${chosen.name}, ${wallMs.toFixed(0)} мс, ` +
    `ядро ${gpu.timing.totalMs.toFixed(0)} мс, паритет ${GPU_STATEFUL_PARITY_LIMIT}`;
  emit({
    kind: 'gpu-stateful',
    ok: true,
    engine: 'stateful-gpu+engine',
    chosen: chosen.key,
    chosenName: chosen.name,
    score: gpu.score,
    path: [...gpu.path],
    order: [...gpu.order],
    wallMs,
    timing: { ...gpu.timing },
    notes: { ...gpu.notes },
    verifyBudgetMs,
    parityLimit: GPU_STATEFUL_PARITY_LIMIT,
    status,
  });
  log(
    `stateful-wiring ok=true engine=stateful-gpu+engine chosen=${chosen.key} ` +
      `wallMs=${wallMs.toFixed(1)} totalMs=${gpu.timing.totalMs.toFixed(1)} ` +
      `encodeMs=${gpu.timing.encodeMs.toFixed(1)} gpuMs=${gpu.timing.gpuMs.toFixed(1)} ` +
      `readbackMs=${gpu.timing.readbackMs.toFixed(1)} ` +
      `path=${gpu.path.join('>')} verifyBudgetMs=${verifyBudgetMs} ` +
      `parity=${GPU_STATEFUL_PARITY_LIMIT}`,
  );

  return {
    ok: true,
    ...base,
    result,
    chosenKey: chosen.key,
    chosenName: chosen.name,
    gpuScore: gpu.score,
    path: gpu.path,
    order: gpu.order,
    status,
  };
}

/**
 * Depth > 2 facade. Same settings, same engine verification as the depth-2
 * path; the proposal itself comes from the validated device-resident beam.
 * Every refusal is typed and loud, and no exception reaches the UI.
 */
export async function tryResidentRecommendation(
  input: StatefulRecommendationInput,
  deps: GpuRecommendationDeps = {},
): Promise<StatefulRecommendationOutcome> {
  const run = deps.runResidentSearch ?? defaultResidentSearch();
  const find = deps.find ?? findBestSkill;
  const now = deps.now ?? defaultNow;
  const log = deps.log ?? gpuDebugLog;
  const emit = deps.emit ?? emitDebugEvent;
  const started = now();
  let gpu: ResidentSearchResult;

  try {
    gpu = await run({
      state: input.state,
      config: input.config,
      skills: [...input.skills],
      condition: input.condition,
      forecast: [...input.forecast],
      completion: input.completion,
      perfection: input.perfection,
      // The engine setting is passed unchanged; no parallel depth constant.
      lookaheadDepth: input.lookaheadDepth,
      searchConfig: input.searchConfig,
    });
  } catch (error: unknown) {
    gpu = {
      chosen: null,
      comparableScore: Number.NaN,
      path: [],
      nodeCounts: [],
      totalNodes: 0,
      completedLevels: 0,
      stoppedByNodeCap: false,
      timing: { gpuMs: 0, totalMs: 0 },
      notes: {},
      refusal: `исключение device-resident GPU: ${String(error)}`,
    };
  }
  const wallMs = now() - started;
  // The resident kernel reports one GPU window; map it onto the panel's
  // existing timing shape rather than inventing encode/readback measurements.
  const timing: StatefulSearchResult['timing'] = {
    encodeMs: 0,
    gpuMs: gpu.timing.gpuMs,
    readbackMs: 0,
    totalMs: gpu.timing.totalMs,
  };
  const base = { wallMs, timing, notes: gpu.notes };
  const refuse = (reason: string): StatefulRecommendationOutcome => {
    emit({
      kind: 'gpu-resident',
      ok: false,
      reason,
      refusal: gpu.refusal ?? null,
      wallMs,
      timing: { ...timing },
      notes: { ...gpu.notes },
      depth: input.lookaheadDepth,
    });
    log(`resident-wiring ok=false reason=${reason} wallMs=${wallMs.toFixed(1)}`);
    return { ok: false, ...base, reason, refusal: gpu.refusal ?? null };
  };

  if (gpu.refusal) return refuse(`device-resident-поиск отказал: ${gpu.refusal}`);
  const incomplete = incompleteNotes(gpu.notes);
  if (incomplete) return refuse(incomplete);
  // The validated fold needs the level-0/1/2 readbacks; fewer levels is an
  // incomplete result, not a shallower answer to a deep request.
  if (gpu.completedLevels < 3) {
    return refuse(
      `неполный device-resident результат: уровней ${gpu.completedLevels}, нужно 3`,
    );
  }
  if (!gpu.chosen || !Number.isFinite(gpu.comparableScore)) {
    return refuse('device-resident-поиск не выбрал ход');
  }
  const chosen = input.skills.find((skill) => skill.key === gpu.chosen);
  if (!chosen) {
    return refuse(`ход device-resident GPU ${String(gpu.chosen)} не найден в наборе приёмов`);
  }

  const verifyBudgetMs = Math.min(
    input.searchConfig.timeBudgetMs,
    GPU_STATEFUL_VERIFY_BUDGET_MS,
  );
  let result: SearchResult;
  try {
    result = find(
      input.state,
      input.config,
      input.completion,
      input.perfection,
      false,
      input.lookaheadDepth,
      input.condition,
      [...input.forecast],
      { ...input.searchConfig, rootSkillKeys: [chosen.key], timeBudgetMs: verifyBudgetMs },
    );
  } catch (error: unknown) {
    return refuse(`проверка хода device-resident GPU прервалась: ${String(error)}`);
  }

  const status =
    `GPU resident → engine: ход ${chosen.name}, глубина ${input.lookaheadDepth}, ` +
    `уровней ${gpu.completedLevels}, узлов ${gpu.totalNodes}, ${wallMs.toFixed(0)} мс, ` +
    `ядро ${gpu.timing.gpuMs.toFixed(0)} мс`;
  emit({
    kind: 'gpu-resident',
    ok: true,
    engine: 'resident-gpu+engine',
    chosen: chosen.key,
    chosenName: chosen.name,
    score: gpu.comparableScore,
    path: [...gpu.path],
    nodeCounts: [...gpu.nodeCounts],
    wallMs,
    timing: { ...timing },
    notes: { ...gpu.notes },
    depth: input.lookaheadDepth,
    verifyBudgetMs,
    status,
  });
  log(
    `resident-wiring ok=true chosen=${chosen.key} depth=${input.lookaheadDepth} ` +
      `levels=${gpu.completedLevels} nodes=${gpu.totalNodes} wallMs=${wallMs.toFixed(1)} ` +
      `gpuMs=${gpu.timing.gpuMs.toFixed(1)} path=${gpu.path.join('>')}`,
  );
  return {
    ok: true,
    ...base,
    result,
    chosenKey: chosen.key,
    chosenName: chosen.name,
    gpuScore: gpu.comparableScore,
    path: gpu.path,
    order: gpu.path,
    status,
  };
}

/**
 * Single entry point for the mod. Depth <= 2 keeps the existing stateful
 * pipeline untouched; depth > 2 uses the validated resident beam. In both
 * branches a refusal simply leaves the old GPU/JS ladder in place.
 */
export async function tryGpuRecommendation(
  input: StatefulRecommendationInput,
  deps: GpuRecommendationDeps = {},
): Promise<StatefulRecommendationOutcome> {
  try {
    return input.lookaheadDepth > 2
      ? await tryResidentRecommendation(input, deps)
      : await tryStatefulRecommendation(input, deps);
  } catch (error: unknown) {
    // Defensive: both facades are contractually non-throwing.
    return {
      ok: false,
      wallMs: 0,
      timing: { encodeMs: 0, gpuMs: 0, readbackMs: 0, totalMs: 0 },
      notes: {},
      reason: `GPU-маршрутизация прервалась: ${String(error)}`,
      refusal: String(error),
    };
  }
}
