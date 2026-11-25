/**
 * Production device-resident depth search (lookaheadDepth > 2).
 *
 * Port of the validated `scripts/live-depth96-stand.ts` orchestration: same
 * table setup, same validated byParent/branchBest fold, the same re-seed of
 * dropped level-1 slots and the same applyRootSurvival. No runtime WGSL
 * patching, no engine edits, no imports from `scripts/`.
 *
 * Every search setting comes from the mod input (`state`, `config`, `skills`,
 * condition queue, targets, `lookaheadDepth`, `searchConfig`). The constants
 * that describe the compiled kernel (BEAM, DEPTH_MAX, SKILL_CAPACITY, …) are
 * imported from `residentBeam`, never copied here. The stand's verification
 * numbers (depth 96, 5e6 nodes, beam 20, 549 ms) are NOT defaults.
 *
 * On any failure this module returns a loud refusal result; it never throws
 * into the UI and never answers a deep request with a shallow plan.
 */

import {
  BEAM,
  CASES,
  CHILD,
  CONDITION_KINDS,
  DEPTH_MAX,
  GAIN,
  MAX_BRANCHES,
  META,
  SKILL_CAPACITY,
  X,
  createResidentBuffers,
  createResidentPipelines,
  destroyResidentBuffers,
  runOneResidentLevel,
} from './residentBeam';
import type { ResidentLevelResult, ResidentPipelines } from './residentBeam';
import { runResidentDepth } from './residentDepthLoop';
import {
  AVAIL_PARAM,
  AVAIL_SKILL,
  AVAIL_SKILL_FLAGS,
  conditionKindCode,
  conditionNameOfKind,
} from './availability';
import {
  CONDITION_KIND,
  HARMONY_OPS,
  NS_STATE,
  buildBuffCatalog,
  buildStateNodes,
  buildStateSkillRecords,
  conditionIdOf,
} from './records';
import { VERDICT_OUT } from './verdict';
import { SCORE_PARAM, buildScoreParams, buildScoreWords } from './kernelScore';
import type { ScoreConfigLike } from './kernelScore';
import { STYLE } from './scoreInputKernel';
import { STATE_PARAM_WORDS } from './statefulSearch';
import { probeCapability } from './capability';
import { gpuDebugLog } from './debug';
import { __testing } from '../search';
import {
  getConditionTransitionsWithProvider,
  normalizeConditionType,
  normalizeForecastConditionQueue,
  VISIBLE_CONDITION_QUEUE_LENGTH,
} from '../search/conditions';
import { DEFAULT_SEARCH_CONFIG } from '../search/config';
import { getGoalPriorityWeights, SCORING } from '../search/scoring';
import { goalsMet } from '../search/goals';
import { estimateWeightedProgressPerTurn } from '../search/scoringContext';
import type { ScoringContext } from '../search/scoringContext';
import {
  applySkill,
  calculateActionSurvivabilityFloor,
  calculateEffectiveActionCosts,
  calculateSkillGains,
  getAvailableSkills,
  getBuffDiscordantConditions,
  getConditionEffectsForConfig,
  isTerminalState,
} from '../skills';
import {
  hasFatalTechniqueFailure,
  hasGuaranteedSafeStabilizeAction,
  shouldUseSurvivabilityFloorForContinuation,
} from '../search/safety';
import { willAutoFinish, type OutcomeBands } from '../outcome';
import type { OptimizerConfig, SkillDefinition } from '../skills';
import type { ConditionEffect } from '../gameTypes';
import type { CraftingState } from '../state';
import type { CraftingConditionType, SearchConfig } from '../search/types';

/** Everything the GPU path needs; no parallel GPU-only tuning knobs. */
export interface ResidentSearchInput {
  readonly state: CraftingState;
  readonly config: OptimizerConfig;
  readonly skills: readonly SkillDefinition[];
  /** Current condition, exactly as passed to the engine search. */
  readonly condition: string | undefined;
  /** Forecast, exactly as passed to the engine search. */
  readonly forecast: readonly string[];
  readonly completion: number;
  readonly perfection: number;
  /** Engine lookaheadDepth, unchanged from the mod setting. */
  readonly lookaheadDepth: number;
  /** Whole engine search config, including timeBudgetMs, unchanged. */
  readonly searchConfig: Partial<SearchConfig> & { timeBudgetMs: number };
}

export interface ResidentSearchResult {
  /** Engine skill `key`; null on refusal. */
  readonly chosen: string | null;
  /** Comparable score from the validated root fold. */
  readonly comparableScore: number;
  /** Device-derived greedy continuation (root first); not engine-verified. */
  readonly path: readonly string[];
  /** Alive node count per completed level. */
  readonly nodeCounts: readonly number[];
  readonly totalNodes: number;
  readonly completedLevels: number;
  readonly stoppedByNodeCap: boolean;
  readonly timing: {
    readonly gpuMs: number;
    readonly totalMs: number;
  };
  /** Numeric device/settings evidence; no stand defaults are reported. */
  readonly notes: Record<string, number>;
  /** Human-readable reason when the search refused; absent on success. */
  readonly refusal?: string;
}

export interface ResidentSearchDeps {
  /** Isolated tests only: supply a device/pipelines pair without WebGPU. */
  readonly getContext?: () => Promise<ResidentDeviceContext>;
  readonly now?: () => number;
  readonly log?: (...parts: unknown[]) => void;
}

export interface ResidentDeviceContext {
  readonly device: GpuDevice;
  readonly pipelines: ResidentPipelines;
}

const clock = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();

let cachedContext: Promise<ResidentDeviceContext> | null = null;

async function createContext(): Promise<ResidentDeviceContext> {
  const capability = await probeCapability();
  if (!capability.available) throw new Error(capability.reason);
  if (!navigator.gpu) throw new Error('WebGPU недоступен: navigator.gpu исчез');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU недоступен: адаптер не найден');
  const device = await adapter.requestDevice();
  const pipelines = await createResidentPipelines(device);
  const fresh: ResidentDeviceContext = { device, pipelines };
  void device.lost.then(() => {
    // A lost device must never be reused; the next search builds a new one.
    const current = cachedContext;
    if (current !== null) {
      void current.then(
        (value) => {
          if (value === fresh) cachedContext = null;
        },
        () => {
          cachedContext = null;
        },
      );
    }
  });
  return fresh;
}

async function getContext(): Promise<ResidentDeviceContext> {
  if (cachedContext === null) {
    cachedContext = createContext().then(
      (created) => {
        cachedContext = Promise.resolve(created);
        return created;
      },
      (error: unknown) => {
        cachedContext = null;
        throw error;
      },
    );
  }
  return cachedContext;
}

function normalizeSkillKey(value: unknown): string {
  return String(value ?? '').trim().toLowerCase().replace(/\s+/g, '_');
}

function effectsOf(config: OptimizerConfig, condition: string): ConditionEffect[] {
  const table = config.conditionEffectsData as unknown as
    | Record<string, ConditionEffect[]>
    | undefined;
  return table?.[condition] ?? [];
}

function conditionDeltas(config: OptimizerConfig, condition: string): {
  control: number;
  intensity: number;
} {
  let control = 0;
  let intensity = 0;
  for (const effect of getConditionEffectsForConfig(config, condition)) {
    if (effect.kind === 'control' && effect.multiplier !== undefined) {
      control += effect.multiplier;
    }
    if (effect.kind === 'intensity' && effect.multiplier !== undefined) {
      intensity += effect.multiplier;
    }
  }
  return { control, intensity };
}

function combine(reduction: boolean, current: number, contribution: number): number {
  return reduction
    ? 100 - ((100 - Math.min(100, current)) * (100 - Math.min(100, contribution))) / 100
    : ((100 + current) * (100 + contribution)) / 100 - 100;
}

interface ResidentEnv {
  readonly config: OptimizerConfig;
  readonly searchCfg: SearchConfig;
  readonly ctx: ScoringContext;
  readonly targetCompletion: number;
  readonly targetPerfection: number;
  readonly modeCompGoal: number;
  readonly modePerfGoal: number;
  readonly bands: OutcomeBands;
  readonly scoreCfg: ScoreConfigLike;
}

interface SurvivalInfo {
  readonly endsCraft: boolean;
  readonly deviceRisk: number;
  readonly deviceP: number;
  readonly deviceDeath: number;
  readonly hostP: number;
  readonly hostDeath: number;
  readonly rawSum: number;
  readonly deviceScore: number;
  readonly hostScore: number;
}

type RootSurvivalFn = (input: ReadonlyMap<number, number>) => {
  corrected: Map<number, number>;
  rootSurvival: Readonly<Record<string, SurvivalInfo>>;
};

/** `s6a-engine-score.ts` engineScoreFor, ported so src/ never imports scripts/. */
function engineScoreFor(
  state: CraftingState,
  env: ResidentEnv,
  condition: string,
): number {
  const { scoreCfg } = env;
  const willFinish =
    willAutoFinish(state, env.bands) || isTerminalState(state, env.config, condition);
  if (willFinish) {
    return __testing.scoreFinishedOutcome(
      state,
      env.targetCompletion,
      env.targetPerfection,
      scoreCfg.isSublime === true,
      scoreCfg.targetMultiplier ?? 2,
      scoreCfg.maxCompletion,
      scoreCfg.maxPerfection,
      env.ctx,
      scoreCfg.goalPriorityBias ?? 0,
      scoreCfg.overcraftAmbition !== false,
      scoreCfg.perfectionBandGoal ?? 0,
      scoreCfg.completionBandCeiling ?? 0,
    );
  }
  const base = __testing.scoreState(
    state,
    env.targetCompletion,
    env.targetPerfection,
    scoreCfg.isSublime === true,
    scoreCfg.targetMultiplier ?? 2,
    scoreCfg.trainingMode === true,
    scoreCfg.maxCompletion,
    scoreCfg.maxPerfection,
    env.ctx,
    scoreCfg.goalPriorityBias ?? 0,
    scoreCfg.overcraftAmbition !== false,
    scoreCfg.perfectionBandGoal ?? 0,
    scoreCfg.completionBandCeiling ?? 0,
  );
  const compRemaining = Math.max(0, env.modeCompGoal - state.completion);
  const perfRemaining = Math.max(0, env.modePerfGoal - state.perfection);
  const totalRemaining = compRemaining + perfRemaining;
  if (totalRemaining <= 0) return base;
  const compNeedShare = compRemaining / totalRemaining;
  const perfNeedShare = perfRemaining / totalRemaining;
  const { completionWeight, perfectionWeight } = getGoalPriorityWeights(
    compNeedShare,
    perfNeedShare,
    env.modeCompGoal,
    env.modePerfGoal,
    scoreCfg.goalPriorityBias ?? 0,
  );
  const totalPriorityWeight = completionWeight + perfectionWeight;
  const compShare =
    totalPriorityWeight > 0 ? completionWeight / totalPriorityWeight : 0.5;
  const perfShare =
    totalPriorityWeight > 0 ? perfectionWeight / totalPriorityWeight : 0.5;
  const deltas = conditionDeltas(env.config, condition);
  const conditionedPotential =
    compShare * (1 + deltas.intensity) + perfShare * (1 + deltas.control);
  const neutralPotential = compShare + perfShare;
  return (
    base +
    (conditionedPotential - neutralPotential) *
      estimateWeightedProgressPerTurn(compShare, perfShare, env.ctx)
  );
}

/**
 * One device-resident deep search. Refuses rather than throwing on missing
 * WebGPU, a wrong beam width, an unsupported depth, a timeout, device loss or
 * an empty root fold.
 */
export async function runResidentSearch(
  input: ResidentSearchInput,
  deps: ResidentSearchDeps = {},
): Promise<ResidentSearchResult> {
  const now = deps.now ?? clock;
  const log = deps.log ?? gpuDebugLog;
  const started = now();
  const notes: Record<string, number> = {};
  const emptyTiming = (): { gpuMs: number; totalMs: number } => ({
    gpuMs: 0,
    totalMs: now() - started,
  });
  const refuse = (reason: string): ResidentSearchResult => {
    notes.refusals = (notes.refusals ?? 0) + 1;
    log(`resident-wiring refusal=${reason}`);
    return {
      chosen: null,
      comparableScore: Number.NaN,
      path: [],
      nodeCounts: [],
      totalNodes: 0,
      completedLevels: 0,
      stoppedByNodeCap: false,
      timing: emptyTiming(),
      notes,
      refusal: reason,
    };
  };

  try {
    const state = input.state;
    const skills: readonly SkillDefinition[] = [...input.skills];
    const searchCfg: SearchConfig = {
      ...DEFAULT_SEARCH_CONFIG,
      ...input.searchConfig,
      conditionDiscordance:
        input.searchConfig.conditionDiscordance ?? getBuffDiscordantConditions(state.buffs),
    };
    const depth = Number(input.lookaheadDepth);
    notes.requestedDepth = depth;
    if (!Number.isInteger(depth) || depth <= 2 || depth > DEPTH_MAX) {
      return refuse(
        `device-resident beam: неподдерживаемая глубина ${input.lookaheadDepth} ` +
          `(ожидается 3..${DEPTH_MAX})`,
      );
    }
    // No parallel limits: these are the exact engine settings handed in.
    notes.timeBudgetMs = searchCfg.timeBudgetMs;
    notes.maxNodes = searchCfg.maxNodes;
    notes.beamWidth = searchCfg.beamWidth;
    notes.goalPriorityBias = searchCfg.goalPriorityBias;
    notes.overcraftAmbition = searchCfg.overcraftAmbition === false ? 0 : 1;
    if (searchCfg.beamWidth !== BEAM) {
      return refuse(
        `device-resident beam: beamWidth ${searchCfg.beamWidth} требует ядра BEAM=${BEAM}`,
      );
    }

    // `timeBudgetMs` from the mod is the deadline. No stand 549 ms anywhere.
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        reject(
          new Error(
            `WebGPU-таймаут ${searchCfg.timeBudgetMs} мс (прошло ${(now() - started).toFixed(0)} мс)`,
          ),
        );
      }, Math.max(1, Math.floor(searchCfg.timeBudgetMs)));
    });

    const condition = normalizeConditionType(input.condition);
    const queue = normalizeForecastConditionQueue(
      condition,
      [...input.forecast] as CraftingConditionType[],
      state.harmony,
      VISIBLE_CONDITION_QUEUE_LENGTH,
      searchCfg.conditionDiscordance,
    );
    const targets = {
      completion: Number(input.completion),
      perfection: Number(input.perfection),
    };
    const discordance = searchCfg.conditionDiscordance ?? 0;
    const config = Object.assign({}, input.config, {
      currentCondition: condition,
    }) as OptimizerConfig & { currentCondition: string };

    const buildEnv = (): ResidentEnv => {
      const ctx = __testing.buildScoringContext(config, state, condition);
      const bands = __testing.deriveBandGoals(
        targets.completion,
        targets.perfection,
        config.isSublimeCraft === true,
        config.maxCompletion,
        config.maxPerfection,
        searchCfg.perfectionBandGoal,
        searchCfg.completionBandCeiling,
      );
      const scoreCfg = {
        targetCompletion: targets.completion,
        targetPerfection: targets.perfection,
        isSublime: config.isSublimeCraft === true,
        trainingMode: config.trainingMode === true,
        targetMultiplier: config.targetMultiplier || 2,
        maxCompletion: config.maxCompletion,
        maxPerfection: config.maxPerfection,
        goalPriorityBias: searchCfg.goalPriorityBias,
        overcraftAmbition: searchCfg.overcraftAmbition,
        perfectionBandGoal: searchCfg.perfectionBandGoal,
        completionBandCeiling: searchCfg.completionBandCeiling,
        ctx,
      } as unknown as ScoreConfigLike;
      return {
        config,
        searchCfg,
        ctx,
        targetCompletion: targets.completion,
        targetPerfection: targets.perfection,
        modeCompGoal: bands.effectiveCompGoal,
        modePerfGoal: bands.effectivePerfGoal,
        bands: bands.bands,
        scoreCfg,
      };
    };
    const env = buildEnv();

    const work = async (): Promise<ResidentSearchResult> => {
      const ctx = await (deps.getContext ?? getContext)();
      const device = ctx.device;
      const pipelines = ctx.pipelines;
      let lost = false;
      void device.lost.then(() => {
        lost = true;
      });
      const checkAlive = (): void => {
        if (lost) throw new Error('WebGPU-устройство потеряно');
        if (now() - started > searchCfg.timeBudgetMs) {
          throw new Error(
            `WebGPU-таймаут ${searchCfg.timeBudgetMs} мс (прошло ${(now() - started).toFixed(0)} мс)`,
          );
        }
      };

      // --- root tables (stand :296-521) ------------------------------------
      const catalog = buildBuffCatalog(state, config, skills as SkillDefinition[]);
      const skillSet = buildStateSkillRecords(state, config, skills as SkillDefinition[], catalog);
      const rootNode = Float32Array.from(
        buildStateNodes(state, config, skills as SkillDefinition[], 1, catalog).nodes,
      );
      const nodes = new Float32Array(BEAM * rootNode.length);
      const meta = new Float32Array(BEAM * META.WORDS);
      const kind = conditionKindCode(condition);
      for (let i = 0; i < BEAM; i++) {
        if (i < Math.min(skills.length, BEAM)) {
          nodes.set(rootNode, i * rootNode.length);
          const base = i * rootNode.length;
          nodes[base + 30] = conditionIdOf(condition);
          for (let q = 0; q < queue.length; q++) {
            nodes[base + 31 + q] = CONDITION_KIND[queue[q] as keyof typeof CONDITION_KIND] ??
              CONDITION_KIND.neutral;
          }
        }
        const m = i * META.WORDS;
        meta[m + META.probability] = 1;
        meta[m + META.rootId] = i;
        meta[m + META.parentId] = -1;
        meta[m + META.depth] = 0;
        meta[m + META.alive] = i < Math.min(skills.length, BEAM) ? 1 : 0;
        meta[m + META.conditionKind] = kind;
      }

      const condEffects = new Float32Array(CONDITION_KINDS * 6);
      const conditionHashes = new Float32Array(CONDITION_KINDS);
      for (let k = 0; k < CONDITION_KINDS; k++) {
        conditionHashes[k] = conditionIdOf(conditionNameOfKind(k));
      }
      const table = new Float32Array(CONDITION_KINDS * 2 * SCORE_PARAM.WORDS);
      for (let k = 0; k < CONDITION_KINDS; k++) {
        const name = conditionNameOfKind(k);
        const effects = getConditionEffectsForConfig(config, name) as Array<{
          kind?: string;
          multiplier?: number;
        }>;
        let control = 0;
        let intensity = 0;
        let poolRed = 0;
        let poolPen = 0;
        let stabRed = 0;
        let stabPen = 0;
        for (const effect of effects) {
          if (effect?.kind === 'control') control += Number(effect.multiplier ?? 0);
          if (effect?.kind === 'intensity') intensity += Number(effect.multiplier ?? 0);
          const mult = Number(effect?.multiplier);
          if (!Number.isFinite(mult) || mult === 1) continue;
          if (effect?.kind === 'pool') {
            if (mult < 1) poolRed = combine(true, poolRed, (1 - mult) * 100);
            else poolPen = combine(false, poolPen, (mult - 1) * 100);
          } else if (effect?.kind === 'stability') {
            if (mult < 1) stabRed = combine(true, stabRed, (1 - mult) * 100);
            else stabPen = combine(false, stabPen, (mult - 1) * 100);
          }
        }
        condEffects.set([control, intensity, poolRed, poolPen, stabRed, stabPen], k * 6);
        const deltas = conditionDeltas(config, name);
        for (let terminal = 0; terminal < 2; terminal++) {
          const block = buildScoreParams({
            targetCompletion: targets.completion,
            targetPerfection: targets.perfection,
            isSublime: config.isSublimeCraft === true,
            trainingMode: config.trainingMode === true,
            targetMultiplier: config.targetMultiplier || 2,
            maxCompletion: config.maxCompletion,
            maxPerfection: config.maxPerfection,
            goalPriorityBias: searchCfg.goalPriorityBias,
            overcraftAmbition: searchCfg.overcraftAmbition,
            perfectionBandGoal: searchCfg.perfectionBandGoal,
            completionBandCeiling: searchCfg.completionBandCeiling,
            ctx: env.ctx,
            conditionControlDelta: deltas.control,
            conditionIntensityDelta: deltas.intensity,
            terminalBits: terminal,
          });
          table.set(block, (k * 2 + terminal) * block.length);
        }
      }

      const gains = new Float32Array(CHILD * CONDITION_KINDS * 7);
      const availSkill = new Float32Array(CHILD * AVAIL_SKILL.WORDS);
      const availSkillPerSkill = new Float32Array(SKILL_CAPACITY * AVAIL_SKILL.WORDS);
      const availParamBase = new Float32Array(SKILL_CAPACITY * 9);
      for (let i = 0; i < skills.length && i < SKILL_CAPACITY; i++) {
        const skill = skills[i] as SkillDefinition;
        const b = i * AVAIL_SKILL.WORDS;
        const req = skill.buffRequirement;
        const cost = skill.buffCost;
        availSkillPerSkill[b + AVAIL_SKILL.dropletCost] = Number(skill.dropletCost ?? 0);
        availSkillPerSkill[b + AVAIL_SKILL.toxicityCost] = Number(skill.toxicityCost ?? 0);
        availSkillPerSkill[b + AVAIL_SKILL.conditionKind] = skill.conditionRequirement
          ? conditionKindCode(skill.conditionRequirement)
          : 0;
        availSkillPerSkill[b + AVAIL_SKILL.cooldownSkillIndex] =
          skill.actionKind === 'item' ? -1 : i;
        availSkillPerSkill[b + AVAIL_SKILL.buffRequirementId] = req
          ? catalog.ids.get(normalizeSkillKey(req.buffName)) ?? -1
          : -1;
        availSkillPerSkill[b + AVAIL_SKILL.buffRequirementAmount] = Number(req?.amount ?? 0);
        availSkillPerSkill[b + AVAIL_SKILL.buffCostId] = cost
          ? catalog.ids.get(normalizeSkillKey(cost.buffName)) ?? -1
          : -1;
        availSkillPerSkill[b + AVAIL_SKILL.buffCostRequired] = cost
          ? cost.consumeAll === true ? 1 : Number(cost.amount ?? 0)
          : 0;
        availSkillPerSkill[b + AVAIL_SKILL.isItem] = skill.actionKind === 'item' ? 1 : 0;
        availSkillPerSkill[b + AVAIL_SKILL.reagentOnlyAtStepZero] =
          skill.reagentOnlyAtStepZero === true ? 1 : 0;
        availSkillPerSkill[b + AVAIL_SKILL.flags] = skill.conditionRequirement
          ? AVAIL_SKILL_FLAGS.hasConditionRequirement
          : 0;
        const itemKey = normalizeSkillKey(skill.itemName || skill.key);
        const p = i * AVAIL_PARAM.WORDS;
        availParamBase[p + AVAIL_PARAM.qiDroplets] = Number(state.qiDroplets ?? 0);
        availParamBase[p + AVAIL_PARAM.itemRemaining] = Number(state.items.get(itemKey) ?? 0);
        availParamBase[p + AVAIL_PARAM.consumedPillsThisTurn] = Number(state.consumedPillsThisTurn ?? 0);
        availParamBase[p + AVAIL_PARAM.pillsPerRound] = Math.max(1, Math.floor(config.pillsPerRound || 1));
        availParamBase[p + AVAIL_PARAM.nativePrecheck] = 1;
        availParamBase[p + AVAIL_PARAM.nativePrecheckPresent] = 0;
        availParamBase[p + AVAIL_PARAM.maxToxicity] = Number(config.maxToxicity || 0);
        availParamBase[p + AVAIL_PARAM.effectiveQiCost] = calculateEffectiveActionCosts(
          state, skill, config.minStability,
          getConditionEffectsForConfig(config, condition), config,
        ).qiCost;
        for (let k = 0; k < CONDITION_KINDS; k++) {
          const name = conditionNameOfKind(k);
          const eff = effectsOf(config, name);
          const expected = calculateSkillGains(state, skill, config, eff, { currentCondition: name });
          const success = calculateSkillGains(state, skill, config, eff, {
            successChance: 1, currentCondition: name,
          });
          const gb = (i * CONDITION_KINDS + k) * 7;
          gains[gb + GAIN.expectedCompletion] = expected.completion;
          gains[gb + GAIN.expectedPerfection] = expected.perfection;
          gains[gb + GAIN.successCompletion] = success.completion;
          gains[gb + GAIN.successPerfection] = success.perfection;
          gains[gb + GAIN.survival] = 1;
        }
      }
      for (let c = 0; c < CHILD; c++) {
        const s = c % SKILL_CAPACITY;
        availSkill.set(
          availSkillPerSkill.subarray(s * AVAIL_SKILL.WORDS, (s + 1) * AVAIL_SKILL.WORDS),
          c * AVAIL_SKILL.WORDS,
        );
      }

      const paramsBase = new Float32Array(STATE_PARAM_WORDS);
      paramsBase[4] = Number(config.maxQi ?? 0);
      paramsBase[5] = Number(state.initialMaxStability);
      paramsBase[6] = HARMONY_OPS[config.craftingType ?? ''] ?? 0;
      paramsBase[23] = targets.completion;
      paramsBase[24] = targets.perfection;
      paramsBase[25] = Number(config.maxCompletion ?? Number.POSITIVE_INFINITY);
      paramsBase[26] = Number(config.maxPerfection ?? Number.POSITIVE_INFINITY);
      paramsBase[28] = config.isSublimeCraft === true ? 1 : 0;

      const styleFrom = (sw: Float32Array): Float32Array => {
        const s = new Float32Array(X.WORDS);
        s[STYLE.kind] = sw[12];
        s[STYLE.forgeHeat] = sw[13];
        s[STYLE.forgeLastBuffedHeat] = sw[14];
        s[STYLE.inscriptionStacks] = sw[15];
        s[STYLE.inscribedRemaining] = sw[16];
        s[STYLE.alchemicalCharges] = sw[17];
        s[STYLE.alchemicalModsFinite] = sw[18];
        for (let i = 0; i < 8; i++) s[STYLE.recommended + i] = sw[20 + i];
        s[STYLE.resonanceStrength] = sw[28];
        s[STYLE.resonanceType] = sw[29];
        s[STYLE.pendingResonanceType] = sw[30];
        s[STYLE.pendingCount] = sw[31];
        s[STYLE.cadenceChain] = sw[32];
        s[STYLE.alchemicalControlMult] = sw[33];
        s[STYLE.alchemicalIntensityMult] = sw[34];
        s[STYLE.alchemicalCritBonus] = sw[35];
        s[STYLE.alchemicalSuccessBonus] = sw[36];
        s[STYLE.alchemicalQiReduction] = sw[37];
        s[STYLE.alchemicalStabReduction] = sw[38];
        s[X.discordance] = discordance;
        s[X.branching] = 1;
        s[X.branchLimit] = searchCfg.conditionBranchLimit;
        s[X.minProbability] = searchCfg.conditionBranchMinProbability;
        s[X.modeCompGoal] = env.modeCompGoal;
        s[X.modePerfGoal] = env.modePerfGoal;
        s[X.targetCompletion] = targets.completion;
        s[X.targetPerfection] = targets.perfection;
        s[X.initialMaxStability] = Number(state.initialMaxStability);
        s[X.nearDeath] = SCORING.NEAR_DEATH_STABILITY;
        s[X.hasSafe] = 0;
        return s;
      };
      const style = styleFrom(buildScoreWordsFor(state, condition));
      const rootDisplays: Array<CraftingState | undefined> = skills.map((skill) =>
        applySkill(
          state, skill, config, effectsOf(config, condition),
          targets.completion, condition, targets.perfection,
        ) ?? undefined,
      );
      const perSkillStyles = skills.map((_skill, i) => {
        const display = rootDisplays[i];
        return display
          ? styleFrom(buildScoreWordsFor(display, condition))
          : style;
      });
      const styleCases = new Float32Array(CASES * STYLE.WORDS);
      for (let i = 0; i < CASES; i++) {
        const src = i < CHILD * MAX_BRANCHES
          ? perSkillStyles[Math.floor(i / MAX_BRANCHES) % SKILL_CAPACITY] ?? style
          : style;
        styleCases.set(src.subarray(0, STYLE.WORDS), i * STYLE.WORDS);
      }

      const bufferSpec = {
        catalog: catalog.records,
        skillRecords: skillSet.records,
        availSkill,
        availParamBase,
        stateParamsBase: paramsBase,
        condEffects,
        conditionHashes,
        gains,
        scoreParamTable: table,
        style,
        harmonyIds: new Float32Array(8).fill(-1),
        styleCases,
        rootNodes: nodes,
        rootMeta: meta,
      };
      const bufferStart = now();

      // --- validated depth-3 fold on live readbacks (stand :523-861) ---------
      const dumpBuffers = createResidentBuffers(device, bufferSpec);
      let fold: {
        readonly byParent: ReadonlyMap<number, number>;
        readonly branchBest: ReadonlyMap<number, { value: number; skill: number; slot: number }>;
      } | null = null;
      let v1: Float32Array = new Float32Array(SKILL_CAPACITY);
      let deviceVerdict: Float32Array = new Float32Array(
        SKILL_CAPACITY * MAX_BRANCHES * VERDICT_OUT.WORDS,
      );
      let deviceDeaths: Float32Array = new Float32Array(SKILL_CAPACITY);
      let rootSurvivalInfo: Readonly<Record<string, SurvivalInfo>> = {};
      let rootSurvivalApply: RootSurvivalFn | null = null;
      let bestChain: readonly ResidentLevelResult[] | null = null;
      let bestRoot = -1;
      try {
        checkAlive();
        const l1 = await runOneResidentLevel(device, pipelines, dumpBuffers, {
          depth, level: 0,
          dumpBuffers: [
            { buffer: 'verdictOut', offsetWords: 0,
              words: SKILL_CAPACITY * MAX_BRANCHES * VERDICT_OUT.WORDS },
            { buffer: 'scores', offsetWords: CHILD * MAX_BRANCHES, words: SKILL_CAPACITY },
          ],
        });
        v1 = l1.values;
        const verdictWords = SKILL_CAPACITY * MAX_BRANCHES * VERDICT_OUT.WORDS;
        deviceVerdict = l1.dump ? l1.dump.subarray(0, verdictWords) : deviceVerdict;
        deviceDeaths = l1.dump ? l1.dump.subarray(verdictWords) : deviceDeaths;

        const m1 = l1.meta;
        const hostAvail2 = new Map<number, Set<number>>();
        let lastGains2: Float32Array | null = null;
        const gainsForLevel2 = (level1: ResidentLevelResult): Float32Array => {
          const gains2 = new Float32Array(CHILD * CONDITION_KINDS * 7);
          for (let slot = 0; slot < BEAM; slot++) {
            const mb = slot * META.WORDS;
            if (level1.meta[mb + META.alive] < 0.5) continue;
            const rootSkill = Math.round(level1.meta[mb + META.skill]);
            let display = rootDisplays[rootSkill];
            if (display === undefined) {
              const skillDef = skills[rootSkill];
              if (!skillDef) continue;
              display = applySkill(
                state, skillDef, config, effectsOf(config, condition),
                targets.completion, condition, targets.perfection,
              ) ?? undefined;
              rootDisplays[rootSkill] = display;
            }
            if (!display) continue;
            const slotCond = conditionNameOfKind(
              Math.round(level1.meta[mb + META.conditionKind]),
            );
            hostAvail2.set(slot, new Set(
              getAvailableSkillIndices(display, config, slotCond, skills),
            ));
            for (let k = 0; k < CONDITION_KINDS; k++) {
              const name = conditionNameOfKind(k);
              const eff = effectsOf(config, name);
              for (let s = 0; s < skills.length && s < SKILL_CAPACITY; s++) {
                const expected = calculateSkillGains(display, skills[s] as SkillDefinition, config, eff, {
                  currentCondition: name,
                });
                const success = calculateSkillGains(display, skills[s] as SkillDefinition, config, eff, {
                  successChance: 1, currentCondition: name,
                });
                const gb = ((slot * SKILL_CAPACITY + s) * CONDITION_KINDS + k) * 7;
                gains2[gb + GAIN.expectedCompletion] = expected.completion;
                gains2[gb + GAIN.expectedPerfection] = expected.perfection;
                gains2[gb + GAIN.successCompletion] = success.completion;
                gains2[gb + GAIN.successPerfection] = success.perfection;
                gains2[gb + GAIN.survival] = 1;
              }
            }
          }
          lastGains2 = gains2;
          return gains2;
        };
        checkAlive();
        device.queue.writeBuffer(dumpBuffers.gains, 0, gainsForLevel2(l1));
        const l2 = await runOneResidentLevel(device, pipelines, dumpBuffers, {
          depth, level: 1,
        });
        const m2 = l2.meta;
        const branchBest = new Map<number, { value: number; skill: number; slot: number }>();
        for (let i = 0; i < BEAM; i++) {
          const mb = i * META.WORDS;
          if (m1[mb + META.alive] < 0.5) continue;
          const p1 = m1[mb + META.probability];
          let best = -Infinity;
          let bestSkill = -1;
          for (let s = 0; s < SKILL_CAPACITY; s++) {
            if (!(hostAvail2.get(i)?.has(s))) continue;
            const value = l2.values[i * SKILL_CAPACITY + s];
            if (Number.isFinite(value) && value > best) { best = value; bestSkill = s; }
          }
          if (bestSkill < 0 || !(p1 > 0)) continue;
          branchBest.set(i, { value: best, skill: bestSkill, slot: -1 });
        }
        for (let j = 0; j < BEAM; j++) {
          const mb = j * META.WORDS;
          if (m2[mb + META.alive] < 0.5) continue;
          const best = branchBest.get(Math.round(m2[mb + META.parentId]));
          if (best && best.slot < 0) best.slot = j;
        }

        const level3InputsFor = (meta2: Float32Array): {
          gains: Float32Array;
          avail: Map<number, Set<number>>;
        } => {
          const gains = new Float32Array(CHILD * CONDITION_KINDS * 7);
          const avail = new Map<number, Set<number>>();
          for (let slot = 0; slot < BEAM; slot++) {
            const mb = slot * META.WORDS;
            if (meta2[mb + META.alive] < 0.5) continue;
            const parent = Math.round(meta2[mb + META.parentId]);
            const rootSkill = Math.round(m1[parent * META.WORDS + META.skill]);
            const display1 = rootDisplays[rootSkill];
            const skill2 = skills[Math.round(meta2[mb + META.skill])];
            if (!display1 || !skill2) continue;
            const cond1 = conditionNameOfKind(Math.round(m1[parent * META.WORDS + META.conditionKind]));
            const display2 = applySkill(
              display1, skill2, config, effectsOf(config, cond1),
              targets.completion, cond1, targets.perfection,
            );
            if (!display2) continue;
            const cond2 = conditionNameOfKind(Math.round(meta2[mb + META.conditionKind]));
            avail.set(slot, new Set(getAvailableSkillIndices(display2, config, cond2, skills)));
            for (let k = 0; k < CONDITION_KINDS; k++) {
              const name = conditionNameOfKind(k);
              const eff = effectsOf(config, name);
              for (let s = 0; s < skills.length && s < SKILL_CAPACITY; s++) {
                const ex = calculateSkillGains(display2, skills[s] as SkillDefinition, config, eff, {
                  currentCondition: name,
                });
                const su = calculateSkillGains(display2, skills[s] as SkillDefinition, config, eff, {
                  successChance: 1, currentCondition: name,
                });
                const gb = ((slot * SKILL_CAPACITY + s) * CONDITION_KINDS + k) * 7;
                gains[gb + GAIN.expectedCompletion] = ex.completion;
                gains[gb + GAIN.expectedPerfection] = ex.perfection;
                gains[gb + GAIN.successCompletion] = su.completion;
                gains[gb + GAIN.successPerfection] = su.perfection;
                gains[gb + GAIN.survival] = 1;
              }
            }
          }
          return { gains, avail };
        };
        const foldLevel3 = (
          meta2: Float32Array,
          avail: ReadonlyMap<number, ReadonlySet<number>>,
          values3: Float32Array,
          target: Map<number, number>,
        ): number => {
          let added = 0;
          for (let slot = 0; slot < BEAM; slot++) {
            const mb = slot * META.WORDS;
            if (meta2[mb + META.alive] < 0.5) continue;
            const parent = Math.round(meta2[mb + META.parentId]);
            const p2 = meta2[mb + META.probability];
            let best = -Infinity;
            (avail.get(slot) ?? new Set<number>()).forEach((s) => {
              const value = values3[slot * SKILL_CAPACITY + s];
              if (Number.isFinite(value) && value > best) best = value;
            });
            if (!(best > -Infinity) || !(p2 > 0)) continue;
            target.set(parent, (target.get(parent) ?? 0) + p2 * best);
            added++;
          }
          return added;
        };
        const byParent = new Map<number, number>();
        const originalInputs = level3InputsFor(m2);
        checkAlive();
        device.queue.writeBuffer(dumpBuffers.gains, 0, originalInputs.gains);
        const level3 = await runOneResidentLevel(device, pipelines, dumpBuffers, {
          depth, level: 2,
        });
        foldLevel3(m2, originalInputs.avail, level3.values, byParent);

        const availParamsForSlot = (display: CraftingState, cond: string): Float32Array => {
          const out = availParamBase.slice();
          const effects = getConditionEffectsForConfig(config, cond);
          for (let s = 0; s < skills.length && s < SKILL_CAPACITY; s++) {
            const skill = skills[s] as SkillDefinition;
            const p = s * AVAIL_PARAM.WORDS;
            const itemKey = normalizeSkillKey(skill.itemName || skill.key);
            out[p + AVAIL_PARAM.qiDroplets] = Number(display.qiDroplets ?? 0);
            out[p + AVAIL_PARAM.itemRemaining] = Number(display.items.get(itemKey) ?? 0);
            out[p + AVAIL_PARAM.consumedPillsThisTurn] = Number(display.consumedPillsThisTurn ?? 0);
            out[p + AVAIL_PARAM.effectiveQiCost] = calculateEffectiveActionCosts(
              display, skill, config.minStability, effects, config,
            ).qiCost;
          }
          return out;
        };
        const dropped: number[] = [];
        for (let i = 0; i < BEAM; i++) {
          const mb = i * META.WORDS;
          if (m1[mb + META.alive] < 0.5 || !(m1[mb + META.probability] > 0)) continue;
          if (!byParent.has(i)) dropped.push(i);
        }
        const emptyMeta = new Float32Array(BEAM * META.WORDS);
        const emptyNodes = new Float32Array(BEAM * NS_STATE.WORDS);
        let reseededSlots = 0;
        let reseededBranches = 0;
        for (const seedSlot of dropped) {
          checkAlive();
          const seedMb = seedSlot * META.WORDS;
          const seedSkill = Math.round(m1[seedMb + META.skill]);
          const seedDisplay = rootDisplays[seedSkill] ?? (skills[seedSkill]
            ? applySkill(
              state, skills[seedSkill] as SkillDefinition, config, effectsOf(config, condition),
              targets.completion, condition, targets.perfection,
            ) ?? undefined
            : undefined);
          const seedCond = conditionNameOfKind(Math.round(m1[seedMb + META.conditionKind]));
          if (seedDisplay) {
            device.queue.writeBuffer(dumpBuffers.availParamBase, 0, availParamsForSlot(seedDisplay, seedCond));
          }
          if (lastGains2) device.queue.writeBuffer(dumpBuffers.gains, 0, lastGains2);
          const seedMeta = emptyMeta.slice();
          const seedNodes = emptyNodes.slice();
          seedMeta.set(m1.subarray(seedMb, seedMb + META.WORDS), seedMb);
          seedNodes.set(
            l1.nodes.subarray(seedSlot * NS_STATE.WORDS, (seedSlot + 1) * NS_STATE.WORDS),
            seedSlot * NS_STATE.WORDS,
          );
          device.queue.writeBuffer(dumpBuffers.nodesB, 0, seedNodes);
          device.queue.writeBuffer(dumpBuffers.metaB, 0, seedMeta);
          const seededLevel2 = await runOneResidentLevel(device, pipelines, dumpBuffers, {
            depth, level: 1,
          });
          const seededInputs = level3InputsFor(seededLevel2.meta);
          device.queue.writeBuffer(dumpBuffers.gains, 0, seededInputs.gains);
          const seededLevel3 = await runOneResidentLevel(device, pipelines, dumpBuffers, {
            depth, level: 2,
          });
          reseededBranches += foldLevel3(
            seededLevel2.meta, seededInputs.avail, seededLevel3.values, byParent,
          );
          reseededSlots++;
        }
        fold = { byParent, branchBest };
        notes.reseedSlots = reseededSlots;
        notes.reseedBranches = reseededBranches;

        const applyRootSurvival = (
          foldInput: ReadonlyMap<number, number>,
        ): { corrected: Map<number, number>; rootSurvival: typeof rootSurvivalInfo } => {
          const corrected = new Map(foldInput);
          const rootSurvival: Record<string, {
            endsCraft: boolean; deviceRisk: number; deviceP: number; deviceDeath: number;
            hostP: number; hostDeath: number; rawSum: number; deviceScore: number; hostScore: number;
          }> = {};
          corrected.forEach((rawSum, rootMove) => {
            const skillDef = skills[rootMove];
            if (!skillDef) return;
            const display = rootDisplays[rootMove];
            if (!display) return;
            const endsCraft = willAutoFinish(display, env.bands);
            if (endsCraft) {
              corrected.set(rootMove, v1[rootMove]);
              return;
            }
            const effects = getConditionEffectsForConfig(config, condition);
            const floor = calculateActionSurvivabilityFloor(
              state, skillDef, config, effects, condition, display,
            );
            const goalsAfter = goalsMet(display, env.modeCompGoal, env.modePerfGoal);
            const baseSecured = goalsMet(display, env.targetCompletion, env.targetPerfection);
            const gate = Math.max(SCORING.NEAR_DEATH_STABILITY, Math.ceil(env.ctx.avgStabilityCostPerTurn));
            const hasSafe =
              state.stability <= gate &&
              hasGuaranteedSafeStabilizeAction(
                state, [...skills], config, effects, condition,
              );
            const minSafe = hasSafe ? Math.max(1, env.ctx.avgStabilityCostPerTurn) : 0;
            const requiresProbabilisticSurvival =
              hasFatalTechniqueFailure(
                state, skillDef, config, effects, condition,
                env.targetCompletion, env.targetPerfection,
              ) ||
              shouldUseSurvivabilityFloorForContinuation({
                currentStability: state.stability,
                displayStability: display.stability,
                floorStability: floor?.stability ?? display.stability,
                goalsMetAfterAction: goalsAfter,
                baseSuccessSecuredAfterAction: baseSecured,
                hasGuaranteedSafeStabilize: hasSafe,
                minimumGuaranteedContinuationStability: minSafe,
              });
            const hostP = floor?.survivalProbability ?? 1;
            const gatePasses = !(
              requiresProbabilisticSurvival || endsCraft || !floor ||
              floor.stability > 0 || !(hostP > 0 && hostP < 1)
            );
            let hostDeath = Number.NaN;
            let hostScore = rawSum;
            if (gatePasses) {
              const transitions = getConditionTransitionsWithProvider(
                condition, queue, display.conditionGenerationHarmony ?? display.harmony,
                env.searchCfg,
              );
              let unweighted1 = 0;
              for (const transition of transitions) {
                unweighted1 += transition.probability *
                  engineScoreFor(display, env, transition.nextCondition);
              }
              hostDeath = (v1[rootMove] - hostP * unweighted1) / (1 - hostP);
              hostScore = hostP * rawSum + (1 - hostP) * hostDeath;
            }
            let deviceRisk = 0;
            let deviceP = 0;
            const edgeBase = rootMove * MAX_BRANCHES * VERDICT_OUT.WORDS;
            for (let b = 0; b < MAX_BRANCHES; b++) {
              const vb = edgeBase + b * VERDICT_OUT.WORDS;
              if (deviceVerdict[vb + VERDICT_OUT.uncollapsedRisk] >= 0.5) {
                deviceRisk = 1;
                deviceP = deviceVerdict[vb + VERDICT_OUT.blendProbability];
                break;
              }
            }
            const deviceDeath = deviceDeaths[rootMove];
            const deviceScore =
              gatePasses && deviceRisk >= 0.5 && deviceDeath >= 0
                ? deviceP * rawSum + (1 - deviceP) * deviceDeath
                : rawSum;
            corrected.set(rootMove, deviceScore);
            rootSurvival[skillSet.names[rootMove] ?? `skill${rootMove}`] = {
              endsCraft, deviceRisk, deviceP, deviceDeath,
              hostP, hostDeath, rawSum, deviceScore, hostScore,
            };
          });
          return { corrected, rootSurvival };
        };
        // The validated correction is applied to the FINAL folded root sums
        // inside `rankRoots`, exactly as in the stand (:1241).
        rootSurvivalApply = applyRootSurvival;
      } finally {
        destroyResidentBuffers(dumpBuffers);
      }

      // --- device-resident loop (stand :863-1251) ----------------------------
      const loopBuffers = createResidentBuffers(device, bufferSpec);
      const slotStates: Array<Array<CraftingState | undefined>> = [];
      const slotConds: string[][] = [];
      const slotAvail = new Map<number, Map<number, Set<number>>>();
      let cacheLevel = -1;
      let cache: {
        readonly gains: Float32Array;
        readonly availParamBase: Float32Array;
        readonly style: Float32Array;
        readonly styleCases: Float32Array;
      } | null = null;
      const availParamsFor = (display: CraftingState, cond: string): Float32Array => {
        const out = availParamBase.slice();
        const effects = getConditionEffectsForConfig(config, cond);
        for (let s = 0; s < skills.length && s < SKILL_CAPACITY; s++) {
          const skill = skills[s] as SkillDefinition;
          const p = s * AVAIL_PARAM.WORDS;
          const itemKey = normalizeSkillKey(skill.itemName || skill.key);
          out[p + AVAIL_PARAM.qiDroplets] = Number(display.qiDroplets ?? 0);
          out[p + AVAIL_PARAM.itemRemaining] = Number(display.items.get(itemKey) ?? 0);
          out[p + AVAIL_PARAM.consumedPillsThisTurn] = Number(display.consumedPillsThisTurn ?? 0);
          out[p + AVAIL_PARAM.effectiveQiCost] = calculateEffectiveActionCosts(
            display, skill, config.minStability, effects, config,
          ).qiCost;
        }
        return out;
      };
      const tablesFor = (level: number, prev: ResidentLevelResult) => {
        if (cacheLevel === level && cache) return cache;
        const states: Array<CraftingState | undefined> = new Array(BEAM).fill(undefined);
        const conds: string[] = new Array(BEAM).fill(condition);
        for (let slot = 0; slot < BEAM; slot++) {
          const mb = slot * META.WORDS;
          if (prev.meta[mb + META.alive] < 0.5) continue;
          const cond = conditionNameOfKind(Math.round(prev.meta[mb + META.conditionKind]));
          conds[slot] = cond;
          if (level === 1) {
            states[slot] = rootDisplays[Math.round(prev.meta[mb + META.skill])] ?? undefined;
          } else {
            const parent = Math.round(prev.meta[mb + META.parentId]);
            const parentState = slotStates[level - 2]?.[parent];
            const parentCond = slotConds[level - 2]?.[parent];
            const skill = skills[Math.round(prev.meta[mb + META.skill])];
            if (!parentState || !parentCond || !skill) continue;
            states[slot] = applySkill(
              parentState, skill, config, effectsOf(config, parentCond),
              targets.completion, parentCond, targets.perfection,
            ) ?? undefined;
          }
        }
        slotStates[level - 1] = states;
        slotConds[level - 1] = conds;
        const gainsNext = new Float32Array(CHILD * CONDITION_KINDS * 7);
        const avail = new Map<number, Set<number>>();
        let representative = -1;
        for (let slot = 0; slot < BEAM; slot++) {
          const display = states[slot];
          if (!display) continue;
          if (representative < 0 || prev.meta[slot * META.WORDS + META.probability] >
              prev.meta[representative * META.WORDS + META.probability]) representative = slot;
          avail.set(slot, new Set(getAvailableSkillIndices(display, config, conds[slot], skills)));
          for (let k = 0; k < CONDITION_KINDS; k++) {
            const name = conditionNameOfKind(k);
            const eff = effectsOf(config, name);
            for (let s = 0; s < skills.length && s < SKILL_CAPACITY; s++) {
              const skill = skills[s] as SkillDefinition;
              const ex = calculateSkillGains(display, skill, config, eff, { currentCondition: name });
              const su = calculateSkillGains(display, skill, config, eff, {
                successChance: 1, currentCondition: name,
              });
              const gb = ((slot * SKILL_CAPACITY + s) * CONDITION_KINDS + k) * 7;
              gainsNext[gb + GAIN.expectedCompletion] = ex.completion;
              gainsNext[gb + GAIN.expectedPerfection] = ex.perfection;
              gainsNext[gb + GAIN.successCompletion] = su.completion;
              gainsNext[gb + GAIN.successPerfection] = su.perfection;
              gainsNext[gb + GAIN.survival] = 1;
            }
          }
        }
        slotAvail.set(level - 1, avail);
        const display = representative >= 0 ? states[representative] : undefined;
        const repCond = representative >= 0 ? conds[representative] : condition;
        const baseStyle = display
          ? styleFrom(buildScoreWordsFor(display, repCond))
          : style;
        const perSkill = skills.map((skill) => {
          const d = display
            ? applySkill(display, skill, config, effectsOf(config, repCond),
              targets.completion, repCond, targets.perfection)
            : undefined;
          return d ? styleFrom(buildScoreWordsFor(d, repCond)) : baseStyle;
        });
        const casesNext = new Float32Array(CASES * STYLE.WORDS);
        for (let c = 0; c < CASES; c++) {
          const src = c < CHILD * MAX_BRANCHES
            ? perSkill[Math.floor(c / MAX_BRANCHES) % SKILL_CAPACITY] ?? baseStyle
            : baseStyle;
          casesNext.set(src.subarray(0, STYLE.WORDS), c * STYLE.WORDS);
        }
        cache = {
          gains: gainsNext,
          availParamBase: display ? availParamsFor(display, repCond) : availParamBase.slice(),
          style: baseStyle,
          styleCases: casesNext,
        };
        cacheLevel = level;
        return cache;
      };

      let loop: Awaited<ReturnType<typeof runResidentDepth>>;
      try {
        loop = await runResidentDepth(device, pipelines, loopBuffers, {
          depth,
          maxNodes: searchCfg.maxNodes,
          beamWidth: searchCfg.beamWidth,
          recompute: {
            gains: (level, prev) => { checkAlive(); return tablesFor(level, prev).gains; },
            availParamBase: (level, prev) => { checkAlive(); return tablesFor(level, prev).availParamBase; },
            style: (level, prev) => { checkAlive(); return tablesFor(level, prev).style; },
            styleCases: (level, prev) => { checkAlive(); return tablesFor(level, prev).styleCases; },
          },
          observe: (level, result) => {
            tablesFor(level + 1, result);
            checkAlive();
          },
          rankRoots: async (ls) => {
            checkAlive();
            const buildBest = (
              chain: readonly ResidentLevelResult[],
            ): Map<number, Map<number, number>> => {
              const outMap = new Map<number, Map<number, number>>();
              for (let l = 1; l < chain.length; l++) {
                const avail = slotAvail.get(l - 1) ?? new Map<number, Set<number>>();
                const prev = chain[l - 1] as ResidentLevelResult;
                const levelResult = chain[l] as ResidentLevelResult;
                const best = new Map<number, number>();
                for (let i = 0; i < BEAM; i++) {
                  if (prev.meta[i * META.WORDS + META.alive] < 0.5) continue;
                  let vBest = -Infinity;
                  (avail.get(i) ?? new Set<number>()).forEach((s) => {
                    const v = levelResult.values[i * SKILL_CAPACITY + s];
                    if (Number.isFinite(v) && v > vBest) vBest = v;
                  });
                  if (vBest > -Infinity) best.set(i, vBest);
                }
                outMap.set(l, best);
              }
              return outMap;
            };
            const foldContinuation = (
              chain: readonly ResidentLevelResult[],
              best: ReadonlyMap<number, ReadonlyMap<number, number>>,
              target: Map<number, number>,
            ): void => {
              const depthN = chain.length;
              let cont = new Map(best.get(depthN - 1) ?? new Map<number, number>());
              for (let l = depthN - 3; l >= 0; l--) {
                checkAlive();
                const r = chain[l + 1] as ResidentLevelResult;
                const parentAgg = new Map<number, number>();
                for (let j = 0; j < BEAM; j++) {
                  const mb = j * META.WORDS;
                  if (r.meta[mb + META.alive] < 0.5) continue;
                  const parent = Math.round(r.meta[mb + META.parentId]);
                  const p = r.meta[mb + META.probability];
                  const v = cont.get(j);
                  if (!(p > 0) || v === undefined || !Number.isFinite(v)) continue;
                  parentAgg.set(parent, (parentAgg.get(parent) ?? 0) + p * v);
                }
                cont = parentAgg;
              }
              cont.forEach((v, k) => target.set(k, (target.get(k) ?? 0) + v));
            };

            const baseBest = buildBest(ls);
            const firstBest = new Map(baseBest.get(1) ?? new Map<number, number>());
            const cont0 = new Map<number, number>();
            // The validated byParent/branchBest fold is always built above.
            if (fold) {
              if (ls.length === 3) {
                fold.byParent.forEach((v, k) => cont0.set(k, v));
                fold.branchBest.forEach((v, k) => {
                  if (!cont0.has(k)) cont0.set(k, v.value);
                });
              } else {
                const byParent2 = new Map<number, number>();
                const l2r = ls[2] as ResidentLevelResult;
                const best3 = baseBest.get(3) ?? new Map<number, number>();
                for (let j = 0; j < BEAM; j++) {
                  const mb = j * META.WORDS;
                  if (l2r.meta[mb + META.alive] < 0.5) continue;
                  const p3 = l2r.meta[mb + META.probability];
                  const v = best3.get(j);
                  if (!(p3 > 0) || v === undefined || !Number.isFinite(v)) continue;
                  const parent = Math.round(l2r.meta[mb + META.parentId]);
                  byParent2.set(parent, (byParent2.get(parent) ?? 0) + p3 * v);
                }
                const l1r = ls[1] as ResidentLevelResult;
                const best2 = baseBest.get(2) ?? new Map<number, number>();
                const represented = new Set<number>();
                for (let j = 0; j < BEAM; j++) {
                  const mb = j * META.WORDS;
                  if (l1r.meta[mb + META.alive] < 0.5) continue;
                  const parent = Math.round(l1r.meta[mb + META.parentId]);
                  represented.add(parent);
                  const p2 = l1r.meta[mb + META.probability];
                  const v = byParent2.get(j) ?? best2.get(j);
                  if (!(p2 > 0) || v === undefined || !Number.isFinite(v)) continue;
                  cont0.set(parent, (cont0.get(parent) ?? 0) + p2 * v);
                }
                fold.byParent.forEach((v, k) => {
                  if (!represented.has(k)) cont0.set(k, (cont0.get(k) ?? 0) + v);
                });
              }
            } else {
              foldContinuation(ls, baseBest, cont0);
            }

            // Validated fold already re-seeded dropped level-1 slots above.
            const raw = new Map<number, number>();
            const first = ls[0] as ResidentLevelResult;
            for (let i = 0; i < BEAM; i++) {
              const mb = i * META.WORDS;
              if (first.meta[mb + META.alive] < 0.5) continue;
              const move = Math.round(first.meta[mb + META.skill]);
              const p = first.meta[mb + META.probability];
              const v = cont0.get(i) ?? firstBest.get(i) ?? 0;
              if (!(p > 0) || !Number.isFinite(v)) continue;
              raw.set(move, (raw.get(move) ?? 0) + p * v);
            }
            // Stand :1241: apply root survival to the folded sums, then rank.
            checkAlive();
            const survival = rootSurvivalApply
              ? rootSurvivalApply(raw)
              : { corrected: new Map(raw), rootSurvival: {} as Readonly<Record<string, SurvivalInfo>> };
            rootSurvivalInfo = survival.rootSurvival;
            notes.rootSurvivalCases = Object.keys(survival.rootSurvival).length;
            const rank: Array<readonly [number, number]> = [];
            survival.corrected.forEach((score, move) => rank.push([move, score]));
            rank.sort((a, b) => b[1] - a[1]);
            if (rank.length === 0) return { move: -1, score: Number.NaN };
            bestChain = ls;
            bestRoot = rank[0][0];
            return { move: rank[0][0], score: rank[0][1] };
          },
        });
      } finally {
        destroyResidentBuffers(loopBuffers);
      }

      notes.completedLevels = loop.completedLevels;
      notes.totalNodes = loop.totalNodes;
      notes.stoppedByNodeCap = loop.stoppedByNodeCap ? 1 : 0;
      notes.gpuMs = bufferMs(bufferStart, now());
      // Kernel skill indices are the indices of the `skills` array uploaded
      // into the tables; use the index, never the display name, as the key.
      const chosenSkill = loop.topMove >= 0 ? skills[loop.topMove] : undefined;
      if (!chosenSkill || !Number.isFinite(loop.topScore)) {
        return refuse('device-resident beam: пустой результат выбора корня');
      }
      return {
        chosen: chosenSkill.key,
        comparableScore: loop.topScore,
        path: reachablePath(bestChain, bestRoot, fold, skills, skillSet),
        nodeCounts: loop.nodeCounts,
        totalNodes: loop.totalNodes,
        completedLevels: loop.completedLevels,
        stoppedByNodeCap: loop.stoppedByNodeCap,
        timing: { gpuMs: bufferMs(bufferStart, now()), totalMs: now() - started },
        notes,
      };
    };

    try {
      return await Promise.race([work(), deadline]);
    } catch (error: unknown) {
      return refuse(`device-resident beam прерван: ${String(error)}`);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  } catch (error: unknown) {
    return refuse(`device-resident beam: ${String(error)}`);
  }
}

// Helpers kept below the entry point; all are pure host-side compositions.
function getAvailableSkillIndices(
  display: CraftingState,
  config: OptimizerConfig,
  condition: string,
  skills: readonly SkillDefinition[],
): number[] {
  return getAvailableSkills(display, config, condition)
    .map((skill) => skills.findIndex((candidate) => candidate.key === skill.key))
    .filter((index) => index >= 0);
}

function buildScoreWordsFor(state: CraftingState, condition: string): Float32Array {
  // Same projection the validated stand used; buildScoreWords reads only the
  // listed fields, so the cast narrows nothing at runtime.
  return buildScoreWords(state as never, conditionIdOf(condition));
}

/** Device-derived greedy continuation; marked as such by the caller's notes. */
function reachablePath(
  chain: readonly ResidentLevelResult[] | null,
  root: number,
  fold: {
    readonly byParent: ReadonlyMap<number, number>;
    readonly branchBest: ReadonlyMap<number, { value: number; skill: number; slot: number }>;
  } | null,
  skills: readonly SkillDefinition[],
  skillSet: { readonly names: readonly string[] },
): readonly string[] {
  const first = chain?.[0];
  if (!first || !fold || root < 0) return [];
  const path: string[] = [skills[root]?.key ?? skillSet.names[root] ?? String(root)];
  let slot = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < BEAM; i++) {
    const mb = i * META.WORDS;
    if (first.meta[mb + META.alive] < 0.5) continue;
    if (Math.round(first.meta[mb + META.skill]) !== root) continue;
    const candidate = fold.branchBest.get(i);
    if (candidate && candidate.value > bestValue) {
      bestValue = candidate.value;
      slot = i;
    }
  }
  if (slot < 0) return path;
  const second = fold.branchBest.get(slot);
  if (second && skills[second.skill]) {
    path.push(skills[second.skill]?.key ?? String(second.skill));
  }
  return path;
}

/** Elapsed GPU-side window for the dump fold + level loop, in ms. */
function bufferMs(started: number, ended: number): number {
  return Math.max(0, Math.round(ended - started));
}

