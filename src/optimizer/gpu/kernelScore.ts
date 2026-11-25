/**
 * Engine-faithful single-state scoring kernel (WGSL).
 *
 * Independent of the transition kernel (kernel.ts / kernelState.ts): this file
 * ports the *scoring* side only — `tools/gpu-probe/mirror.mjs`
 * `scoreStateMirrorDetailed` and its helpers, which are verified to 5.8e-11
 * against `src/optimizer/search/scoring.ts:393-732` on the 179 replay children.
 * Every helper below cites the model / mirror lines it reproduces.
 *
 * Layouts are flat f32 arrays (no uniform-struct alignment surprises):
 *   states[i * SCORE_INPUT.WORDS + word] — one score input per case;
 *   params[i * SCORE_PARAM.WORDS + word] — one config-derived parameter block
 *     per case, including all 35 scoring weights. Putting the weights in the
 *     parameter block is what makes the harness failure proof possible: a
 *     temporary copy with one changed weight must report widespread diffs.
 *   scores[i] — the computed score (f32).
 *
 * f32 caveats (the JS reference uses IEEE doubles):
 *   - `Math.floor(x * 1.3)` (outcome.ts:144,160) can land on a different side
 *     of an integer boundary when `x * 1.3` is within ~1 ulp of an integer;
 *   - `Math.ceil` of the pairwise-rate coverage (scoringContext.ts:173) can
 *     differ by one turn for the same reason;
 *   - `Math.floor` / weighted sums accumulate in f32, so large magnitudes
 *     (targets up to ~6e5) carry absolute error up to ~0.06 there.
 *   JS-only features with no f32 analogue are normalised at the export
 *   boundary: `undefined` / non-finite caps and band goals become -1,
 *   `Number.POSITIVE_INFINITY` becomes INF (1e30), and `Number.isFinite`
 *   becomes a NaN/Inf comparison. `targetMultiplier` is read into params for
 *   diagnostics but the engine explicitly ignores it (scoring.ts:412-413).
 */

/** Per-case state input. Values are plain f32 words written by the exporter. */
export const SCORE_INPUT = {
  completion: 0,
  perfection: 1,
  qi: 2,
  stability: 3,
  step: 4,
  controlBuffTurns: 5,
  controlBuffMultiplier: 6,
  intensityBuffTurns: 7,
  intensityBuffMultiplier: 8,
  toxicity: 9,
  maxToxicity: 10,
  harmony: 11,
  // Harmony subsystem snapshot (harmonyScoring.ts:99-238). Opcode 0 = none,
  // 1 forge, 2 inscription, 3 alchemical, 4 resonance, 5 captivating cadence.
  harmonyKind: 12,
  forgeHeat: 13,
  forgeLastBuffedHeat: 14,
  inscribedStacks: 15,
  inscribedRemaining: 16,
  alchemicalCharges: 17,
  alchemicalModsFinite: 18,
  recommendedCount: 19,
  recommended: 20, // 8 words: 0 fusion, 1 refine, 2 other
  resonanceStrength: 28,
  resonanceType: 29, // -1 = undefined
  pendingResonanceType: 30, // -1 = none
  pendingCount: 31,
  cadenceChain: 32,
  alchemicalControlMult: 33,
  alchemicalIntensityMult: 34,
  alchemicalCritBonus: 35,
  alchemicalSuccessBonus: 36,
  alchemicalQiReduction: 37,
  alchemicalStabReduction: 38,
  // Cosmetic/parity mirror of the current condition hash (records.conditionIdOf).
  // WGSL does not read it: condition effects are config-derived and arrive in
  // SCORE_PARAM as control/intensity deltas, exactly like every other cfg value.
  conditionId: 39,
  WORDS: 40,
} as const;

/** Maximum number of ctx.progressRates pairs carried in a parameter block. */
export const SCORE_MAX_RATES = 8;
/** Maximum number of recommended technique types (harmonyScoring.ts:164). */
export const SCORE_MAX_RECOMMENDED = 8;

/** Weights quoted verbatim from scoring.ts:43-161 (mirror.mjs SCORE_W). */
export const SCORE_WEIGHTS = {
  TIER_VALUE_SCALE: 2,
  GATE_WEIGHT: 1,
  BALANCE_WEIGHT: 0.35,
  BONUS_ROLL_WEIGHT: 0.2,
  IN_TIER_PROGRESS_WEIGHT: 0.05,
  EXTRA_BAND_WEIGHT: 0.75,
  COMPLETION_EXTRA_BAND_WEIGHT: 0.25,
  RESOURCE_TIEBREAKER: 0.001,
  STEP_PENALTY: 0.5,
  STEP_PENALTY_PROGRESS_FRACTION: 0.25,
  BUFF_VALUE_PER_MULTIPLIER_POINT: 25,
  BUFF_NEED_FLOOR: 0.5,
  GOAL_PRIORITY_WEIGHT_SHIFT: 0.5,
  QI_RESOURCE_WEIGHT: 0.05,
  STABILITY_BASE_WEIGHT: 0.01,
  STABILITY_WORK_WEIGHT: 0.01,
  OVERSHOOT_PENALTY_WEIGHT: 0.3,
  FINISHED_UNMET_PENALTY_WEIGHT: 1,
  HARD_CAP_PENALTY_WEIGHT: 3,
  STABILITY_THRESHOLD_TURNS_BASE: 1.4,
  STABILITY_THRESHOLD_TURNS_SCALE: 2.6,
  STABILITY_THRESHOLD_TURNS_BASE_TRAINING: 0.8,
  STABILITY_THRESHOLD_TURNS_SCALE_TRAINING: 0.8,
  STABILITY_PENALTY_FRACTION: 0.45,
  STABILITY_PENALTY_FRACTION_TRAINING: 0.08,
  STABILITY_PENALTY_FLOOR: 45,
  STABILITY_PENALTY_FLOOR_TRAINING: 8,
  NEAR_DEATH_STABILITY: 10,
  DEATH_PENALTY_MULTIPLIER: 3,
  RUNWAY_GAP_FRACTION: 0.1,
  RUNWAY_GAP_FRACTION_TRAINING: 0.04,
  TOXICITY_PENALTY_FRACTION: 0.025,
  HARMONY_BONUS_WEIGHT: 0.4,
  HARMONY_SUBSYSTEM_QUALITY_WEIGHT: 0.15,
  GOAL_UNLOCK_ORDERING_WEIGHT: 0.5,
} as const;

export type ScoreWeightName = keyof typeof SCORE_WEIGHTS;
export const SCORE_WEIGHT_NAMES = Object.keys(SCORE_WEIGHTS) as ScoreWeightName[];

/** WGSL weight indices, generated from the same order as SCORE_WEIGHTS. */
export const SCORE_W_INDEX = Object.fromEntries(
  SCORE_WEIGHT_NAMES.map((name, index) => [name, index]),
) as Readonly<Record<ScoreWeightName, number>>;

/** Per-case parameter block. -1 means "undefined" for optional numbers. */
export const SCORE_PARAM = {
  targetCompletion: 0,
  targetPerfection: 1,
  isSublime: 2,
  trainingMode: 3,
  maxCompletion: 4, // -1 when undefined / non-finite
  maxPerfection: 5,
  hasMaxCompletion: 6, // hard-cap penalty gate (scoring.ts:635)
  hasMaxPerfection: 7,
  goalPriorityBias: 8,
  overcraftAmbition: 9,
  perfectionBandGoal: 10, // -1 when undefined / non-finite / <= 0
  completionBandCeiling: 11,
  targetMultiplier: 12, // unused by the engine; carried for diagnostics
  ctxCompletionGain: 13,
  ctxPerfectionGain: 14,
  ctxAvgGain: 15,
  ctxStabilityCost: 16,
  ctxQiCost: 17,
  ctxRateCount: 18,
  ctxRates: 19, // 8 pairs (completion, perfection) -> 16 words
  // outcome.ts:144,160 computes floor(width * 1.3) in JS doubles. f32's 1.3 is
  // slightly below the double value, so 2590*1.3 floors to 3366 instead of
  // 3367. The exact ladders and derived goals are therefore computed on the
  // CPU and passed here; WGSL only walks the supplied thresholds.
  bandReqC: 35,
  bandReqP: 36,
  targetTierRank: 37,
  derivedCompFlat: 38,
  derivedPerfFlat: 39,
  capCompBands: 40, // -1 = undefined
  capPerfBands: 41,
  baseCompGoal: 42,
  basePerfGoal: 43,
  effectiveCompGoal: 44,
  effectivePerfGoal: 45,
  ladderComp: 46, // 8 exact thresholds: target, floor(target*1.3), ...
  ladderPerf: 54,
  weights: 62,
  // ── condition-aware scoring (lookahead.ts:237-331) ────────────────────
  // Precomputed by the exporter from getConditionEffectsForConfig. The two
  // deltas are config-derived. `terminalMode` is NOT the decision any more:
  // it is a packed HOST boundary word (see ScoreConfigLike.terminalBits):
  //   bit 0 — availability-derived isTerminalState (skills.ts:6104-6110),
  //           host-owned until S3;
  //   bit 1 — forced finished-outcome pricing of a stand case (death/node/
  //           finished); a pricing mode, not an availability decision;
  //   bit 2 — N8 negative probe only: force terminalMode=false for one case.
  // The device derives willAutoFinish from the score-input bars and combines.
  conditionControlDelta: 62 + Object.keys(SCORE_WEIGHTS).length,
  conditionIntensityDelta: 63 + Object.keys(SCORE_WEIGHTS).length,
  terminalMode: 64 + Object.keys(SCORE_WEIGHTS).length,
  WORDS: 65 + Object.keys(SCORE_WEIGHTS).length,
} as const;

/** Word count of the per-case terminality readout written by the S2 entry. */
export const TERMINALITY_WORDS = 5;

const BAND_GROWTH_RATIO = 1.3;

/** Exact JS-double band ladder, matching outcome.ts:130-152. */
function bandLadder(target: number, capacity: number): number[] {
  const values: number[] = [];
  let current = target;
  for (let i = 0; i < capacity; i++) {
    values.push(current);
    if (!(current > 0)) break;
    current = Math.floor(current * BAND_GROWTH_RATIO);
  }
  while (values.length < capacity) values.push(0);
  return values;
}

function bandThresholdFromLadder(ladder: readonly number[], count: number): number {
  let total = 0;
  for (let i = 0; i < Math.min(count, ladder.length); i++) total += ladder[i];
  return Math.floor(total);
}

function guaranteedFromLadder(ladder: readonly number[], value: number): number {
  let remaining = value;
  let guaranteed = 0;
  for (const threshold of ladder) {
    if (!(remaining > 0) || !(threshold > 0) || remaining < threshold) break;
    remaining -= threshold;
    guaranteed++;
  }
  return guaranteed;
}

/** Structural view of the state fields scoreState reads (state.ts:469-498). */
export interface ScoreStateLike {
  completion: number;
  perfection: number;
  qi: number;
  stability: number;
  step: number;
  controlBuffTurns: number;
  controlBuffMultiplier: number;
  intensityBuffTurns: number;
  intensityBuffMultiplier: number;
  toxicity: number;
  maxToxicity: number;
  harmony: number;
  harmonyData?: unknown;
  /** Pre-resolved alchemical stat modifiers (harmony.ts:1136-1141). */
  resolvedAlchemicalModifiers?: Readonly<Record<string, number>>;
}

function fin(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function obj(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : {};
}

function alignment(type: unknown): number {
  return type === 'fusion' ? 0 : type === 'refine' ? 1 : 2;
}

/** Flatten one engine state into SCORE_INPUT words (exporter-side only). */
export function buildScoreWords(
  state: ScoreStateLike,
  conditionId: number = 0,
): Float32Array {
  const words = new Float32Array(SCORE_INPUT.WORDS);
  words[SCORE_INPUT.conditionId] = conditionId;
  words[SCORE_INPUT.completion] = fin(state.completion);
  words[SCORE_INPUT.perfection] = fin(state.perfection);
  words[SCORE_INPUT.qi] = fin(state.qi);
  words[SCORE_INPUT.stability] = fin(state.stability);
  words[SCORE_INPUT.step] = fin(state.step);
  words[SCORE_INPUT.controlBuffTurns] = fin(state.controlBuffTurns);
  words[SCORE_INPUT.controlBuffMultiplier] = fin(state.controlBuffMultiplier, 1);
  words[SCORE_INPUT.intensityBuffTurns] = fin(state.intensityBuffTurns);
  words[SCORE_INPUT.intensityBuffMultiplier] = fin(state.intensityBuffMultiplier, 1);
  words[SCORE_INPUT.toxicity] = fin(state.toxicity);
  words[SCORE_INPUT.maxToxicity] = fin(state.maxToxicity);
  words[SCORE_INPUT.harmony] = fin(state.harmony);

  const hd = obj(state.harmonyData);
  if (hd.forgeWorks) {
    const fw = obj(hd.forgeWorks);
    words[SCORE_INPUT.harmonyKind] = 1;
    words[SCORE_INPUT.forgeHeat] = fin(fw.heat);
    words[SCORE_INPUT.forgeLastBuffedHeat] = fin(fw.lastBuffedHeat, words[SCORE_INPUT.forgeHeat]);
  } else if (hd.inscribedPatterns) {
    const ins = obj(hd.inscribedPatterns);
    words[SCORE_INPUT.harmonyKind] = 2;
    words[SCORE_INPUT.inscribedStacks] = fin(ins.stacks);
    const block = ins.currentBlock;
    words[SCORE_INPUT.inscribedRemaining] = fin(
      Array.isArray(block) ? block.length : 5,
      5,
    );
  } else if (hd.alchemicalArts) {
    const alc = obj(hd.alchemicalArts);
    words[SCORE_INPUT.harmonyKind] = 3;
    const charges = alc.charges;
    words[SCORE_INPUT.alchemicalCharges] = fin(Array.isArray(charges) ? charges.length : 0);
    const mods = state.resolvedAlchemicalModifiers ?? {};
    words[SCORE_INPUT.alchemicalModsFinite] =
      Object.values(mods).every((v) => Number.isFinite(v)) ? 1 : 0;
    words[SCORE_INPUT.alchemicalControlMult] = fin(mods.controlMultiplier, 1);
    words[SCORE_INPUT.alchemicalIntensityMult] = fin(mods.intensityMultiplier, 1);
    words[SCORE_INPUT.alchemicalCritBonus] = fin(mods.critChanceBonus);
    words[SCORE_INPUT.alchemicalSuccessBonus] = fin(mods.successChanceBonus);
    words[SCORE_INPUT.alchemicalQiReduction] = fin(mods.qiCostReduction);
    words[SCORE_INPUT.alchemicalStabReduction] = fin(mods.stabilityCostReduction);
    const recommended = Array.isArray(hd.recommendedTechniqueTypes)
      ? hd.recommendedTechniqueTypes
      : [];
    words[SCORE_INPUT.recommendedCount] = Math.min(recommended.length, SCORE_MAX_RECOMMENDED);
    for (let i = 0; i < words[SCORE_INPUT.recommendedCount]; i++) {
      words[SCORE_INPUT.recommended + i] = alignment(recommended[i]);
    }
  } else if (hd.resonance) {
    const res = obj(hd.resonance);
    words[SCORE_INPUT.harmonyKind] = 4;
    words[SCORE_INPUT.resonanceStrength] = fin(res.strength);
    words[SCORE_INPUT.resonanceType] =
      res.resonance === undefined ? -1 : alignment(res.resonance);
    words[SCORE_INPUT.pendingResonanceType] = res.pendingResonance
      ? alignment(res.pendingResonance)
      : -1;
    words[SCORE_INPUT.pendingCount] = fin(res.pendingCount);
  } else if (hd.captivatingCadence) {
    words[SCORE_INPUT.harmonyKind] = 5;
    words[SCORE_INPUT.cadenceChain] = fin(obj(hd.captivatingCadence).chain);
  }
  return words;
}

/** Config-derived scoring context, as built by scoringContext.ts:26-42. */
export interface ScoreContextLike {
  avgCompletionGainPerTurn: number;
  avgPerfectionGainPerTurn: number;
  avgGainPerTurn: number;
  avgStabilityCostPerTurn: number;
  avgQiCostPerTurn: number;
  progressRates?: ReadonlyArray<{ completion: number; perfection: number }>;
}

export interface ScoreConfigLike {
  targetCompletion: number;
  targetPerfection: number;
  isSublime?: boolean;
  trainingMode?: boolean;
  targetMultiplier?: number;
  maxCompletion?: number;
  maxPerfection?: number;
  goalPriorityBias?: number;
  overcraftAmbition?: boolean;
  perfectionBandGoal?: number;
  completionBandCeiling?: number;
  ctx?: ScoreContextLike;
  weights?: Partial<Readonly<Record<ScoreWeightName, number>>>;
  /** Sum of `multiplier` over `kind: 'control'` effects of the branch condition. */
  conditionControlDelta?: number;
  /** Sum of `multiplier` over `kind: 'intensity'` effects of the branch condition. */
  conditionIntensityDelta?: number;
  /**
   * Legacy 1/0 form (keeps s6a-engine-score.ts and export-score-cases.ts
   * working): true maps to host bit 0, availability-derived terminality.
   */
  terminalMode?: boolean;
  /** Packed host boundary word. Takes precedence over `terminalMode`. */
  terminalBits?: number;
}

function optionalPositive(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : -1;
}

/** Build one parameter block (exporter-side only). */
export function buildScoreParams(config: ScoreConfigLike): Float32Array {
  const params = new Float32Array(SCORE_PARAM.WORDS);
  params[SCORE_PARAM.targetCompletion] = fin(config.targetCompletion);
  params[SCORE_PARAM.targetPerfection] = fin(config.targetPerfection);
  params[SCORE_PARAM.isSublime] = config.isSublime === true ? 1 : 0;
  params[SCORE_PARAM.trainingMode] = config.trainingMode === true ? 1 : 0;
  const maxC = config.maxCompletion;
  const maxP = config.maxPerfection;
  params[SCORE_PARAM.maxCompletion] =
    maxC !== undefined && Number.isFinite(maxC) ? maxC : -1;
  params[SCORE_PARAM.maxPerfection] =
    maxP !== undefined && Number.isFinite(maxP) ? maxP : -1;
  params[SCORE_PARAM.hasMaxCompletion] =
    maxC !== undefined && Number.isFinite(maxC) ? 1 : 0;
  params[SCORE_PARAM.hasMaxPerfection] =
    maxP !== undefined && Number.isFinite(maxP) ? 1 : 0;
  params[SCORE_PARAM.goalPriorityBias] = fin(config.goalPriorityBias);
  params[SCORE_PARAM.overcraftAmbition] = config.overcraftAmbition !== false ? 1 : 0;
  params[SCORE_PARAM.perfectionBandGoal] = optionalPositive(config.perfectionBandGoal);
  params[SCORE_PARAM.completionBandCeiling] = optionalPositive(config.completionBandCeiling);
  params[SCORE_PARAM.targetMultiplier] = fin(config.targetMultiplier, 2);
  const ctx = config.ctx;
  params[SCORE_PARAM.ctxCompletionGain] = fin(ctx?.avgCompletionGainPerTurn, 16);
  params[SCORE_PARAM.ctxPerfectionGain] = fin(ctx?.avgPerfectionGainPerTurn, 16);
  params[SCORE_PARAM.ctxAvgGain] = fin(ctx?.avgGainPerTurn, 16);
  params[SCORE_PARAM.ctxStabilityCost] = fin(ctx?.avgStabilityCostPerTurn, 10);
  params[SCORE_PARAM.ctxQiCost] = fin(ctx?.avgQiCostPerTurn);
  const rates = (ctx?.progressRates ?? []).slice(0, SCORE_MAX_RATES);
  params[SCORE_PARAM.ctxRateCount] = rates.length;
  for (let i = 0; i < rates.length; i++) {
    params[SCORE_PARAM.ctxRates + i * 2] = fin(rates[i].completion);
    params[SCORE_PARAM.ctxRates + i * 2 + 1] = fin(rates[i].perfection);
  }

  // outcome.ts:169-183 / 260-318 + goals.ts:317-383, computed in JS doubles.
  const sublime = config.isSublime === true;
  const reqC = sublime ? 2 : 1;
  const reqP = sublime ? 2 : 1;
  const targetRank = sublime ? 3 : 2;
  const ladderC = bandLadder(Math.max(0, config.targetCompletion), 8);
  const ladderP = bandLadder(Math.max(0, config.targetPerfection), 8);
  const derivedCompFlat = bandThresholdFromLadder(ladderC, reqC);
  const derivedPerfFlat = bandThresholdFromLadder(ladderP, reqP);
  const finishC = maxC !== undefined && maxC > 0 ? maxC : derivedCompFlat;
  const finishP = maxP !== undefined && maxP > 0 ? maxP : derivedPerfFlat;
  const capBandsC = finishC > 0 && config.targetCompletion > 0
    ? guaranteedFromLadder(ladderC, finishC) : -1;
  const capBandsP = finishP > 0 && config.targetPerfection > 0
    ? guaranteedFromLadder(ladderP, finishP) : -1;
  const clampGoal = (goal: number, cap: number | undefined) =>
    cap !== undefined && Number.isFinite(cap) && cap > 0 ? Math.min(goal, cap) : goal;
  const ambitionPerf = optionalPositive(config.perfectionBandGoal);
  const perfGoalBands = ambitionPerf > reqP ? ambitionPerf : reqP;
  params[SCORE_PARAM.bandReqC] = reqC;
  params[SCORE_PARAM.bandReqP] = reqP;
  params[SCORE_PARAM.targetTierRank] = targetRank;
  params[SCORE_PARAM.derivedCompFlat] = derivedCompFlat;
  params[SCORE_PARAM.derivedPerfFlat] = derivedPerfFlat;
  params[SCORE_PARAM.capCompBands] = capBandsC;
  params[SCORE_PARAM.capPerfBands] = capBandsP;
  params[SCORE_PARAM.baseCompGoal] = config.targetCompletion > 0
    ? clampGoal(bandThresholdFromLadder(ladderC, 1), maxC) : 0;
  params[SCORE_PARAM.basePerfGoal] = config.targetPerfection > 0
    ? clampGoal(bandThresholdFromLadder(ladderP, 1), maxP) : 0;
  params[SCORE_PARAM.effectiveCompGoal] = config.targetCompletion > 0
    ? clampGoal(derivedCompFlat, maxC) : 0;
  params[SCORE_PARAM.effectivePerfGoal] = config.targetPerfection > 0
    ? clampGoal(bandThresholdFromLadder(ladderP, perfGoalBands), maxP) : 0;
  for (let i = 0; i < 8; i++) {
    params[SCORE_PARAM.ladderComp + i] = ladderC[i];
    params[SCORE_PARAM.ladderPerf + i] = ladderP[i];
  }
  for (const name of SCORE_WEIGHT_NAMES) {
    params[SCORE_PARAM.weights + SCORE_W_INDEX[name]] = fin(
      config.weights?.[name] ?? SCORE_WEIGHTS[name],
    );
  }
  params[SCORE_PARAM.conditionControlDelta] = fin(config.conditionControlDelta);
  params[SCORE_PARAM.conditionIntensityDelta] = fin(config.conditionIntensityDelta);
  params[SCORE_PARAM.terminalMode] =
    config.terminalBits ?? (config.terminalMode === true ? 1 : 0);
  return params;
}

/** WGSL source. Cites are model line numbers in search/scoring.ts et al. */
export const SCORE_KERNEL_WGSL = `
@group(0) @binding(0) var<storage, read> states: array<f32>;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<storage, read> scoreParams: array<f32>;
// S2 readout. Only the scoreWithTerminalityKernel entry point uses it, so the
// auto layout of the original scoreStateKernel (3 bindings) stays unchanged.
@group(0) @binding(3) var<storage, read_write> terminality: array<f32>;
// Stand-only split readout. Only scoreSplitKernel uses binding 4, so existing
// entry-point auto layouts stay unchanged.
@group(0) @binding(4) var<storage, read_write> split: array<f32>;

const SI: u32 = ${SCORE_INPUT.WORDS}u;
const PS: u32 = ${SCORE_PARAM.WORDS}u;
const PW: u32 = ${SCORE_PARAM.weights}u;
const INF: f32 = 1.0e30;
const MAX_BAND_STEPS: u32 = 64u;
const MAX_RATES: u32 = ${SCORE_MAX_RATES}u;
const MAX_RECOMMENDED: u32 = ${SCORE_MAX_RECOMMENDED}u;

// tier ranks, outcome.ts:169-183
const TIER_FAILED: f32 = 0.0;
const TIER_BASIC: f32 = 1.0;
const TIER_PERFECT: f32 = 2.0;
const TIER_SUBLIME: f32 = 3.0;

const S_COMPLETION: u32 = ${SCORE_INPUT.completion}u;
const S_PERFECTION: u32 = ${SCORE_INPUT.perfection}u;
const S_QI: u32 = ${SCORE_INPUT.qi}u;
const S_STABILITY: u32 = ${SCORE_INPUT.stability}u;
const S_STEP: u32 = ${SCORE_INPUT.step}u;
const S_CONTROL_TURNS: u32 = ${SCORE_INPUT.controlBuffTurns}u;
const S_CONTROL_MULT: u32 = ${SCORE_INPUT.controlBuffMultiplier}u;
const S_INTENSITY_TURNS: u32 = ${SCORE_INPUT.intensityBuffTurns}u;
const S_INTENSITY_MULT: u32 = ${SCORE_INPUT.intensityBuffMultiplier}u;
const S_TOXICITY: u32 = ${SCORE_INPUT.toxicity}u;
const S_MAX_TOXICITY: u32 = ${SCORE_INPUT.maxToxicity}u;
const S_HARMONY: u32 = ${SCORE_INPUT.harmony}u;
const S_HKIND: u32 = ${SCORE_INPUT.harmonyKind}u;
const S_HEAT: u32 = ${SCORE_INPUT.forgeHeat}u;
const S_LAST_HEAT: u32 = ${SCORE_INPUT.forgeLastBuffedHeat}u;
const S_INS_STACKS: u32 = ${SCORE_INPUT.inscribedStacks}u;
const S_INS_REMAIN: u32 = ${SCORE_INPUT.inscribedRemaining}u;
const S_CHARGES: u32 = ${SCORE_INPUT.alchemicalCharges}u;
const S_MODS_FINITE: u32 = ${SCORE_INPUT.alchemicalModsFinite}u;
const S_REC_COUNT: u32 = ${SCORE_INPUT.recommendedCount}u;
const S_REC: u32 = ${SCORE_INPUT.recommended}u;
const S_RES_STRENGTH: u32 = ${SCORE_INPUT.resonanceStrength}u;
const S_RES_TYPE: u32 = ${SCORE_INPUT.resonanceType}u;
const S_PENDING_TYPE: u32 = ${SCORE_INPUT.pendingResonanceType}u;
const S_PENDING_COUNT: u32 = ${SCORE_INPUT.pendingCount}u;
const S_CADENCE: u32 = ${SCORE_INPUT.cadenceChain}u;
const S_AC: u32 = ${SCORE_INPUT.alchemicalControlMult}u;
const S_AI: u32 = ${SCORE_INPUT.alchemicalIntensityMult}u;
const S_ACRIT: u32 = ${SCORE_INPUT.alchemicalCritBonus}u;
const S_ASUCCESS: u32 = ${SCORE_INPUT.alchemicalSuccessBonus}u;
const S_AQI: u32 = ${SCORE_INPUT.alchemicalQiReduction}u;
const S_ASTAB: u32 = ${SCORE_INPUT.alchemicalStabReduction}u;
const S_CONDITION_ID: u32 = ${SCORE_INPUT.conditionId}u; // parity/diagnostic only

const P_TARGET_C: u32 = ${SCORE_PARAM.targetCompletion}u;
const P_TARGET_P: u32 = ${SCORE_PARAM.targetPerfection}u;
const P_SUBLIME: u32 = ${SCORE_PARAM.isSublime}u;
const P_TRAINING: u32 = ${SCORE_PARAM.trainingMode}u;
const P_MAX_C: u32 = ${SCORE_PARAM.maxCompletion}u;
const P_MAX_P: u32 = ${SCORE_PARAM.maxPerfection}u;
const P_HAS_MAX_C: u32 = ${SCORE_PARAM.hasMaxCompletion}u;
const P_HAS_MAX_P: u32 = ${SCORE_PARAM.hasMaxPerfection}u;
const P_BIAS: u32 = ${SCORE_PARAM.goalPriorityBias}u;
const P_AMBITION: u32 = ${SCORE_PARAM.overcraftAmbition}u;
const P_PBG: u32 = ${SCORE_PARAM.perfectionBandGoal}u;
const P_CBC: u32 = ${SCORE_PARAM.completionBandCeiling}u;
const P_CGAIN: u32 = ${SCORE_PARAM.ctxCompletionGain}u;
const P_PGAIN: u32 = ${SCORE_PARAM.ctxPerfectionGain}u;
const P_AGAIN: u32 = ${SCORE_PARAM.ctxAvgGain}u;
const P_SCOST: u32 = ${SCORE_PARAM.ctxStabilityCost}u;
const P_QCOST: u32 = ${SCORE_PARAM.ctxQiCost}u;
const P_RCOUNT: u32 = ${SCORE_PARAM.ctxRateCount}u;
const P_RATES: u32 = ${SCORE_PARAM.ctxRates}u;
const P_REQ_C: u32 = ${SCORE_PARAM.bandReqC}u;
const P_REQ_P: u32 = ${SCORE_PARAM.bandReqP}u;
const P_TIER_RANK: u32 = ${SCORE_PARAM.targetTierRank}u;
const P_DERIVED_C: u32 = ${SCORE_PARAM.derivedCompFlat}u;
const P_DERIVED_P: u32 = ${SCORE_PARAM.derivedPerfFlat}u;
const P_CAP_BANDS_C: u32 = ${SCORE_PARAM.capCompBands}u;
const P_CAP_BANDS_P: u32 = ${SCORE_PARAM.capPerfBands}u;
const P_BASE_C: u32 = ${SCORE_PARAM.baseCompGoal}u;
const P_BASE_P: u32 = ${SCORE_PARAM.basePerfGoal}u;
const P_EFF_C: u32 = ${SCORE_PARAM.effectiveCompGoal}u;
const P_EFF_P: u32 = ${SCORE_PARAM.effectivePerfGoal}u;
const P_LADDER_C: u32 = ${SCORE_PARAM.ladderComp}u;
const P_LADDER_P: u32 = ${SCORE_PARAM.ladderPerf}u;
const P_COND_CONTROL: u32 = ${SCORE_PARAM.conditionControlDelta}u;
const P_COND_INTENSITY: u32 = ${SCORE_PARAM.conditionIntensityDelta}u;
const P_TERMINAL: u32 = ${SCORE_PARAM.terminalMode}u;

const W_TIER_VALUE_SCALE: u32 = ${SCORE_W_INDEX.TIER_VALUE_SCALE}u;
const W_GATE_WEIGHT: u32 = ${SCORE_W_INDEX.GATE_WEIGHT}u;
const W_BALANCE_WEIGHT: u32 = ${SCORE_W_INDEX.BALANCE_WEIGHT}u;
const W_BONUS_ROLL_WEIGHT: u32 = ${SCORE_W_INDEX.BONUS_ROLL_WEIGHT}u;
const W_IN_TIER_PROGRESS_WEIGHT: u32 = ${SCORE_W_INDEX.IN_TIER_PROGRESS_WEIGHT}u;
const W_EXTRA_BAND_WEIGHT: u32 = ${SCORE_W_INDEX.EXTRA_BAND_WEIGHT}u;
const W_COMPLETION_EXTRA_BAND_WEIGHT: u32 = ${SCORE_W_INDEX.COMPLETION_EXTRA_BAND_WEIGHT}u;
const W_RESOURCE_TIEBREAKER: u32 = ${SCORE_W_INDEX.RESOURCE_TIEBREAKER}u;
const W_STEP_PENALTY: u32 = ${SCORE_W_INDEX.STEP_PENALTY}u;
const W_STEP_PENALTY_PROGRESS_FRACTION: u32 = ${SCORE_W_INDEX.STEP_PENALTY_PROGRESS_FRACTION}u;
const W_BUFF_VALUE_PER_MULTIPLIER_POINT: u32 = ${SCORE_W_INDEX.BUFF_VALUE_PER_MULTIPLIER_POINT}u;
const W_BUFF_NEED_FLOOR: u32 = ${SCORE_W_INDEX.BUFF_NEED_FLOOR}u;
const W_GOAL_PRIORITY_WEIGHT_SHIFT: u32 = ${SCORE_W_INDEX.GOAL_PRIORITY_WEIGHT_SHIFT}u;
const W_QI_RESOURCE_WEIGHT: u32 = ${SCORE_W_INDEX.QI_RESOURCE_WEIGHT}u;
const W_STABILITY_BASE_WEIGHT: u32 = ${SCORE_W_INDEX.STABILITY_BASE_WEIGHT}u;
const W_STABILITY_WORK_WEIGHT: u32 = ${SCORE_W_INDEX.STABILITY_WORK_WEIGHT}u;
const W_OVERSHOOT_PENALTY_WEIGHT: u32 = ${SCORE_W_INDEX.OVERSHOOT_PENALTY_WEIGHT}u;
const W_FINISHED_UNMET_PENALTY_WEIGHT: u32 = ${SCORE_W_INDEX.FINISHED_UNMET_PENALTY_WEIGHT}u;
const W_HARD_CAP_PENALTY_WEIGHT: u32 = ${SCORE_W_INDEX.HARD_CAP_PENALTY_WEIGHT}u;
const W_STABILITY_THRESHOLD_TURNS_BASE: u32 = ${SCORE_W_INDEX.STABILITY_THRESHOLD_TURNS_BASE}u;
const W_STABILITY_THRESHOLD_TURNS_SCALE: u32 = ${SCORE_W_INDEX.STABILITY_THRESHOLD_TURNS_SCALE}u;
const W_STABILITY_THRESHOLD_TURNS_BASE_TRAINING: u32 = ${SCORE_W_INDEX.STABILITY_THRESHOLD_TURNS_BASE_TRAINING}u;
const W_STABILITY_THRESHOLD_TURNS_SCALE_TRAINING: u32 = ${SCORE_W_INDEX.STABILITY_THRESHOLD_TURNS_SCALE_TRAINING}u;
const W_STABILITY_PENALTY_FRACTION: u32 = ${SCORE_W_INDEX.STABILITY_PENALTY_FRACTION}u;
const W_STABILITY_PENALTY_FRACTION_TRAINING: u32 = ${SCORE_W_INDEX.STABILITY_PENALTY_FRACTION_TRAINING}u;
const W_STABILITY_PENALTY_FLOOR: u32 = ${SCORE_W_INDEX.STABILITY_PENALTY_FLOOR}u;
const W_STABILITY_PENALTY_FLOOR_TRAINING: u32 = ${SCORE_W_INDEX.STABILITY_PENALTY_FLOOR_TRAINING}u;
const W_NEAR_DEATH_STABILITY: u32 = ${SCORE_W_INDEX.NEAR_DEATH_STABILITY}u;
const W_DEATH_PENALTY_MULTIPLIER: u32 = ${SCORE_W_INDEX.DEATH_PENALTY_MULTIPLIER}u;
const W_RUNWAY_GAP_FRACTION: u32 = ${SCORE_W_INDEX.RUNWAY_GAP_FRACTION}u;
const W_RUNWAY_GAP_FRACTION_TRAINING: u32 = ${SCORE_W_INDEX.RUNWAY_GAP_FRACTION_TRAINING}u;
const W_TOXICITY_PENALTY_FRACTION: u32 = ${SCORE_W_INDEX.TOXICITY_PENALTY_FRACTION}u;
const W_HARMONY_BONUS_WEIGHT: u32 = ${SCORE_W_INDEX.HARMONY_BONUS_WEIGHT}u;
const W_HARMONY_SUBSYSTEM_QUALITY_WEIGHT: u32 = ${SCORE_W_INDEX.HARMONY_SUBSYSTEM_QUALITY_WEIGHT}u;
const W_GOAL_UNLOCK_ORDERING_WEIGHT: u32 = ${SCORE_W_INDEX.GOAL_UNLOCK_ORDERING_WEIGHT}u;

fn pa(pb: u32, word: u32) -> f32 { return scoreParams[pb + word]; }
fn w(pb: u32, index: u32) -> f32 { return scoreParams[pb + PW + index]; }
fn isFiniteF(v: f32) -> bool { return v == v && abs(v) < INF * 0.5; }
fn clamp01(v: f32) -> f32 { return max(0.0, min(1.0, v)); }

// outcome.ts:45-72 / mirror.mjs:130-152 — band ladder with fractional roll.
// The thresholds are the exact JS-double ladder from buildScoreParams; f32
// multiplication by 1.3 drifts by one integer on every target in the corpus.
// x = guaranteed bands, y = bonus chance, z = next threshold (unused here).
fn scoreGetBonusAndChance(pb: u32, ladderBase: u32, value: f32) -> vec3<f32> {
  var currentTarget = pa(pb, ladderBase);
  if (currentTarget <= 0.0) { return vec3<f32>(0.0, 0.0, 0.0); }
  var remainingValue = value;
  var guaranteed = 0.0;
  var steps = 0u;
  while (remainingValue > 0.0 && currentTarget > 0.0 &&
         remainingValue >= currentTarget && steps < MAX_BAND_STEPS) {
    remainingValue = remainingValue - currentTarget;
    guaranteed = guaranteed + 1.0;
    steps = steps + 1u;
    currentTarget = pa(pb, ladderBase + steps);
  }
  let bonusChance = select(0.0, remainingValue / currentTarget, currentTarget > 0.0);
  return vec3<f32>(guaranteed, bonusChance, value + (currentTarget - remainingValue));
}

// outcome.ts:358-373 / mirror.mjs:178-183.
fn scoreTierForBands(cBands: f32, pBands: f32, hasSublime: bool) -> f32 {
  if (cBands == 0.0) { return TIER_FAILED; }
  if (cBands > 1.0 && pBands > 1.0 && hasSublime) { return TIER_SUBLIME; }
  if (pBands > 0.0) { return TIER_PERFECT; }
  return TIER_BASIC;
}

struct Bands {
  completionTarget: f32,
  perfectionTarget: f32,
  completionFinishFlat: f32,
  perfectionFinishFlat: f32,
  completionCapBandCount: f32, // -1 = undefined
  perfectionCapBandCount: f32,
  ambitionPerfectionBands: f32,
  ambitionCompletionCeilingBands: f32,
  targetTierRank: f32,
  reqCompletion: f32,
  reqPerfection: f32,
  hasSublimeOutcome: bool,
};

struct Outcome {
  tierRank: f32,
  optimisticTierRank: f32,
  completionBands: f32,
  perfectionBands: f32,
  completionBonusChance: f32,
  perfectionBonusChance: f32,
  completionMargin: f32,
  perfectionMargin: f32,
  completionShortfall: f32,
  perfectionShortfall: f32,
};

struct Goals {
  baseCompGoal: f32,
  basePerfGoal: f32,
  effectiveCompGoal: f32,
  effectivePerfGoal: f32,
  bands: Bands,
};

// outcome.ts:260-318, 248-254 / mirror.mjs:186-240. Exact band values are
// supplied by buildScoreParams (JS doubles); WGSL only assembles the struct.
fn scoreBuildOutcomeBands(pb: u32) -> Bands {
  let capC = pa(pb, P_MAX_C);
  let capP = pa(pb, P_MAX_P);
  var finishC = pa(pb, P_DERIVED_C);
  var finishP = pa(pb, P_DERIVED_P);
  if (capC > 0.0) { finishC = capC; }
  if (capP > 0.0) { finishP = capP; }
  return Bands(
    max(0.0, pa(pb, P_TARGET_C)), max(0.0, pa(pb, P_TARGET_P)),
    finishC, finishP,
    pa(pb, P_CAP_BANDS_C), pa(pb, P_CAP_BANDS_P),
    pa(pb, P_PBG), pa(pb, P_CBC),
    pa(pb, P_TIER_RANK), pa(pb, P_REQ_C), pa(pb, P_REQ_P),
    pa(pb, P_SUBLIME) > 0.5,
  );
}

// goals.ts:317-383 / mirror.mjs:250-302 — goals from the exact ladder.
fn scoreDeriveBandGoals(pb: u32) -> Goals {
  return Goals(
    pa(pb, P_BASE_C), pa(pb, P_BASE_P),
    pa(pb, P_EFF_C), pa(pb, P_EFF_P),
    scoreBuildOutcomeBands(pb),
  );
}

// outcome.ts:475-485 / mirror.mjs:305-309.
fn scoreRequirementProgress(guaranteed: f32, bonusChance: f32, required: f32) -> f32 {
  if (required <= 0.0) { return 1.0; }
  let achieved = guaranteed + clamp01(bonusChance);
  return clamp01(achieved / required);
}

// outcome.ts:382-472 / mirror.mjs:311-385.
fn scoreClassifyOutcome(pb: u32, completion: f32, perfection: f32, bands: Bands) -> Outcome {
  let comp = max(0.0, completion);
  let perf = max(0.0, perfection);
  var compBand = vec3<f32>(1.0, 0.0, 0.0);
  if (bands.completionTarget > 0.0) {
    compBand = scoreGetBonusAndChance(pb, P_LADDER_C, comp);
  }
  var perfBand = vec3<f32>(1.0, 0.0, 0.0);
  if (bands.perfectionTarget > 0.0) {
    perfBand = scoreGetBonusAndChance(pb, P_LADDER_P, perf);
  }
  let tier = scoreTierForBands(compBand.x, perfBand.x, bands.hasSublimeOutcome);
  let optimisticComp = compBand.x + select(0.0, 1.0, compBand.y > 0.0);
  let optimisticPerf = perfBand.x + select(0.0, 1.0, perfBand.y > 0.0);
  let optimistic = scoreTierForBands(optimisticComp, optimisticPerf, bands.hasSublimeOutcome);
  let reqC = select(0.0, bands.reqCompletion, bands.completionTarget > 0.0);
  let reqP = select(0.0, bands.reqPerfection, bands.perfectionTarget > 0.0);
  let compMargin = scoreRequirementProgress(compBand.x, compBand.y, reqC);
  let perfMargin = scoreRequirementProgress(perfBand.x, perfBand.y, reqP);
  let compShortfall = select(
    0.0, max(0.0, pa(pb, P_DERIVED_C) - comp), bands.completionTarget > 0.0);
  let perfShortfall = select(
    0.0, max(0.0, pa(pb, P_DERIVED_P) - perf), bands.perfectionTarget > 0.0);
  return Outcome(tier, optimistic, compBand.x, perfBand.x, compBand.y, perfBand.y,
    compMargin, perfMargin, compShortfall, perfShortfall);
}

// outcome.ts:553-601 / mirror.mjs:388-422. fractional is false in live scoreState and true in
// scoreFinishedOutcome, where the roll fraction is the final expected reward.
fn scoreOvercraftExtras(outcome: Outcome, bands: Bands, fractional: bool) -> vec2<f32> {
  let zero = vec2<f32>(0.0, 0.0);
  if (outcome.completionMargin < 1.0 || outcome.perfectionMargin < 1.0) { return zero; }
  let reqC = select(0.0, bands.reqCompletion, bands.completionTarget > 0.0);
  let reqP = select(0.0, bands.reqPerfection, bands.perfectionTarget > 0.0);
  let completionCount = outcome.completionBands +
    select(0.0, max(0.0, outcome.completionBonusChance), fractional);
  let perfectionCount = outcome.perfectionBands +
    select(0.0, max(0.0, outcome.perfectionBonusChance), fractional);
  var compCeiling = INF;
  if (bands.completionCapBandCount >= 0.0) { compCeiling = bands.completionCapBandCount; }
  let refundMax = select(reqC, 5.0, bands.hasSublimeOutcome); // outcome.ts:514
  compCeiling = min(compCeiling, refundMax);
  if (bands.ambitionCompletionCeilingBands >= 0.0) {
    compCeiling = min(compCeiling, bands.ambitionCompletionCeilingBands);
  }
  compCeiling = max(reqC, compCeiling);
  var perfCeiling = INF;
  if (bands.perfectionCapBandCount >= 0.0) { perfCeiling = bands.perfectionCapBandCount; }
  return vec2<f32>(
    max(0.0, min(completionCount, compCeiling) - reqC),
    max(0.0, min(perfectionCount, perfCeiling) - reqP),
  );
}

// scoring.ts:194-211 / mirror.mjs:425-438.
fn scoreBonusCredit(outcome: Outcome) -> f32 {
  if (outcome.optimisticTierRank <= outcome.tierRank) { return 0.0; }
  let completionFactor = select(
    clamp01(outcome.completionBonusChance), 1.0, outcome.completionMargin >= 1.0);
  let perfectionFactor = select(
    clamp01(outcome.perfectionBonusChance), 1.0, outcome.perfectionMargin >= 1.0);
  return clamp01(completionFactor * perfectionFactor);
}

// scoring.ts:217-238 / mirror.mjs:440-456.
fn scoreResidualShortfall(pb: u32, outcome: Outcome, bands: Bands) -> f32 {
  let compNeed = select(0.0, pa(pb, P_DERIVED_C), bands.completionTarget > 0.0);
  let perfNeed = select(0.0, pa(pb, P_DERIVED_P), bands.perfectionTarget > 0.0);
  let totalNeed = compNeed + perfNeed;
  if (totalNeed <= 0.0) { return 1.0; }
  let filled = max(0.0, compNeed - outcome.completionShortfall) +
    max(0.0, perfNeed - outcome.perfectionShortfall);
  return clamp01(filled / totalNeed);
}

// scoring.ts:240-255 / mirror.mjs:458-471.
fn scoreWeightedMarginAverage(c: f32, p: f32, cw: f32, pw: f32) -> f32 {
  let total = cw + pw;
  if (!(total > 0.0)) { return (c + p) * 0.5; }
  return (c * cw + p * pw) / total;
}

// scoring.ts:257-283 / mirror.mjs:473-491.
fn scoreExtraResolvedBands(outcome: Outcome, bands: Bands) -> f32 {
  let reqC = select(0.0, bands.reqCompletion, bands.completionTarget > 0.0);
  let reqP = select(0.0, bands.reqPerfection, bands.perfectionTarget > 0.0);
  let extraC = max(0.0, outcome.completionBands - reqC);
  let extraP = max(0.0, outcome.perfectionBands - reqP);
  if (reqC > 0.0 && reqP > 0.0) { return min(extraC, extraP); }
  if (reqC > 0.0) { return extraC; }
  if (reqP > 0.0) { return extraP; }
  return 0.0;
}

// scoring.ts:294-307 / mirror.mjs:493-502. countBasicCheckpoint=false in live
// scoring, true in terminal scoring (basic is a real runtime item floor).
fn scoreGoalTierRank(outcome: Outcome, targetTierRank: f32) -> f32 {
  if (outcome.tierRank == TIER_BASIC && targetTierRank >= TIER_PERFECT) {
    return TIER_FAILED;
  }
  return outcome.tierRank;
}

// scoring.ts:320-379 / mirror.mjs:504-551.
fn scoreConjunctiveGoal(pb: u32, outcome: Outcome, bands: Bands,
                        goals: Goals, completionWeight: f32, perfectionWeight: f32,
                        countBasic: bool, fractionalExtras: bool) -> f32 {
  let totalTargetMagnitude = max(1.0, max(0.0, goals.effectiveCompGoal) + max(0.0, goals.effectivePerfGoal));
  let baseTargetMagnitude = max(1.0, max(0.0, goals.baseCompGoal) + max(0.0, goals.basePerfGoal));
  let gate = min(outcome.completionMargin, outcome.perfectionMargin);
  let balance = scoreWeightedMarginAverage(outcome.completionMargin, outcome.perfectionMargin,
    completionWeight, perfectionWeight);
  let bonusCredit = scoreBonusCredit(outcome);
  let residual = scoreResidualShortfall(pb, outcome, bands);
  var extrasScore = 0.0;
  if (pa(pb, P_AMBITION) > 0.5) {
    let extras = scoreOvercraftExtras(outcome, bands, fractionalExtras);
    extrasScore = w(pb, W_EXTRA_BAND_WEIGHT) * extras.y +
      w(pb, W_COMPLETION_EXTRA_BAND_WEIGHT) * extras.x;
  } else {
    extrasScore = w(pb, W_EXTRA_BAND_WEIGHT) * scoreExtraResolvedBands(outcome, bands);
  }
  let scoredTierRank = select(
    scoreGoalTierRank(outcome, bands.targetTierRank), outcome.tierRank, countBasic);
  return totalTargetMagnitude * (
      w(pb, W_TIER_VALUE_SCALE) * scoredTierRank +
      w(pb, W_GATE_WEIGHT) * gate +
      w(pb, W_BALANCE_WEIGHT) * balance +
      w(pb, W_BONUS_ROLL_WEIGHT) * bonusCredit +
      w(pb, W_IN_TIER_PROGRESS_WEIGHT) * residual) +
    extrasScore * baseTargetMagnitude;
}

// scoringContext.ts:83-101 / mirror.mjs:554-564.
fn scoreEstimateProgressPerTurn(pb: u32, compShare: f32, perfShare: f32) -> f32 {
  let completionGain = max(1.0, pa(pb, P_CGAIN));
  let perfectionGain = max(1.0, pa(pb, P_PGAIN));
  let totalShare = max(0.0, compShare) + max(0.0, perfShare);
  if (totalShare <= 0.0) { return max(1.0, pa(pb, P_AGAIN)); }
  return (max(0.0, compShare) * completionGain + max(0.0, perfShare) * perfectionGain) / totalShare;
}

// scoringContext.ts:103-181 / mirror.mjs:566-620.
fn scoreEstimateTurnsRemaining(pb: u32, compRemaining: f32, perfRemaining: f32) -> f32 {
  let completionGain = max(1.0, pa(pb, P_CGAIN));
  let perfectionGain = max(1.0, pa(pb, P_PGAIN));
  if (compRemaining <= 0.0 && perfRemaining <= 0.0) { return 0.0; }
  if (compRemaining > 0.0 && perfRemaining <= 0.0) {
    return ceil(compRemaining / completionGain);
  }
  if (perfRemaining > 0.0 && compRemaining <= 0.0) {
    return ceil(perfRemaining / perfectionGain);
  }
  let count = min(u32(pa(pb, P_RCOUNT)), MAX_RATES);
  if (count == 0u) {
    return ceil(max(max(compRemaining / completionGain, perfRemaining / perfectionGain),
      (compRemaining + perfRemaining) / max(1.0, pa(pb, P_AGAIN))));
  }
  var turns = INF;
  for (var i = 0u; i < count; i = i + 1u) {
    let firstC = pa(pb, P_RATES + i * 2u);
    let firstP = pa(pb, P_RATES + i * 2u + 1u);
    turns = min(turns, max(
      select(INF, compRemaining / firstC, firstC > 0.0),
      select(INF, perfRemaining / firstP, firstP > 0.0)));
    for (var j = i + 1u; j < count; j = j + 1u) {
      let secondC = pa(pb, P_RATES + j * 2u);
      let secondP = pa(pb, P_RATES + j * 2u + 1u);
      let determinant = firstC * secondP - secondC * firstP;
      if (determinant == 0.0) { continue; }
      let firstTurns = (compRemaining * secondP - perfRemaining * secondC) / determinant;
      let secondTurns = (perfRemaining * firstC - compRemaining * firstP) / determinant;
      if (firstTurns >= 0.0 && secondTurns >= 0.0) {
        turns = min(turns, firstTurns + secondTurns);
      }
    }
  }
  if (turns < INF * 0.5) { return ceil(turns); }
  return ceil(compRemaining / completionGain + perfRemaining / perfectionGain);
}

// searchGoalPriority.ts:1-22 / mirror.mjs:96-104. Math.round is floor(x+0.5).
fn clampGoalPriorityBias(value: f32) -> f32 {
  if (!isFiniteF(value)) { return 0.0; }
  let snapped = floor(value / 25.0 + 0.5) * 25.0;
  return max(-100.0, min(100.0, snapped));
}

// scoring.ts:163-192 / mirror.mjs:107-126.
fn scoreGoalPriorityWeights(pb: u32, compNeedShare: f32, perfNeedShare: f32,
                            compGoal: f32, perfGoal: f32) -> vec2<f32> {
  var completionWeight = 1.0 + compNeedShare;
  var perfectionWeight = 1.0 + perfNeedShare;
  if (!(compGoal > 0.0 && perfGoal > 0.0)) {
    return vec2<f32>(completionWeight, perfectionWeight);
  }
  let normalizedBias = clampGoalPriorityBias(pa(pb, P_BIAS));
  if (normalizedBias == 0.0) {
    return vec2<f32>(completionWeight, perfectionWeight);
  }
  let shift = (normalizedBias / 100.0) * w(pb, W_GOAL_PRIORITY_WEIGHT_SHIFT);
  completionWeight = max(0.25, completionWeight + shift);
  perfectionWeight = max(0.25, perfectionWeight - shift);
  return vec2<f32>(completionWeight, perfectionWeight);
}

// harmonyScoring.ts:9-15 / mirror.mjs:691-693.
fn clampQuality(v: f32) -> f32 { return max(-1.0, min(1.0, v)); }
fn normMult(m: f32) -> f32 {
  if (m >= 1.5) { return 1.0; }
  if (m <= 0.0) { return -1.0; }
  return (m - 1.0) / 0.5;
}
// harmonyScoring.ts:17-30 / mirror.mjs:694-698.
fn needAlignment(op: f32, compShare: f32, perfShare: f32) -> f32 {
  if (op < 0.0) { return 0.0; }
  if (op < 0.5) { return compShare * 2.0 - 1.0; }
  if (op < 1.5) { return perfShare * 2.0 - 1.0; }
  return 0.0;
}
// harmony.ts:182-254 / mirror.mjs:635-648 — effective forge heat.
fn effectiveForgeHeat(heat: f32, lastHeat: f32) -> f32 {
  if (abs(heat - 1.0) < 0.5) { return lastHeat; }
  return heat;
}
fn forgeMods(heat: f32) -> vec2<f32> { // (control, intensity)
  if (heat >= 4.0 && heat <= 6.0) { return vec2<f32>(1.5, 1.5); }
  if (heat >= 2.0 && heat <= 3.0) { return vec2<f32>(0.5, 1.0); }
  if (heat >= 7.0 && heat <= 9.0) { return vec2<f32>(1.0, 0.5); }
  if (heat < 0.5) { return vec2<f32>(-9.0, 1.0); }
  if (heat >= 9.5) { return vec2<f32>(1.0, -9.0); }
  return vec2<f32>(1.0, 1.0);
}
// harmonyScoring.ts:32-83 / mirror.mjs:699-721.
fn genericQuality(control: f32, intensity: f32, crit: f32, success: f32,
                  qiRed: f32, stabRed: f32, compShare: f32, perfShare: f32) -> f32 {
  var weightedTotal = 0.0;
  var totalWeight = 0.0;
  if (isFiniteF(control) && isFiniteF(intensity) &&
      (intensity != 1.0 || control != 1.0)) {
    weightedTotal += compShare * normMult(intensity) + perfShare * normMult(control);
    totalWeight += 1.0;
  }
  if (isFiniteF(crit) && crit != 0.0) {
    weightedTotal += clampQuality(crit / 25.0);
    totalWeight += 1.0;
  }
  if (isFiniteF(success) && success != 0.0) {
    weightedTotal += clampQuality(success / 0.25);
    totalWeight += 1.0;
  }
  if (isFiniteF(qiRed) && qiRed != 0.0) {
    weightedTotal += clampQuality(qiRed / 25.0);
    totalWeight += 1.0;
  }
  if (isFiniteF(stabRed) && stabRed != 0.0) {
    weightedTotal += clampQuality(stabRed / 25.0);
    totalWeight += 1.0;
  }
  if (totalWeight <= 0.0) { return 0.0; }
  return clampQuality(weightedTotal / totalWeight);
}

// harmonyScoring.ts:99-238 / mirror.mjs:690-788.
fn scoreHarmonyQuality(base: u32, compShare: f32, perfShare: f32) -> f32 {
  let kind = states[base + S_HKIND];
  if (kind < 0.5) { return 0.0; }
  if (kind < 1.5) { // forge
    let rawHeat = states[base + S_HEAT];
    let mods = forgeMods(effectiveForgeHeat(rawHeat, states[base + S_LAST_HEAT]));
    let weighted = compShare * normMult(mods.y) + perfShare * normMult(mods.x);
    let turnsToSweetSpot = select(
      select(rawHeat - 6.0, ceil((4.0 - rawHeat) * 0.5), rawHeat < 4.0),
      0.0, rawHeat >= 4.0 && rawHeat <= 6.0);
    var correctionPressure = 0.0;
    if (rawHeat < 4.0) { correctionPressure = (turnsToSweetSpot * 0.5) * perfShare; }
    else if (rawHeat > 6.0) { correctionPressure = (turnsToSweetSpot * 0.25) * compShare; }
    return clampQuality(weighted - correctionPressure);
  }
  if (kind < 2.5) { // inscription
    let stackQuality = states[base + S_INS_STACKS] * 0.02 * 5.0;
    let blockProgress = max(0.0, (5.0 - states[base + S_INS_REMAIN]) * 0.2);
    return clampQuality(stackQuality + blockProgress * 0.3);
  }
  let generic = genericQuality(
    states[base + S_AC], states[base + S_AI], states[base + S_ACRIT],
    states[base + S_ASUCCESS], states[base + S_AQI], states[base + S_ASTAB],
    compShare, perfShare);
  if (kind < 3.5) { // alchemical
    if (states[base + S_CHARGES] <= 0.0 || states[base + S_REC_COUNT] <= 0.0) {
      return generic;
    }
    let setupProgress = clamp01(states[base + S_CHARGES] * 0.5);
    var best = -INF;
    for (var i = 0u; i < min(u32(states[base + S_REC_COUNT]), MAX_RECOMMENDED); i = i + 1u) {
      best = max(best, needAlignment(states[base + S_REC + i], compShare, perfShare));
    }
    let setupQuality = clampQuality(setupProgress * best);
    if (generic == 0.0) { return setupQuality; }
    return clampQuality((generic + setupQuality) * 0.5);
  }
  if (kind < 4.5) { // resonance
    let strength = states[base + S_RES_STRENGTH];
    let resonanceModQuality = genericQuality(1.0, 1.0, strength * 3.0,
      strength * 0.03, 0.0, 0.0, compShare, perfShare);
    let currentType = states[base + S_RES_TYPE];
    let currentAlignment = needAlignment(currentType, compShare, perfShare);
    var quality = select(resonanceModQuality,
      clampQuality((resonanceModQuality + currentAlignment) * 0.5), currentType >= 0.0);
    let pendingType = states[base + S_PENDING_TYPE];
    if (pendingType >= 0.0 && states[base + S_PENDING_COUNT] > 0.0 && pendingType != currentType) {
      let switchWeight = max(0.2, min(0.8, 1.0 - (states[base + S_PENDING_COUNT] - 1.0) * 0.25));
      let switchAlignment = clampQuality(
        currentAlignment * (1.0 - switchWeight) +
        needAlignment(pendingType, compShare, perfShare) * switchWeight);
      quality = clampQuality((resonanceModQuality + switchAlignment) * 0.5);
    }
    return quality;
  }
  // captivating cadence: chain -> both multipliers
  let bonus = states[base + S_CADENCE] * 0.02;
  return genericQuality(1.0 + bonus, 1.0 + bonus, 0.0, 0.0, 0.0, 0.0, compShare, perfShare);
}

// scoring.ts:393-732 / mirror.mjs:801-1129 — the full engine score.
fn scoreStateWgsl(idx: u32) -> f32 {
  let base = idx * SI;
  let pb = idx * PS;
  let targetCompletion = pa(pb, P_TARGET_C);
  let targetPerfection = pa(pb, P_TARGET_P);
  let completion = states[base + S_COMPLETION];
  let perfection = states[base + S_PERFECTION];
  // scoring.ts:408-410
  if (targetCompletion == 0.0 && targetPerfection == 0.0) {
    return min(completion, perfection);
  }
  let goals = scoreDeriveBandGoals(pb);
  let bands = goals.bands;
  let outcome = scoreClassifyOutcome(pb, completion, perfection, bands);
  // scoring.ts:432-437
  let baseTargetsMet =
    (goals.baseCompGoal <= 0.0 || completion >= goals.baseCompGoal) &&
    (goals.basePerfGoal <= 0.0 || perfection >= goals.basePerfGoal);
  let modeTargetsMet =
    (goals.effectiveCompGoal <= 0.0 || completion >= goals.effectiveCompGoal) &&
    (goals.effectivePerfGoal <= 0.0 || perfection >= goals.effectivePerfGoal);

  // scoring.ts:439-471
  let compRemaining = max(0.0, goals.effectiveCompGoal - completion);
  let perfRemaining = max(0.0, goals.effectivePerfGoal - perfection);
  let totalRemaining = compRemaining + perfRemaining;
  let compNeedShare = select(0.5, compRemaining / totalRemaining, totalRemaining > 0.0);
  let perfNeedShare = select(0.5, perfRemaining / totalRemaining, totalRemaining > 0.0);
  let compNeedPct = select(0.0, clamp01(compRemaining / goals.effectiveCompGoal),
    goals.effectiveCompGoal > 0.0);
  let perfNeedPct = select(0.0, clamp01(perfRemaining / goals.effectivePerfGoal),
    goals.effectivePerfGoal > 0.0);
  let remainingWorkPct = max(max(compNeedPct, perfNeedPct), (compNeedPct + perfNeedPct) * 0.5);
  let bindingRemainingWork = max(compNeedPct, perfNeedPct);
  let estimatedProgressPerTurn = scoreEstimateProgressPerTurn(pb, compNeedShare, perfNeedShare);
  let estimatedTurnsRemaining = scoreEstimateTurnsRemaining(pb, compRemaining, perfRemaining);

  // scoring.ts:473-503
  let baseCompRemaining = select(0.0,
    max(0.0, goals.baseCompGoal - min(completion, goals.baseCompGoal)), goals.baseCompGoal > 0.0);
  let basePerfRemaining = select(0.0,
    max(0.0, goals.basePerfGoal - min(perfection, goals.basePerfGoal)), goals.basePerfGoal > 0.0);
  let baseTotalRemaining = baseCompRemaining + basePerfRemaining;
  let baseCompNeedShare = select(0.5, baseCompRemaining / baseTotalRemaining, baseTotalRemaining > 0.0);
  let basePerfNeedShare = select(0.5, basePerfRemaining / baseTotalRemaining, baseTotalRemaining > 0.0);
  let baseCompNeedPct = select(0.0, clamp01(baseCompRemaining / goals.baseCompGoal),
    goals.baseCompGoal > 0.0);
  let basePerfNeedPct = select(0.0, clamp01(basePerfRemaining / goals.basePerfGoal),
    goals.basePerfGoal > 0.0);
  let baseRemainingWorkPct = max(max(baseCompNeedPct, basePerfNeedPct),
    (baseCompNeedPct + basePerfNeedPct) * 0.5);
  let baseEstimatedTurnsRemaining = scoreEstimateTurnsRemaining(pb, baseCompRemaining, basePerfRemaining);
  // scoring.ts:504-507
  let stepPenaltyWeight = max(w(pb, W_STEP_PENALTY),
    estimatedProgressPerTurn * w(pb, W_STEP_PENALTY_PROGRESS_FRACTION));

  // scoring.ts:509-520
  let priority = scoreGoalPriorityWeights(pb, compNeedShare, perfNeedShare,
    goals.effectiveCompGoal, goals.effectivePerfGoal);
  let totalPriorityWeight = priority.x + priority.y;
  let completionPriorityShare = select(0.5, priority.x / totalPriorityWeight, totalPriorityWeight > 0.0);
  let perfectionPriorityShare = select(0.5, priority.y / totalPriorityWeight, totalPriorityWeight > 0.0);

  // scoring.ts:532-543
  var score = scoreConjunctiveGoal(pb, outcome, bands, goals, priority.x, priority.y,
    false, false);
  let totalTargetMagnitude = max(1.0,
    max(0.0, goals.effectiveCompGoal) + max(0.0, goals.effectivePerfGoal));
  let baseTargetMagnitude = max(1.0,
    max(0.0, goals.baseCompGoal) + max(0.0, goals.basePerfGoal));
  let resourceRemainingWorkPct = select(baseRemainingWorkPct, remainingWorkPct, baseTargetsMet);
  let resourceEstimatedTurnsRemaining = select(baseEstimatedTurnsRemaining,
    estimatedTurnsRemaining, baseTargetsMet);
  let resourceEstimatedProgressPerTurn = select(
    scoreEstimateProgressPerTurn(pb, baseCompNeedShare, basePerfNeedShare),
    estimatedProgressPerTurn, baseTargetsMet);
  let survivabilityTargetMagnitude = select(baseTargetMagnitude, totalTargetMagnitude, baseTargetsMet);

  // scoring.ts:571-575
  if (modeTargetsMet) {
    score += states[base + S_QI] * w(pb, W_RESOURCE_TIEBREAKER) +
      states[base + S_STABILITY] * w(pb, W_RESOURCE_TIEBREAKER) -
      states[base + S_STEP] * stepPenaltyWeight;
  }

  if (!modeTargetsMet) {
    // scoring.ts:579-598 — state.ts:469/476 gates
    if (states[base + S_CONTROL_TURNS] > 0.0) {
      let controlBuffBoost = (states[base + S_CONTROL_MULT] - 1.0) *
        w(pb, W_BUFF_VALUE_PER_MULTIPLIER_POINT);
      score += states[base + S_CONTROL_TURNS] * controlBuffBoost *
        (w(pb, W_BUFF_NEED_FLOOR) + perfectionPriorityShare) * remainingWorkPct;
    }
    if (states[base + S_INTENSITY_TURNS] > 0.0) {
      let intensityBuffBoost = (states[base + S_INTENSITY_MULT] - 1.0) *
        w(pb, W_BUFF_VALUE_PER_MULTIPLIER_POINT);
      score += states[base + S_INTENSITY_TURNS] * intensityBuffBoost *
        (w(pb, W_BUFF_NEED_FLOOR) + completionPriorityShare) * remainingWorkPct;
    }
    // scoring.ts:601-609
    let qiCost = pa(pb, P_QCOST);
    if (qiCost > 0.0 && estimatedTurnsRemaining > 0.0) {
      let estimatedQiNeeded = resourceEstimatedTurnsRemaining * qiCost;
      let qiShortfall = max(0.0, estimatedQiNeeded - states[base + S_QI]);
      if (qiShortfall > 0.0) {
        score -= (qiShortfall / qiCost) * resourceEstimatedProgressPerTurn;
      }
    }
    // scoring.ts:611-616
    score += states[base + S_STABILITY] *
      (w(pb, W_STABILITY_BASE_WEIGHT) + resourceRemainingWorkPct * w(pb, W_STABILITY_WORK_WEIGHT));
    score -= states[base + S_STEP] * stepPenaltyWeight;
  }

  // scoring.ts:626-644
  let modeCompLimit = goals.effectiveCompGoal;
  let modePerfLimit = goals.effectivePerfGoal;
  let compOver = select(0.0, max(0.0, completion - modeCompLimit), modeCompLimit > 0.0);
  let perfOver = select(0.0, max(0.0, perfection - modePerfLimit), modePerfLimit > 0.0);
  score -= (compOver + perfOver) * w(pb, W_OVERSHOOT_PENALTY_WEIGHT);
  if (pa(pb, P_HAS_MAX_C) > 0.5) {
    score -= max(0.0, completion - pa(pb, P_MAX_C)) * w(pb, W_HARD_CAP_PENALTY_WEIGHT);
  }
  if (pa(pb, P_HAS_MAX_P) > 0.5) {
    score -= max(0.0, perfection - pa(pb, P_MAX_P)) * w(pb, W_HARD_CAP_PENALTY_WEIGHT);
  }

  // scoring.ts:651-701
  if (!modeTargetsMet) {
    let training = pa(pb, P_TRAINING) > 0.5;
    let thresholdBase = select(w(pb, W_STABILITY_THRESHOLD_TURNS_BASE),
      w(pb, W_STABILITY_THRESHOLD_TURNS_BASE_TRAINING), training);
    let thresholdScale = select(w(pb, W_STABILITY_THRESHOLD_TURNS_SCALE),
      w(pb, W_STABILITY_THRESHOLD_TURNS_SCALE_TRAINING), training);
    let stabilityThreshold =
      (thresholdBase + resourceRemainingWorkPct * thresholdScale) * pa(pb, P_SCOST);
    let penaltyFraction = select(w(pb, W_STABILITY_PENALTY_FRACTION),
      w(pb, W_STABILITY_PENALTY_FRACTION_TRAINING), training);
    let penaltyFloor = select(w(pb, W_STABILITY_PENALTY_FLOOR),
      w(pb, W_STABILITY_PENALTY_FLOOR_TRAINING), training);
    let stabilityPenaltyWeight = max(penaltyFloor, survivabilityTargetMagnitude * penaltyFraction);
    let stability = states[base + S_STABILITY];
    if (stability < stabilityThreshold) {
      let risk = (stabilityThreshold - stability) / stabilityThreshold;
      score -= risk * risk * stabilityPenaltyWeight;
    }
    if (stability <= 0.0) {
      score -= totalTargetMagnitude * w(pb, W_DEATH_PENALTY_MULTIPLIER);
    } else if (stability <= w(pb, W_NEAR_DEATH_STABILITY)) {
      score -= (w(pb, W_NEAR_DEATH_STABILITY) - stability) * stabilityPenaltyWeight;
    }
    let stabilityCost = pa(pb, P_SCOST);
    let estimatedRunwayTurns = select(INF,
      floor(max(0.0, stability) / stabilityCost), stabilityCost > 0.0);
    if (resourceEstimatedTurnsRemaining > estimatedRunwayTurns) {
      let gap = resourceEstimatedTurnsRemaining - estimatedRunwayTurns;
      let gapFraction = select(w(pb, W_RUNWAY_GAP_FRACTION),
        w(pb, W_RUNWAY_GAP_FRACTION_TRAINING), training);
      score -= gap * survivabilityTargetMagnitude * gapFraction;
    }
  }

  // scoring.ts:704-706 — state.ts:497-498
  if (states[base + S_MAX_TOXICITY] > 0.0 &&
      states[base + S_TOXICITY] >= states[base + S_MAX_TOXICITY] * 0.8) {
    score -= totalTargetMagnitude * w(pb, W_TOXICITY_PENALTY_FRACTION);
  }
  // scoring.ts:707-729
  if (pa(pb, P_SUBLIME) > 0.5 && !modeTargetsMet) {
    let normalizedHarmony = clampQuality(states[base + S_HARMONY] * 0.01);
    score += normalizedHarmony * bindingRemainingWork * totalTargetMagnitude *
      w(pb, W_HARMONY_BONUS_WEIGHT);
    if (states[base + S_HKIND] > 0.5) {
      let quality = scoreHarmonyQuality(base, completionPriorityShare, perfectionPriorityShare);
      score += quality * remainingWorkPct * totalTargetMagnitude *
        w(pb, W_HARMONY_SUBSYSTEM_QUALITY_WEIGHT);
    }
  }
  return score;
}

// scoring.ts:749-846 — terminal states price the real runtime tier (basic
// counts) and the bonus-roll fraction as the final expected reward. All
// live-only layers (buffs, resource value, survivability, toxicity, harmony,
// overshoot, hard caps) are intentionally absent.
fn scoreFinishedOutcomeWgsl(idx: u32) -> f32 {
  let base = idx * SI;
  let pb = idx * PS;
  let targetCompletion = pa(pb, P_TARGET_C);
  let targetPerfection = pa(pb, P_TARGET_P);
  let completion = states[base + S_COMPLETION];
  let perfection = states[base + S_PERFECTION];
  if (targetCompletion == 0.0 && targetPerfection == 0.0) {
    return min(completion, perfection);
  }
  let goals = scoreDeriveBandGoals(pb);
  let bands = goals.bands;
  let outcome = scoreClassifyOutcome(pb, completion, perfection, bands);
  let compRemaining = max(0.0, goals.effectiveCompGoal - completion);
  let perfRemaining = max(0.0, goals.effectivePerfGoal - perfection);
  let totalRemaining = compRemaining + perfRemaining;
  let compNeedShare = select(0.5, compRemaining / totalRemaining, totalRemaining > 0.0);
  let perfNeedShare = select(0.5, perfRemaining / totalRemaining, totalRemaining > 0.0);
  let priority = scoreGoalPriorityWeights(pb, compNeedShare, perfNeedShare,
    goals.effectiveCompGoal, goals.effectivePerfGoal);
  var score = scoreConjunctiveGoal(pb, outcome, bands, goals, priority.x, priority.y,
    true, true);
  let totalTargetMagnitude = max(1.0,
    max(0.0, goals.effectiveCompGoal) + max(0.0, goals.effectivePerfGoal));
  let targetGoalsMet = outcome.completionMargin >= 1.0 && outcome.perfectionMargin >= 1.0;
  if (states[base + S_STABILITY] <= 0.0 && !targetGoalsMet) {
    score -= totalTargetMagnitude * w(pb, W_DEATH_PENALTY_MULTIPLIER);
  }
  score -= (compRemaining + perfRemaining) * w(pb, W_FINISHED_UNMET_PENALTY_WEIGHT);
  let stepPenaltyWeight = max(w(pb, W_STEP_PENALTY),
    pa(pb, P_AGAIN) * w(pb, W_STEP_PENALTY_PROGRESS_FRACTION));
  return score - states[base + S_STEP] * stepPenaltyWeight;
}

// lookahead.ts:279-329 — live score plus the condition-effects potential term.
// The multipliers are config-derived, so they arrive as param deltas; this is
// the only place scoreStateWithTerminalPenalty differs from scoreState.
fn conditionPotentialWgsl(idx: u32) -> f32 {
  let base = idx * SI;
  let pb = idx * PS;
  let targetCompletion = pa(pb, P_TARGET_C);
  let targetPerfection = pa(pb, P_TARGET_P);
  if (targetCompletion == 0.0 && targetPerfection == 0.0) { return 0.0; }
  let goals = scoreDeriveBandGoals(pb);
  let compRemaining = max(0.0, goals.effectiveCompGoal - states[base + S_COMPLETION]);
  let perfRemaining = max(0.0, goals.effectivePerfGoal - states[base + S_PERFECTION]);
  let totalRemaining = compRemaining + perfRemaining;
  if (!(totalRemaining > 0.0)) { return 0.0; }
  let compNeedShare = compRemaining / totalRemaining;
  let perfNeedShare = perfRemaining / totalRemaining;
  let priority = scoreGoalPriorityWeights(pb, compNeedShare, perfNeedShare,
    goals.effectiveCompGoal, goals.effectivePerfGoal);
  let totalPriorityWeight = priority.x + priority.y;
  let completionPriorityShare = select(0.5, priority.x / totalPriorityWeight,
    totalPriorityWeight > 0.0);
  let perfectionPriorityShare = select(0.5, priority.y / totalPriorityWeight,
    totalPriorityWeight > 0.0);
  let conditionedPotential =
    completionPriorityShare * (1.0 + pa(pb, P_COND_INTENSITY)) +
    perfectionPriorityShare * (1.0 + pa(pb, P_COND_CONTROL));
  let neutralPotential = completionPriorityShare + perfectionPriorityShare;
  let conditionWeightedProgress =
    scoreEstimateProgressPerTurn(pb, completionPriorityShare, perfectionPriorityShare);
  return (conditionedPotential - neutralPotential) * conditionWeightedProgress;
}

fn scoreLiveWithConditionWgsl(idx: u32) -> f32 {
  return scoreStateWgsl(idx) + conditionPotentialWgsl(idx);
}

// ── S2: terminality decided on the device where the bars already live ──────
// outcome.ts:494-505 — willAutoFinish(state, bands). No availability needed:
// stability <= 0 || (completion >= finishFlat && perfection >= finishFlat).
// bands.completionFinishFlat / perfectionFinishFlat are the same exact
// JS-double finish flats buildScoreParams already supplies for the band ladder.
fn deviceWillAutoFinish(pb: u32, idx: u32) -> bool {
  let base = idx * SI;
  let bands = scoreBuildOutcomeBands(pb);
  let stability = states[base + S_STABILITY];
  let completion = states[base + S_COMPLETION];
  let perfection = states[base + S_PERFECTION];
  return stability <= 0.0 ||
    (completion >= bands.completionFinishFlat && perfection >= bands.perfectionFinishFlat);
}

fn terminalHostBits(pb: u32) -> u32 { return u32(pa(pb, P_TERMINAL)); }

// lookahead.ts:244-247 — scoreStateWithTerminalPenalty:
//   stateWillAutoFinish(candidate) || (knownBlocked ?? isTerminalState(...))
// Only the second operand needs availability and it arrives as host bit 0.
// Bit 1 forces the harness's finished-outcome pricing; bit 2 is the N8 fault
// injection and never set in a normal run.
fn deviceTerminalMode(pb: u32, idx: u32, bits: u32) -> bool {
  if ((bits & 4u) != 0u) { return false; }
  return deviceWillAutoFinish(pb, idx) ||
    (bits & 1u) != 0u || (bits & 2u) != 0u;
}

// goals.ts:25-43 — goalsMet(state, targetCompletion, targetPerfection) on the
// raw (unclamped) bar values; non-finite goals were normalised to 0 at export.
fn deviceGoalsMet(pb: u32, completion: f32, perfection: f32) -> bool {
  let tc = pa(pb, P_TARGET_C);
  let tp = pa(pb, P_TARGET_P);
  let hasC = tc > 0.0;
  let hasP = tp > 0.0;
  if (!hasC && !hasP) { return false; }
  return (!hasC || completion >= tc) && (!hasP || perfection >= tp);
}

// Guided path: the exact S1 decision (host bit as a plain 1/0). Keeping this
// entry untouched is what makes the guided baseline byte-identical; strict
// uses scoreWithTerminalityKernel below, where the device decides.
fn scoreStateWithTerminalPenaltyWgsl(idx: u32) -> f32 {
  if (pa(idx * PS, P_TERMINAL) > 0.5) { return scoreFinishedOutcomeWgsl(idx); }
  return scoreLiveWithConditionWgsl(idx);
}

@compute @workgroup_size(64)
fn scoreStateKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&scores)) { return; }
  scores[idx] = scoreStateWithTerminalPenaltyWgsl(idx);
}

/**
 * S2 stand entry: same score, plus a five-word terminality tuple per case:
 *   0 autoFinish      — device, outcome.ts:494-505
 *   1 isTerminal      — HOST half, availability-derived (skills.ts:6104-6110)
 *   2 endsCraft       — device, lookahead.ts:1537 stateWillAutoFinish(newState)
 *   3 terminalMode    — device combination feeding the scorer
 *   4 belowTarget     — (endsCraft || isTerminal) && !goalsMet, lookahead.ts:1577-1579
 * Keeping the host half in its own word is the named boundary: the stand must
 * never present word 1 as a device decision (S3 moves it to the device).
 */
@compute @workgroup_size(64)
fn scoreWithTerminalityKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = arrayLength(&scores);
  if (idx >= count) { return; }
  let base = idx * SI;
  let pb = idx * PS;
  let bits = terminalHostBits(pb);
  let autoFinish = deviceWillAutoFinish(pb, idx);
  let hostTerminal = (bits & 1u) != 0u;
  let forced = (bits & 2u) != 0u;
  let combined = deviceTerminalMode(pb, idx, bits);
  let belowTarget = combined &&
    !deviceGoalsMet(pb, states[base + S_COMPLETION], states[base + S_PERFECTION]);
  let t = idx * ${TERMINALITY_WORDS}u;
  terminality[t] = select(0.0, 1.0, autoFinish);
  terminality[t + 1u] = select(0.0, 1.0, hostTerminal);
  terminality[t + 2u] = select(0.0, 1.0, autoFinish);
  terminality[t + 3u] = select(0.0, 1.0, combined);
  terminality[t + 4u] = select(0.0, 1.0, belowTarget);
  scores[idx] = select(scoreLiveWithConditionWgsl(idx),
    scoreFinishedOutcomeWgsl(idx), combined);
}

/** Stand-only readout: base scoreStateWgsl and condition potential, unmixed. */
@compute @workgroup_size(64)
fn scoreSplitKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&scores)) { return; }
  let total = scores[idx];
  scores[idx] = total; // keep the read_write binding in this entry's auto layout
  let t = idx * ${TERMINALITY_WORDS}u;
  terminality[t] = terminality[t]; // keep binding 3 aligned with the stand's group
  let base = scoreStateWgsl(idx);
  split[idx * 8u] = base;
  split[idx * 8u + 1u] = conditionPotentialWgsl(idx);
  let hb = idx * SI;
  split[idx * 8u + 2u] = states[hb + S_HKIND];
  split[idx * 8u + 3u] = states[hb + S_HEAT];
  split[idx * 8u + 4u] = states[hb + S_LAST_HEAT];
  split[idx * 8u + 5u] = states[hb + S_HARMONY];
  let pbb = idx * PS;
  split[idx * 8u + 6u] = scoreHarmonyQuality(hb, 0.5, 0.5);
  split[idx * 8u + 7u] = pa(pbb, P_SUBLIME);
}
`;

/** Compile the scoring kernel and fail loudly on any WGSL error. */
export async function createScoreKernelPipeline(
  device: GpuDevice,
  entryPoint: 'scoreStateKernel' | 'scoreWithTerminalityKernel' | 'scoreSplitKernel' = 'scoreStateKernel',
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: SCORE_KERNEL_WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `WGSL: ${errors.map((error) => `${error.lineNum}:${error.message}`).join(' | ')}`,
    );
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
  });
}
