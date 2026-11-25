/**
 * Standalone device-resident beam primitives (S6/S7 slice, no production wiring).
 *
 * One level of expansion is encoded into a single GpuCommandEncoder and
 * submitted once: there is no host construction of per-case SCORE_INPUT and no
 * await between dispatches. Buffers are ping-ponged across levels. See
 * docs/GPU_RESIDENT_BEAM.md for the design; this file deliberately does NOT
 * touch runStatefulSearch / runRecommendationSearch, the engine, the compact
 * kernel or the existing stateful kernels.
 *
 * Residual (documented in the design, step 7): expected/success skill gains are
 * still host-derived tables, uploaded once per search as a constant table keyed
 * by (node, skill, conditionKind). They are not per-case score inputs.
 */

import {
  COND_PARAM_WORDS,
  STATE_PARAM_WORDS,
  BLEND_IN_WORDS,
  P_COND_POOL_PEN,
  P_COND_POOL_RED,
  P_COND_STAB_PEN,
  P_COND_STAB_RED,
  finalKernelSources,
} from './statefulSearch';
import { NS_STATE } from './records';
import {
  HARMONY_CHARGE_CAPACITY,
  HARMONY_RECOMMEND_CAPACITY,
} from './records';
import {
  SCORE_INPUT,
  SCORE_PARAM,
  TERMINALITY_WORDS,
  createScoreKernelPipeline,
} from './kernelScore';
import {
  AVAIL_PARAM,
  AVAIL_SKILL,
  AVAIL_OUT,
  createAvailabilityKernelPipeline,
} from './availability';
import { VERDICT_IN, VERDICT_OUT, createVerdictPipeline } from './verdict';
import { STYLE, createScoreInputPipeline } from './scoreInputKernel';

export const BEAM = 20;
export const SKILL_CAPACITY = 64;
export const MAX_BRANCHES = 5;
export const DEPTH_MAX = 96;
export const CHILD = BEAM * SKILL_CAPACITY;
export const CASES = CHILD * MAX_BRANCHES + CHILD * 2;
export const CONDITION_KINDS = 6;
export const COND_EFFECT_WORDS = 6;
export const GAIN_WORDS = 7;
export const META_WORDS = 8;
export const SCORE_TABLE_VARIANTS = CONDITION_KINDS * 2;

/** meta words: branchProbability, rootId, parentId, skillIndex, depth, alive, conditionKind, branchIndex */
export const META = {
  probability: 0,
  rootId: 1,
  parentId: 2,
  skill: 3,
  depth: 4,
  alive: 5,
  conditionKind: 6,
  branch: 7,
  WORDS: 8,
} as const;

/** Host-derived gain table: [expectedC, expectedP, successC, successP, terminal, survival, displayEnds]. */
export const GAIN = {
  expectedCompletion: 0,
  expectedPerfection: 1,
  successCompletion: 2,
  successPerfection: 3,
  terminal: 4,
  survival: 5,
  displayEnds: 6,
} as const;

/** Constants appended to the immutable STYLE buffer, one upload per search. */
export const X = {
  discordance: STYLE.WORDS + 0,
  branching: STYLE.WORDS + 1,
  branchLimit: STYLE.WORDS + 2,
  minProbability: STYLE.WORDS + 3,
  modeCompGoal: STYLE.WORDS + 4,
  modePerfGoal: STYLE.WORDS + 5,
  targetCompletion: STYLE.WORDS + 6,
  targetPerfection: STYLE.WORDS + 7,
  initialMaxStability: STYLE.WORDS + 8,
  nearDeath: STYLE.WORDS + 9,
  minSafe: STYLE.WORDS + 10,
  hasSafe: STYLE.WORDS + 11,
  WORDS: STYLE.WORDS + 12,
} as const;

const COND_RECORD_WORDS = 1 + 5 * 5;
const TRACE_WORDS = DEPTH_MAX * BEAM * 2;

// ---------------------------------------------------------------------------
// New WGSL. Existing kernels are imported/compiled verbatim, never patched.
// ---------------------------------------------------------------------------

const fn = (name: string): string => `${name}Kernel`;

/** Step 1: (node, skill) -> three parameter variants (EV/success/failure). */
export const PREPARE_STATE_PARAMS_WGSL = `
@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read> nodeMeta: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
@group(0) @binding(3) var<storage, read> condEffects: array<f32>;
@group(0) @binding(4) var<storage, read_write> evOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> successOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> failureOut: array<f32>;

fn combine(reduction: bool, current: f32, contribution: f32) -> f32 {
  if (reduction) {
    return 100.0 - ((100.0 - min(100.0, current)) * (100.0 - min(100.0, contribution))) / 100.0;
  }
  return ((100.0 + current) * (100.0 + contribution)) / 100.0 - 100.0;
}

@compute @workgroup_size(64)
fn ${fn('prepareStateParams')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let child = gid.x;
  if (child >= ${CHILD}u) { return; }
  let node = child / ${SKILL_CAPACITY}u;
  let skill = child % ${SKILL_CAPACITY}u;
  var p: array<f32, ${STATE_PARAM_WORDS}>;
  for (var q = 0u; q < ${STATE_PARAM_WORDS}u; q = q + 1u) { p[q] = base[q]; }
  let nb = node * ${NS_STATE.WORDS}u;
  let kind = u32(nodeMeta[node * ${META.WORDS}u + ${META.conditionKind}u]);
  let eb = kind * ${COND_EFFECT_WORDS}u;
  p[0] = f32(skill);
  p[1] = nodes[nb + ${NS_STATE.conditionId}u];
  p[2] = condEffects[eb + 0u];
  p[3] = condEffects[eb + 1u];
  p[${P_COND_POOL_RED}u] = condEffects[eb + 2u];
  p[${P_COND_POOL_PEN}u] = condEffects[eb + 3u];
  p[${P_COND_STAB_RED}u] = condEffects[eb + 4u];
  p[${P_COND_STAB_PEN}u] = condEffects[eb + 5u];
  p[12] = nodes[nb + ${NS_STATE.harmonyDerivedMask}u];
  let htype = base[6];
  if (htype > 0.5 && htype < 1.5) {
    var heat = nodes[nb + ${NS_STATE.harmonyForgeHeat}u];
    if (abs(heat - 1.0) < 0.001) {
      let last = nodes[nb + ${NS_STATE.harmonyForgeLastBuffedHeat}u];
      if (last >= 0.0) { heat = last; }
    }
    heat = clamp(floor(heat), 0.0, 10.0);
    var control = 1.0;
    var intensity = 1.0;
    if (heat >= 4.0 && heat <= 6.0) { control = 1.5; intensity = 1.5; }
    else if (heat >= 2.0 && heat <= 3.0) { control = 0.5; }
    else if (heat >= 7.0 && heat <= 9.0) { intensity = 0.5; }
    else if (heat < 0.5) { control = -9.0; }
    else if (heat > 9.5) { intensity = -9.0; }
    p[7] = control;
    p[8] = intensity;
  } else if (htype > 1.5 && htype < 2.5) {
    // HARMONY_REACTION_KEYS order: control, intensity, crit, qiReduction,
    // stabilityReduction, successBonus (records.ts:2130).
    let rb = nb + ${NS_STATE.harmonyReactionMods}u;
    p[7] = nodes[rb + 0u];
    p[8] = nodes[rb + 1u];
    p[9] = nodes[rb + 2u];
    p[10] = nodes[rb + 3u];
    p[11] = nodes[rb + 4u];
    p[29] = nodes[rb + 5u];
  } else if (htype > 2.5 && htype < 3.5) {
    let strength = nodes[nb + ${NS_STATE.harmonyResonanceStrength}u];
    p[9] = strength * 3.0;
    p[29] = strength * 0.03;
  }
  let o = child * ${STATE_PARAM_WORDS}u;
  p[31] = 0.0;
  for (var q = 0u; q < ${STATE_PARAM_WORDS}u; q = q + 1u) { evOut[o + q] = p[q]; }
  p[31] = 1.0;
  for (var q = 0u; q < ${STATE_PARAM_WORDS}u; q = q + 1u) { successOut[o + q] = p[q]; }
  p[31] = 2.0;
  for (var q = 0u; q < ${STATE_PARAM_WORDS}u; q = q + 1u) { failureOut[o + q] = p[q]; }
}
`;

/**
 * Step 2: (post-action child node) + immutable config words -> COND_PARAM_WORDS
 * per child. The engine generates condition transitions from the post-action
 * state's generation harmony (lookahead.ts:588-591, 986-989); reading the parent
 * Records node here made every branch use the root harmony instead.
 */
export const PREPARE_COND_PARAMS_WGSL = `
@group(0) @binding(0) var<storage, read> evNodes: array<f32>;
@group(0) @binding(1) var<storage, read> style: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn ${fn('prepareCondParams')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let child = gid.x;
  if (child >= ${CHILD}u) { return; }
  let cb = child * ${NS_STATE.WORDS}u;
  let ob = child * ${COND_PARAM_WORDS}u;
  out[ob + 0u] = evNodes[cb + ${NS_STATE.conditionGenerationHarmony}u];
  out[ob + 1u] = style[${X.discordance}u];
  out[ob + 2u] = style[${X.branching}u];
  out[ob + 3u] = style[${X.branchLimit}u];
  out[ob + 4u] = style[${X.minProbability}u];
  out[ob + 5u] = evNodes[cb + ${NS_STATE.conditionId}u];
}
`;

/**
 * Per-case harmony snapshot rebuilt from the POST-ACTION node on device.
 *
 * The engine scores `candidate` (lookahead.ts:264-278), i.e. the state after
 * the action, and generates the next condition transitions from that state's
 * `conditionGenerationHarmony ?? harmony` (lookahead.ts:986-990,
 * conditions.ts:90-168). The host `styleCases` upload is keyed by ROOT skill
 * only: at level 2 `child = node*SKILL_CAPACITY + skill` uses a beam slot as
 * `node`, so every child inherited the root's forge heat and harmony kind
 * (first divergence: `forgeHeat: device 4 vs host 3`).
 *
 * Uncarried words keep the host base value; every word NS_STATE carries is
 * overwritten from `caseNodes[case]`. Runs once per level before scoreInput.
 */
export const PREPARE_STYLE_CASES_WGSL = `
@group(0) @binding(0) var<storage, read> caseNodes: array<f32>;
@group(0) @binding(1) var<storage, read> baseStyle: array<f32>;
@group(0) @binding(2) var<storage, read_write> styleCases: array<f32>;

@compute @workgroup_size(64)
fn ${fn('prepareStyleCases')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= ${CASES}u) { return; }
  let cb = c * ${NS_STATE.WORDS}u;
  let db = c * ${STYLE.WORDS}u;
  // Fallback index matches the host upload: child = floor(c / MAX_BRANCHES).
  let sb = ((c / ${MAX_BRANCHES}u) % ${SKILL_CAPACITY}u) * ${STYLE.WORDS}u;
  for (var q = 0u; q < ${STYLE.WORDS}u; q = q + 1u) {
    styleCases[db + q] = baseStyle[sb + q];
  }
  // NS_STATE.harmonyType opcode -> STYLE.kind opcode (kernelScore.ts:46-48):
  // forge=1, alchemical=2, resonance=3, inscription=5, cadence=6.
  // Eccentric decree has no harmonyData branch in buildScoreWords -> 0.
  let opcode = u32(caseNodes[cb + ${NS_STATE.harmonyType}u] + 0.5);
  var kind = 0u;
  if (opcode == 1u) { kind = 1u; }
  else if (opcode == 2u) { kind = 3u; }
  else if (opcode == 3u) { kind = 4u; }
  else if (opcode == 5u) { kind = 2u; }
  else if (opcode == 6u) { kind = 5u; }
  if (kind == 0u) { return; }
  styleCases[db + ${STYLE.kind}u] = f32(kind);
  if (kind == 1u) {
    let heat = caseNodes[cb + ${NS_STATE.harmonyForgeHeat}u];
    let last = caseNodes[cb + ${NS_STATE.harmonyForgeLastBuffedHeat}u];
    styleCases[db + ${STYLE.forgeHeat}u] = heat;
    // -1 is the node's "absent" sentinel; buildScoreWords falls back to heat.
    styleCases[db + ${STYLE.forgeLastBuffedHeat}u] = select(heat, last, last >= 0.0);
  } else if (kind == 3u) {
    let ab = cb + ${NS_STATE.harmonyAlchCharges}u;
    var charges = 0.0;
    for (var i = 0u; i < ${HARMONY_CHARGE_CAPACITY}u; i = i + 1u) {
      if (caseNodes[ab + i] >= 0.0) { charges = charges + 1.0; }
    }
    styleCases[db + ${STYLE.alchemicalCharges}u] = charges;
    // HARMONY_REACTION_KEYS order (records.ts:2130-2137).
    let rb = cb + ${NS_STATE.harmonyReactionMods}u;
    styleCases[db + ${STYLE.alchemicalControlMult}u] = caseNodes[rb + 0u];
    styleCases[db + ${STYLE.alchemicalIntensityMult}u] = caseNodes[rb + 1u];
    styleCases[db + ${STYLE.alchemicalCritBonus}u] = caseNodes[rb + 2u];
    styleCases[db + ${STYLE.alchemicalQiReduction}u] = caseNodes[rb + 3u];
    styleCases[db + ${STYLE.alchemicalStabReduction}u] = caseNodes[rb + 4u];
    styleCases[db + ${STYLE.alchemicalSuccessBonus}u] = caseNodes[rb + 5u];
    // Recommendation bitmask: bit (id-1); alignment fusion=0, refine=1, else=2.
    let mask = u32(caseNodes[cb + ${NS_STATE.harmonyRecommendMask}u]);
    var rc = 0u;
    for (var bit = 0u; bit < ${HARMONY_RECOMMEND_CAPACITY}u; bit = bit + 1u) {
      if ((mask & (1u << bit)) != 0u && rc < 8u) {
        styleCases[db + ${STYLE.recommended}u + rc] = select(2.0, f32(bit), bit < 2u);
        rc = rc + 1u;
      }
    }
    styleCases[db + ${STYLE.recommendedCount}u] = f32(rc);
  } else if (kind == 4u) {
    styleCases[db + ${STYLE.resonanceStrength}u] =
      caseNodes[cb + ${NS_STATE.harmonyResonanceStrength}u];
    // Node ids: fusion=1, refine=2, stabilize/support=3/4; alignment maps
    // fusion->0, refine->1, everything else->2, absent->-1.
    let t = caseNodes[cb + ${NS_STATE.harmonyResonanceType}u];
    styleCases[db + ${STYLE.resonanceType}u] =
      select(-1.0, select(2.0, t - 1.0, t < 3.0), t >= 0.0);
    let p = caseNodes[cb + ${NS_STATE.harmonyResonancePendingType}u];
    styleCases[db + ${STYLE.pendingResonanceType}u] =
      select(-1.0, select(2.0, p - 1.0, p < 3.0), p >= 0.0);
    styleCases[db + ${STYLE.pendingCount}u] =
      caseNodes[cb + ${NS_STATE.harmonyResonancePendingCount}u];
  }
  // Inscription stacks and cadence chain have no live NS_STATE slot; the host
  // base value is kept for those subsystems.
}
`;

/** Step 3: (node, skill) -> AVAIL_PARAM_WORDS, base numbers uploaded once. */
export const PREPARE_AVAIL_PARAMS_WGSL = `
@group(0) @binding(0) var<storage, read> nodeMeta: array<f32>;
@group(0) @binding(1) var<storage, read> base: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn ${fn('prepareAvailParams')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let child = gid.x;
  if (child >= ${CHILD}u) { return; }
  let node = child / ${SKILL_CAPACITY}u;
  let skill = child % ${SKILL_CAPACITY}u;
  let bb = skill * ${AVAIL_PARAM.WORDS}u;
  let ob = child * ${AVAIL_PARAM.WORDS}u;
  for (var q = 0u; q < ${AVAIL_PARAM.WORDS}u; q = q + 1u) { out[ob + q] = base[bb + q]; }
  // qiDroplets is a separate stock, not NS_STATE.pool (= qi); the host sends
  // state.qiDroplets in the base table (statefulSearch.ts:989). Keep it.
  out[ob + ${AVAIL_PARAM.currentConditionKind}u] =
    nodeMeta[node * ${META.WORDS}u + ${META.conditionKind}u];
}
`;

/**
 * Availability kernel indexes `nodes[candidate * NS_STATE.WORDS]` with
 * candidate = child (0..CHILD), but the ping-pong beam buffer holds only BEAM
 * parent nodes. Replicate each parent into all SKILL_CAPACITY child slots so
 * the availability verdict reads that child's own parent state. Without this,
 * every child with node>=1 sees an all-zero state and is marked unavailable.
 */
export const REPLICATE_AVAIL_NODES_WGSL = `
@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn ${fn('replicateAvailNodes')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let child = gid.x;
  if (child >= ${CHILD}u) { return; }
  let node = child / ${SKILL_CAPACITY}u;
  let nb = node * ${NS_STATE.WORDS}u;
  let ob = child * ${NS_STATE.WORDS}u;
  for (var q = 0u; q < ${NS_STATE.WORDS}u; q = q + 1u) { out[ob + q] = nodes[nb + q]; }
}
`;

/** Step 7: device assembly of VERDICT_IN; gains remain the one host table. */
export const PREPARE_VERDICT_INPUTS_WGSL = `
@group(0) @binding(0) var<storage, read> inputNodes: array<f32>;
@group(0) @binding(1) var<storage, read> evNodes: array<f32>;
@group(0) @binding(2) var<storage, read> failNodes: array<f32>;
@group(0) @binding(3) var<storage, read> style: array<f32>;
@group(0) @binding(4) var<storage, read> gains: array<f32>;
@group(0) @binding(5) var<storage, read> availSkills: array<f32>;
@group(0) @binding(6) var<storage, read> branches: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

fn barsMet(c: f32, p: f32, goalC: f32, goalP: f32) -> f32 {
  let hasC = goalC > 0.0;
  let hasP = goalP > 0.0;
  if (!hasC && !hasP) { return 0.0; }
  if (hasC && c < goalC) { return 0.0; }
  if (hasP && p < goalP) { return 0.0; }
  return 1.0;
}

@compute @workgroup_size(64)
fn ${fn('prepareVerdictInputs')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${CHILD * MAX_BRANCHES}u) { return; }
  let child = i / ${MAX_BRANCHES}u;
  let branch = i % ${MAX_BRANCHES}u;
  let node = child / ${SKILL_CAPACITY}u;
  let skill = child % ${SKILL_CAPACITY}u;
  let nb = node * ${NS_STATE.WORDS}u;
  let cb = child * ${NS_STATE.WORDS}u;
  // One condition record per (node, skill) child — generated from that child's
  // post-action harmony, as in statefulSearch.ts:1677-1684.
  let parent = child * ${COND_RECORD_WORDS}u;
  let kind = u32(branches[parent + 1u + branch * 5u + 1u]);
  let gb = (child * ${CONDITION_KINDS}u + kind) * ${GAIN_WORDS}u;
  let ob = i * ${VERDICT_IN.WORDS}u;
  let comp = evNodes[cb + ${NS_STATE.completion}u];
  let perf = evNodes[cb + ${NS_STATE.perfection}u];
  out[ob + ${VERDICT_IN.currentStability}u] = inputNodes[nb + ${NS_STATE.stability}u];
  out[ob + ${VERDICT_IN.displayStability}u] = evNodes[cb + ${NS_STATE.stability}u];
  out[ob + ${VERDICT_IN.floorStability}u] = evNodes[cb + ${NS_STATE.floorStability}u];
  out[ob + ${VERDICT_IN.floorMaxStability}u] = evNodes[cb + ${NS_STATE.floorMaxStability}u];
  out[ob + ${VERDICT_IN.survivalProbability}u] = evNodes[cb + ${NS_STATE.survivalProbability}u];
  out[ob + ${VERDICT_IN.goalsMetAfterAction}u] = barsMet(comp, perf, style[${X.modeCompGoal}u], style[${X.modePerfGoal}u]);
  out[ob + ${VERDICT_IN.baseSuccessSecuredAfterAction}u] = barsMet(comp, perf, style[${X.targetCompletion}u], style[${X.targetPerfection}u]);
  out[ob + ${VERDICT_IN.hasGuaranteedSafeStabilize}u] = style[${X.hasSafe}u];
  out[ob + ${VERDICT_IN.minimumGuaranteedContinuationStability}u] = style[${X.minSafe}u];
  out[ob + ${VERDICT_IN.nearDeathStability}u] = style[${X.nearDeath}u];
  out[ob + ${VERDICT_IN.expectedCompletion}u] = gains[gb + ${GAIN.expectedCompletion}u];
  out[ob + ${VERDICT_IN.expectedPerfection}u] = gains[gb + ${GAIN.expectedPerfection}u];
  out[ob + ${VERDICT_IN.successCompletion}u] = gains[gb + ${GAIN.successCompletion}u];
  out[ob + ${VERDICT_IN.successPerfection}u] = gains[gb + ${GAIN.successPerfection}u];
  out[ob + ${VERDICT_IN.failureCompletion}u] = failNodes[cb + ${NS_STATE.completion}u];
  out[ob + ${VERDICT_IN.failurePerfection}u] = failNodes[cb + ${NS_STATE.perfection}u];
  out[ob + ${VERDICT_IN.failureStability}u] = failNodes[cb + ${NS_STATE.stability}u];
  out[ob + ${VERDICT_IN.isItem}u] = availSkills[skill * ${AVAIL_SKILL.WORDS}u + ${AVAIL_SKILL.isItem}u];
  out[ob + ${VERDICT_IN.targetCompletion}u] = style[${X.targetCompletion}u];
  out[ob + ${VERDICT_IN.targetPerfection}u] = style[${X.targetPerfection}u];
  out[ob + ${VERDICT_IN.displayEndsCraft}u] = gains[gb + ${GAIN.displayEnds}u];
  out[ob + ${VERDICT_IN.initialMaxStability}u] = style[${X.initialMaxStability}u];
}
`;

/** Step 8 preparation: materialize branch/death/finished nodes for the scorer. */
export const PREPARE_SCORE_CASES_WGSL = `
@group(0) @binding(0) var<storage, read> evNodes: array<f32>;
@group(0) @binding(1) var<storage, read> failNodes: array<f32>;
@group(0) @binding(2) var<storage, read> branches: array<f32>;
@group(0) @binding(3) var<storage, read_write> caseNodes: array<f32>;
@group(0) @binding(4) var<storage, read_write> stability: array<f32>;
@group(0) @binding(5) var<storage, read> conditionHashes: array<f32>;

@compute @workgroup_size(64)
fn ${fn('prepareScoreCases')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= ${CASES}u) { return; }
  let leafCount = ${CHILD * MAX_BRANCHES}u;
  let cb = c * ${NS_STATE.WORDS}u;
  if (c < leafCount) {
    let child = c / ${MAX_BRANCHES}u;
    let branch = c % ${MAX_BRANCHES}u;
    let parent = child * ${COND_RECORD_WORDS}u;
    let src = child * ${NS_STATE.WORDS}u;
    for (var q = 0u; q < ${NS_STATE.WORDS}u; q = q + 1u) { caseNodes[cb + q] = evNodes[src + q]; }
    if (branch < u32(branches[parent])) {
      let bb = parent + 1u + branch * 5u;
      // The condition kernel reports a *kind* code (0..5) at bb+1; the host
      // maps kind -> canonical name -> conditionIdOf(name) before scoring
      // (statefulSearch.ts:1706-1712). NS_STATE.conditionId is that hash, not
      // the kind, so resolve it through the 6-word table uploaded once.
      caseNodes[cb + ${NS_STATE.conditionId}u] = conditionHashes[u32(branches[bb + 1u])];
      for (var q = 0u; q < 3u; q = q + 1u) {
        caseNodes[cb + ${NS_STATE.conditionQueue}u + q] = branches[bb + 2u + q];
      }
    }
    stability[c] = evNodes[src + ${NS_STATE.stability}u];
  } else {
    // Engine floor-state terminal (safety.ts:138-140, lookahead.ts:332-333):
    //   scoreFinishedState(applySurvivabilityFloorToState(display, floor)).
    // The scored bars come from the post-action EV/display node, NOT from the
    // forced-failure transition (which re-applies the branch condition and f32
    // accumulation to completion/perfection). Only the stability bars are
    // clamped to the survivability floor. The parent condition kind and the
    // terminal bit are selected by prepareScoreParams below.
    let child = c - leafCount;
    let src = child * ${NS_STATE.WORDS}u;
    for (var q = 0u; q < ${NS_STATE.WORDS}u; q = q + 1u) { caseNodes[cb + q] = evNodes[src + q]; }
    let clamped = max(0.0, min(evNodes[src + ${NS_STATE.stability}u],
      failNodes[src + ${NS_STATE.floorStability}u]));
    caseNodes[cb + ${NS_STATE.stability}u] = clamped;
    caseNodes[cb + ${NS_STATE.maxStability}u] = failNodes[src + ${NS_STATE.floorMaxStability}u];
    stability[c] = clamped;
  }
}
`;

/** scoreParams table is uploaded once; terminal bits + condition deltas are selected on device. */
export const PREPARE_SCORE_PARAMS_WGSL = `
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> branches: array<f32>;
@group(0) @binding(2) var<storage, read> gains: array<f32>;
@group(0) @binding(3) var<storage, read> nodeMeta: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn ${fn('prepareScoreParams')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= ${CASES}u) { return; }
  let leafCount = ${CHILD * MAX_BRANCHES}u;
  var kind = 0u;
  var terminal = 0u;
  if (c < leafCount) {
    let child = c / ${MAX_BRANCHES}u;
    let branch = c % ${MAX_BRANCHES}u;
    let node = child / ${SKILL_CAPACITY}u;
    let parent = child * ${COND_RECORD_WORDS}u;
    let bkind = u32(branches[parent + 1u + branch * 5u + 1u]);
    kind = bkind;
    let childKind = u32(nodeMeta[node * ${META.WORDS}u + ${META.conditionKind}u]);
    let gb = (child * ${CONDITION_KINDS}u + childKind) * ${GAIN_WORDS}u;
    if (branch < u32(branches[parent]) && gains[gb + ${GAIN.terminal}u] > 0.5) { terminal = 1u; }
  } else {
    let child = c - leafCount;
    let node = child / ${SKILL_CAPACITY}u;
    kind = u32(nodeMeta[node * ${META.WORDS}u + ${META.conditionKind}u]);
    terminal = 1u;
  }
  let src = (kind * 2u + terminal) * ${SCORE_PARAM.WORDS}u;
  let dst = c * ${SCORE_PARAM.WORDS}u;
  for (var q = 0u; q < ${SCORE_PARAM.WORDS}u; q = q + 1u) { out[dst + q] = table[src + q]; }
}
`;

/** Survival mixes for every leaf edge, one dispatch before BLEND_WGSL. */
export const PREPARE_BLEND_INPUTS_WGSL = `
@group(0) @binding(0) var<storage, read> verdict: array<f32>;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> blendIn: array<f32>;

@compute @workgroup_size(64)
fn ${fn('prepareBlendInputs')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${CHILD * MAX_BRANCHES}u) { return; }
  let child = i / ${MAX_BRANCHES}u;
  let o = i * ${BLEND_IN_WORDS}u;
  blendIn[o] = verdict[i * ${VERDICT_OUT.WORDS}u + ${VERDICT_OUT.uncollapsedRisk}u];
  blendIn[o + 1u] = verdict[i * ${VERDICT_OUT.WORDS}u + ${VERDICT_OUT.blendProbability}u];
  blendIn[o + 2u] = scores[i];
  blendIn[o + 3u] = scores[${CHILD * MAX_BRANCHES}u + child];
}
`;

/**
 * Per-skill EV over condition branches, weighted by branch probability.
 * Matches engine `lookahead.ts:993-1000`: `score += probability * branchScore`
 * over the transitions already grouped by `getConditionTransitionsWithProvider`
 * (`lookahead.ts:986-992`). The later top-K (`lookahead.ts:1004-1017`) ranks
 * candidates by this expectation, not by a single best branch, so beamApply
 * ranks on `values` rather than on any one `blended` edge.
 */
/**
 * Layout of `values` (one buffer, already bound by both follow-up kernels):
 *   [0, CHILD)                             — per-(node, skill) expectation;
 *   [CHILD, CHILD + CHILD*MAX_BRANCHES)    — branch probability, -1 if absent;
 *   [CHILD*(1+MAX_BRANCHES), ...)          — branch condition kind;
 *   [CHILD*(2+MAX_BRANCHES), ...)          — availability decision per child.
 * The trailing regions replace the ninth storage binding in beamApply
 * (WebGPU's default maxStorageBuffersPerShaderStage is 8).
 */
export const VALUES_LAYOUT = {
  expectation: 0,
  probability: CHILD,
  kind: CHILD * (1 + MAX_BRANCHES),
  available: CHILD * (2 + MAX_BRANCHES),
  WORDS: CHILD * (2 + 2 * MAX_BRANCHES),
} as const;

export const BEAM_VALUE_WGSL = `
@group(0) @binding(0) var<storage, read> branches: array<f32>;
@group(0) @binding(1) var<storage, read> blended: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read> availOut: array<f32>;

@compute @workgroup_size(64)
fn ${fn('beamValue')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let child = gid.x;
  if (child >= ${CHILD}u) { return; }
  let parent = child * ${COND_RECORD_WORDS}u;
  let count = u32(branches[parent]);
  var value = 0.0;
  for (var b = 0u; b < ${MAX_BRANCHES}u; b = b + 1u) {
    let pi = ${VALUES_LAYOUT.probability}u + child * ${MAX_BRANCHES}u + b;
    let ki = ${VALUES_LAYOUT.kind}u + child * ${MAX_BRANCHES}u + b;
    if (b < count) {
      let p = branches[parent + 1u + b * 5u];
      values[pi] = p;
      values[ki] = branches[parent + 1u + b * 5u + 1u];
      value = value + p * blended[child * ${MAX_BRANCHES}u + b];
    } else {
      values[pi] = -1.0;
      values[ki] = 0.0;
    }
  }
  values[child] = value;
  // The availability kernel's verdict must gate expansion; otherwise the
  // resident beam explores skills the host's getAvailableSkills rejects.
  values[${VALUES_LAYOUT.available}u + child] =
    availOut[child * ${AVAIL_OUT.WORDS}u + ${AVAIL_OUT.available}u];
}
`;

/**
 * Per-parent retention: rank PARENTS by their best expected child, then let
 * each retained parent keep all of that child's branches (one slot per edge).
 * A plain global top-K over CHILD candidates let a single level-1 node own all
 * 20 slots; the engine instead evaluates each candidate's own transitions
 * (lookahead.ts:993-1001) before the winner is picked (1004-1017), so a beam
 * that drops every other parent loses whole root/branch subtrees.
 *
 * Level 0 keeps the global child ranking: its 20 nodes are replicas of one
 * root and the gains table is valid only for node 0, so per-parent grouping
 * there would promote four identical root moves. From level 1, slot s maps to
 * (parent of rank s / MAX_BRANCHES, branch s % MAX_BRANCHES). Exactly eight
 * storage bindings: values carries expectation + probability + kind, so no
 * ninth `branches` binding is needed (WebGPU limit is 8).
 */
export const BEAM_SELECT_APPLY_WGSL = `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> caseNodes: array<f32>;
@group(0) @binding(2) var<storage, read> metaIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> metaOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodesOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> trace: array<f32>;

@compute @workgroup_size(64)
fn ${fn('beamApply')}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= ${BEAM}u) { return; }
  let mb = slot * ${META.WORDS}u;
  let targetRank = slot / ${MAX_BRANCHES}u;
  let bestBranch = slot % ${MAX_BRANCHES}u;
  let level = atomicLoad(&counters[0]);
  var bestChild = ${CHILD}u;
  if (level == 0u) {
    // Single root at level 0: keep the global child ranking. The root beam is
    // one replicated state, so grouping by replica slot would force four
    // identical root moves and lose the level-1 skill diversity the engine has.
    for (var child = 0u; child < ${CHILD}u; child = child + 1u) {
      if (values[${VALUES_LAYOUT.available}u + child] < 0.5) { continue; }
      let v = values[child];
      if (!(v > -1.0e29)) { continue; }
      var rank = 0u;
      for (var other = 0u; other < ${CHILD}u; other = other + 1u) {
        if (values[${VALUES_LAYOUT.available}u + other] < 0.5) { continue; }
        let ov = values[other];
        if (ov > -1.0e29 && (ov > v || (ov == v && other < child))) { rank = rank + 1u; }
      }
      if (rank == targetRank) { bestChild = child; break; }
    }
  } else {
    // From level 1 on, rank PARENTS by their best expected continuation and
    // keep one child per parent, so a single level-1 node cannot own all 20
    // branch slots (lookahead.ts:993-1001 evaluates each candidate's own
    // transitions before the winner is chosen at 1004-1017).
    var groupBest: array<f32, ${BEAM}u>;
    var groupChild: array<u32, ${BEAM}u>;
    for (var node = 0u; node < ${BEAM}u; node = node + 1u) {
      // Dead beam slots carry zero/stale states; their children can produce
      // finite but meaningless expectations that would outrank live parents.
      if (metaIn[node * ${META.WORDS}u + ${META.alive}u] < 0.5) { continue; }
      var gv = -1.0e30;
      var gc = ${CHILD}u;
      for (var s = 0u; s < ${SKILL_CAPACITY}u; s = s + 1u) {
        let child = node * ${SKILL_CAPACITY}u + s;
        if (values[${VALUES_LAYOUT.available}u + child] < 0.5) { continue; }
        let v = values[child];
        if (v > -1.0e29 && (v > gv || (v == gv && child < gc))) { gv = v; gc = child; }
      }
      groupBest[node] = gv;
      groupChild[node] = gc;
    }
    for (var node = 0u; node < ${BEAM}u; node = node + 1u) {
      if (groupChild[node] >= ${CHILD}u) { continue; }
      var rank = 0u;
      for (var other = 0u; other < ${BEAM}u; other = other + 1u) {
        if (other == node || groupChild[other] >= ${CHILD}u) { continue; }
        if (groupBest[other] > groupBest[node] ||
            (groupBest[other] == groupBest[node] && other < node)) { rank = rank + 1u; }
      }
      if (rank == targetRank) { bestChild = groupChild[node]; break; }
    }
  }
  if (bestChild >= ${CHILD}u) {
    metaOut[mb + ${META.alive}u] = 0.0;
    return;
  }
  let probability = values[${VALUES_LAYOUT.probability}u + bestChild * ${MAX_BRANCHES}u + bestBranch];
  if (probability < 0.0) {
    metaOut[mb + ${META.alive}u] = 0.0;
    return;
  }
  let node = bestChild / ${SKILL_CAPACITY}u;
  let skill = bestChild % ${SKILL_CAPACITY}u;
  let src = (bestChild * ${MAX_BRANCHES}u + bestBranch) * ${NS_STATE.WORDS}u;
  let dst = slot * ${NS_STATE.WORDS}u;
  for (var q = 0u; q < ${NS_STATE.WORDS}u; q = q + 1u) { nodesOut[dst + q] = caseNodes[src + q]; }
  let pmb = node * ${META.WORDS}u;
  // The continued edge's own probability and condition kind, not the parent's.
  metaOut[mb + ${META.probability}u] = probability;
  metaOut[mb + ${META.rootId}u] = select(metaIn[pmb + ${META.rootId}u], f32(slot), metaIn[pmb + ${META.rootId}u] < 0.0);
  metaOut[mb + ${META.parentId}u] = f32(node);
  metaOut[mb + ${META.skill}u] = f32(skill);
  metaOut[mb + ${META.depth}u] = metaIn[pmb + ${META.depth}u] + 1.0;
  metaOut[mb + ${META.alive}u] = 1.0;
  metaOut[mb + ${META.conditionKind}u] = values[${VALUES_LAYOUT.kind}u + bestChild * ${MAX_BRANCHES}u + bestBranch];
  metaOut[mb + ${META.branch}u] = f32(bestBranch);
  let tb = (level * ${BEAM}u + slot) * 2u;
  trace[tb] = f32(node);
  trace[tb + 1u] = f32(skill);
  // BEAM invocations write concurrently: a plain f32 read-modify-write loses
  // updates (20 slots used to report 1). u32 atomics are core WebGPU.
  atomicAdd(&counters[1], 1u);
}
`;

const NEW_KERNELS: ReadonlyArray<readonly [string, string]> = [
  ['prepareStateParams', PREPARE_STATE_PARAMS_WGSL],
  ['prepareCondParams', PREPARE_COND_PARAMS_WGSL],
  ['prepareStyleCases', PREPARE_STYLE_CASES_WGSL],
  ['prepareAvailParams', PREPARE_AVAIL_PARAMS_WGSL],
  ['replicateAvailNodes', REPLICATE_AVAIL_NODES_WGSL],
  ['prepareVerdictInputs', PREPARE_VERDICT_INPUTS_WGSL],
  ['prepareScoreCases', PREPARE_SCORE_CASES_WGSL],
  ['prepareScoreParams', PREPARE_SCORE_PARAMS_WGSL],
  ['prepareBlendInputs', PREPARE_BLEND_INPUTS_WGSL],
  ['beamValue', BEAM_VALUE_WGSL],
  ['beamApply', BEAM_SELECT_APPLY_WGSL],
];

async function compile(device: GpuDevice, wgsl: string, entryPoint: string): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: wgsl });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(`${entryPoint} WGSL: ${errors.map((e) => `${e.lineNum}:${e.message}`).join(' | ')}`);
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
  });
}

export interface ResidentPipelines {
  readonly state: GpuComputePipeline;
  readonly condition: GpuComputePipeline;
  readonly availability: GpuComputePipeline;
  readonly verdict: GpuComputePipeline;
  readonly scoreInput: GpuComputePipeline;
  readonly score: GpuComputePipeline;
  /** Stand-only: writes base + potential to `buffers.split`. */
  readonly split: GpuComputePipeline;
  readonly blend: GpuComputePipeline;
  readonly prepareStateParams: GpuComputePipeline;
  readonly prepareCondParams: GpuComputePipeline;
  readonly prepareStyleCases: GpuComputePipeline;
  readonly prepareAvailParams: GpuComputePipeline;
  readonly replicateAvailNodes: GpuComputePipeline;
  readonly prepareVerdictInputs: GpuComputePipeline;
  readonly prepareScoreCases: GpuComputePipeline;
  readonly prepareScoreParams: GpuComputePipeline;
  readonly prepareBlendInputs: GpuComputePipeline;
  readonly beamValue: GpuComputePipeline;
  readonly beamApply: GpuComputePipeline;
}

export async function createResidentPipelines(device: GpuDevice): Promise<ResidentPipelines> {
  const sources = finalKernelSources();
  const compiled: Record<string, GpuComputePipeline> = {};
  for (const [name, wgsl] of NEW_KERNELS) compiled[name] = await compile(device, wgsl, `${name}Kernel`);
  compiled.state = await compile(device, sources.state, 'applyStateKernel');
  compiled.condition = await compile(device, sources.condition, 'conditionTransitions');
  compiled.blend = await compile(device, sources.blend, 'blendKernel');
  compiled.availability = await createAvailabilityKernelPipeline(device);
  compiled.verdict = await createVerdictPipeline(device);
  compiled.scoreInput = await createScoreInputPipeline(device);
  compiled.score = await createScoreKernelPipeline(device, 'scoreWithTerminalityKernel');
  compiled.split = await createScoreKernelPipeline(device, 'scoreSplitKernel');
  return compiled as unknown as ResidentPipelines;
}

/** Per-search buffer set. Ping-pong nodes/meta, one staging readback. */
export interface ResidentBuffers {
  readonly nodesA: GpuBuffer;
  readonly metaA: GpuBuffer;
  readonly nodesB: GpuBuffer;
  readonly metaB: GpuBuffer;
  readonly readback: GpuBuffer;
  readonly stateEv: GpuBuffer;
  readonly stateSuccess: GpuBuffer;
  readonly stateFailure: GpuBuffer;
  readonly paramsEv: GpuBuffer;
  readonly paramsSuccess: GpuBuffer;
  readonly paramsFailure: GpuBuffer;
  readonly condParams: GpuBuffer;
  readonly branches: GpuBuffer;
  readonly availParam: GpuBuffer;
  readonly availNodes: GpuBuffer;
  readonly availOut: GpuBuffer;
  readonly verdictIn: GpuBuffer;
  readonly verdictOut: GpuBuffer;
  readonly caseNodes: GpuBuffer;
  readonly caseStability: GpuBuffer;
  readonly scoreInput: GpuBuffer;
  readonly scoreParams: GpuBuffer;
  readonly scores: GpuBuffer;
  readonly terminality: GpuBuffer;
  /** Stand-only: 8 diagnostic words per score case. */
  readonly split: GpuBuffer;
  readonly blendIn: GpuBuffer;
  readonly blendOut: GpuBuffer;
  readonly values: GpuBuffer;
  readonly counters: GpuBuffer;
  readonly trace: GpuBuffer;
  readonly style: GpuBuffer;
  readonly condEffects: GpuBuffer;
  readonly gains: GpuBuffer;
  readonly scoreParamTable: GpuBuffer;
  readonly availParamBase: GpuBuffer;
  readonly catalog: GpuBuffer;
  readonly skillRecords: GpuBuffer;
  readonly availSkill: GpuBuffer;
  readonly stateParamsBase: GpuBuffer;
  readonly harmonyIds: GpuBuffer;
  /** Host-uploaded base snapshot (root skill index), read by prepareStyleCases. */
  readonly styleCasesBase: GpuBuffer;
  /** Per-case snapshot rebuilt from post-action caseNodes on device. */
  readonly styleCases: GpuBuffer;
  readonly stateTrace: GpuBuffer;
  readonly conditionHashes: GpuBuffer;
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const STORAGE_COPY_SRC = STORAGE | GPUBufferUsage.COPY_SRC;

function makeBuffer(device: GpuDevice, words: number, usage: number = STORAGE): GpuBuffer {
  return device.createBuffer({ size: Math.max(16, words * 4), usage });
}

function upload(device: GpuDevice, data: Float32Array, usage: number = STORAGE): GpuBuffer {
  const buffer = makeBuffer(device, data.length, usage);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/** Stand-only score dump: 40 SCORE_INPUT words + SCORE_PARAM words. */
export const SCORE_DUMP_WORDS = SCORE_INPUT.WORDS + SCORE_PARAM.WORDS;

/** Stand-only: a slice of one device buffer, read back after a level submit. */
export interface ResidentDumpSpec {
  readonly buffer: keyof ResidentBuffers;
  readonly offsetWords: number;
  readonly words: number;
}
export const DUMP_LIMIT_WORDS = 2048;
export const READBACK_WORDS =
  BEAM * META_WORDS + BEAM * NS_STATE.WORDS + TRACE_WORDS + 4 + CASES + CHILD + CHILD + CASES * 8
  + SCORE_DUMP_WORDS + DUMP_LIMIT_WORDS;

/** One upload per search; no buffer is created or mapped between level dispatches. */
export function createResidentBuffers(
  device: GpuDevice,
  constants: {
    readonly catalog: Float32Array;
    readonly skillRecords: Float32Array;
    readonly availSkill: Float32Array;
    readonly availParamBase: Float32Array;
    readonly stateParamsBase: Float32Array;
    readonly condEffects: Float32Array;
    /** conditionIdOf(conditionNameOfKind(k)), k = 0..5. */
    readonly conditionHashes: Float32Array;
    readonly gains: Float32Array;
    readonly scoreParamTable: Float32Array;
    readonly style: Float32Array;
    readonly harmonyIds: Float32Array;
    readonly styleCases: Float32Array;
    readonly rootNodes: Float32Array;
    readonly rootMeta: Float32Array;
  },
): ResidentBuffers {
  return {
    nodesA: upload(device, constants.rootNodes, STORAGE_COPY_SRC),
    metaA: upload(device, constants.rootMeta, STORAGE_COPY_SRC),
    nodesB: makeBuffer(device, BEAM * NS_STATE.WORDS, STORAGE_COPY_SRC),
    metaB: makeBuffer(device, BEAM * META_WORDS, STORAGE_COPY_SRC),
    readback: makeBuffer(device, READBACK_WORDS, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    stateEv: makeBuffer(device, CHILD * NS_STATE.WORDS, STORAGE_COPY_SRC),
    stateSuccess: makeBuffer(device, CHILD * NS_STATE.WORDS),
    stateFailure: makeBuffer(device, CHILD * NS_STATE.WORDS, STORAGE_COPY_SRC),
    paramsEv: makeBuffer(device, CHILD * STATE_PARAM_WORDS, STORAGE_COPY_SRC),
    paramsSuccess: makeBuffer(device, CHILD * STATE_PARAM_WORDS),
    paramsFailure: makeBuffer(device, CHILD * STATE_PARAM_WORDS),
    condParams: makeBuffer(device, CHILD * COND_PARAM_WORDS),
    branches: makeBuffer(device, CHILD * COND_RECORD_WORDS, STORAGE_COPY_SRC),
    availParam: makeBuffer(device, CHILD * AVAIL_PARAM.WORDS),
    availNodes: makeBuffer(device, CHILD * NS_STATE.WORDS, STORAGE_COPY_SRC),
    availOut: makeBuffer(device, CHILD * AVAIL_OUT.WORDS),
    verdictIn: makeBuffer(device, CHILD * MAX_BRANCHES * VERDICT_IN.WORDS),
    verdictOut: makeBuffer(device, CHILD * MAX_BRANCHES * VERDICT_OUT.WORDS, STORAGE_COPY_SRC),
    caseNodes: makeBuffer(device, CASES * NS_STATE.WORDS, STORAGE_COPY_SRC),
    caseStability: makeBuffer(device, CASES, STORAGE_COPY_SRC),
    scoreInput: makeBuffer(device, CASES * SCORE_INPUT.WORDS, STORAGE_COPY_SRC),
    scoreParams: makeBuffer(device, CASES * SCORE_PARAM.WORDS, STORAGE_COPY_SRC),
    scores: makeBuffer(device, CASES, STORAGE_COPY_SRC),
    terminality: makeBuffer(device, CASES * TERMINALITY_WORDS, STORAGE_COPY_SRC),
    split: makeBuffer(device, CASES * 8, STORAGE_COPY_SRC),
    blendIn: makeBuffer(device, CHILD * MAX_BRANCHES * BLEND_IN_WORDS),
    blendOut: makeBuffer(device, CHILD * MAX_BRANCHES, STORAGE_COPY_SRC),
    values: makeBuffer(device, VALUES_LAYOUT.WORDS, STORAGE_COPY_SRC),
    counters: makeBuffer(device, 4, STORAGE_COPY_SRC),
    trace: makeBuffer(device, TRACE_WORDS, STORAGE_COPY_SRC),
    style: upload(device, constants.style, STORAGE_COPY_SRC),
    condEffects: upload(device, constants.condEffects),
    conditionHashes: upload(device, constants.conditionHashes, STORAGE_COPY_SRC),
    gains: upload(device, constants.gains),
    scoreParamTable: upload(device, constants.scoreParamTable),
    availParamBase: upload(device, constants.availParamBase),
    catalog: upload(device, constants.catalog),
    skillRecords: upload(device, constants.skillRecords),
    availSkill: upload(device, constants.availSkill),
    stateParamsBase: upload(device, constants.stateParamsBase),
    harmonyIds: upload(device, constants.harmonyIds),
    styleCasesBase: upload(device, constants.styleCases),
    styleCases: makeBuffer(device, constants.styleCases.length, STORAGE_COPY_SRC),
    stateTrace: makeBuffer(device, 128 * 7),
  };
}

export function destroyResidentBuffers(buffers: ResidentBuffers): void {
  Object.keys(buffers).forEach((key) => {
    (buffers as unknown as Record<string, GpuBuffer>)[key].destroy();
  });
}

export interface ResidentLevelOptions {
  /** Requested search depth, clamped to 0..96. The slice executes one level. */
  readonly depth?: number;
  /** Current level index; 0 is the first expansion. */
  readonly level: number;
  /** Stand-only: score case whose SCORE_INPUT/SCORE_PARAM words are read back. */
  readonly dumpCase?: number;
  /** Stand-only: raw device-buffer slices to read back after the submit. */
  readonly dumpBuffers?: readonly ResidentDumpSpec[];
}

/** Two chained resident levels: A→B then B→A, no host node reconstruction. */
export interface ResidentTwoLevelResult {
  readonly level1: ResidentLevelResult;
  readonly level2: ResidentLevelResult;
  readonly timing: {
    readonly level1Ms: number;
    readonly level2Ms: number;
    readonly totalMs: number;
  };
}

export interface ResidentLevelResult {
  readonly nodes: Float32Array;
  readonly meta: Float32Array;
  readonly trace: Float32Array;
  /** [level, nodesExplored, depth, overflow] — u32 atomics on the device. */
  readonly counters: Uint32Array;
  readonly scores: Float32Array;
  /** Σ_b P(branch_b) × blended_b per (node, skill) — lookahead.ts:993-1000. */
  readonly values: Float32Array;
  /** Device availability verdict per (node, skill) child. */
  readonly availability: Float32Array;
  /** Stand-only [base, potential] per score case. */
  readonly split: Float32Array;
  readonly scoreDump?: { readonly input: Float32Array; readonly params: Float32Array };
  /** Stand-only concatenated slices, in the order given by dumpBuffers. */
  readonly dump?: Float32Array;
  readonly timing: { encodeSubmitMs: number; readbackMs: number; totalMs: number };
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const wg = (count: number): number => Math.max(1, Math.ceil(count / 64));

function group(device: GpuDevice, pipeline: GpuComputePipeline, buffers: readonly GpuBuffer[]): GpuBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

/**
 * One level of expansion: all 13 dispatches are encoded into ONE encoder and
 * submitted once; nothing is awaited between them. The only await is the final
 * staging-buffer map, after the submit.
 */
export async function runOneResidentLevel(
  device: GpuDevice,
  pipelines: ResidentPipelines,
  buffers: ResidentBuffers,
  options: ResidentLevelOptions,
): Promise<ResidentLevelResult> {
  const depth = Number.isFinite(options.depth)
    ? Math.max(0, Math.min(DEPTH_MAX, Math.floor(options.depth as number)))
    : 2;
  const level = Math.max(0, Math.min(DEPTH_MAX - 1, Math.floor(options.level)));
  const readNodes = level % 2 === 0 ? buffers.nodesA : buffers.nodesB;
  const readMeta = level % 2 === 0 ? buffers.metaA : buffers.metaB;
  const writeNodes = level % 2 === 0 ? buffers.nodesB : buffers.nodesA;
  const writeMeta = level % 2 === 0 ? buffers.metaB : buffers.metaA;

  device.queue.writeBuffer(buffers.counters, 0, new Uint32Array([level, 0, depth, 0]));

  const start = now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  const dispatch = (pipeline: GpuComputePipeline, bind: readonly GpuBuffer[], count: number): void => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group(device, pipeline, bind));
    pass.dispatchWorkgroups(wg(count));
  };

  // 1-3: prepare/availability, then EV transitions. prepareCondParams must run
  // AFTER the state kernel: it reads the post-action generation harmony.
  dispatch(pipelines.prepareStateParams, [
    readNodes, readMeta, buffers.stateParamsBase, buffers.condEffects,
    buffers.paramsEv, buffers.paramsSuccess, buffers.paramsFailure,
  ], CHILD);
  dispatch(pipelines.prepareAvailParams, [
    readMeta, buffers.availParamBase, buffers.availParam,
  ], CHILD);
  // Availability indexes nodes by child; feed it one replicated parent state
  // per child so slots with node>=1 are not read as an empty region.
  dispatch(pipelines.replicateAvailNodes, [readNodes, buffers.availNodes], CHILD);
  dispatch(pipelines.availability, [
    buffers.availNodes, buffers.availSkill, buffers.availParam, buffers.availOut,
  ], CHILD);

  // 4-5: EV transition, forced failure transition; success runs to parity tooling.
  // The final state kernel indexes inState[nodeBase] with nodeBase = child *
  // NS_STATE.WORDS (child = 0..CHILD-1), while readNodes holds only BEAM parent
  // states. At level 2 that reads out of bounds for child>=BEAM and yields
  // all-zero parent states (first zero: stateEv[66] for beam slot 1, skill 2).
  // availNodes is already the per-child replication of the parent state (step
  // 3), so bind it as inState there. Level 0 keeps readNodes: its BEAM roots are
  // replicas of one state, so child<BEAM already reads the right parent and the
  // accidental zeroing of node>=1 children is what keeps the single-root
  // top-20 ranking intact.
  const stateInput = level > 0 ? buffers.availNodes : readNodes;
  const stateBind = (params: GpuBuffer, output: GpuBuffer): GpuBuffer[] => [
    stateInput, output, buffers.catalog, buffers.skillRecords, params,
    buffers.stateTrace, params,
  ];
  dispatch(pipelines.state, stateBind(buffers.paramsEv, buffers.stateEv), CHILD);
  dispatch(pipelines.state, stateBind(buffers.paramsSuccess, buffers.stateSuccess), CHILD);
  dispatch(pipelines.state, stateBind(buffers.paramsFailure, buffers.stateFailure), CHILD);

  // 6: condition branches. One record per post-action child, not per parent.
  dispatch(pipelines.prepareCondParams, [buffers.stateEv, buffers.style, buffers.condParams], CHILD);
  dispatch(pipelines.condition, [buffers.stateEv, buffers.branches, buffers.condParams], CHILD);

  // 7: verdict inputs -> verdict.
  dispatch(pipelines.prepareVerdictInputs, [
    readNodes, buffers.stateEv, buffers.stateFailure, buffers.style,
    buffers.gains, buffers.availSkill, buffers.branches, buffers.verdictIn,
  ], CHILD * MAX_BRANCHES);
  dispatch(pipelines.verdict, [buffers.verdictIn, buffers.verdictOut], CHILD * MAX_BRANCHES);

  // 8-9: score inputs and scoring, assembled by the device.
  dispatch(pipelines.prepareScoreCases, [
    buffers.stateEv, buffers.stateFailure, buffers.branches,
    buffers.caseNodes, buffers.caseStability, buffers.conditionHashes,
  ], CASES);
  // Rebuild the per-case harmony snapshot from each post-action node before
  // SCORE_INPUT assembly; the host base remains a fallback for uncarried words.
  dispatch(pipelines.prepareStyleCases, [
    buffers.caseNodes, buffers.styleCasesBase, buffers.styleCases,
  ], CASES);
  dispatch(pipelines.scoreInput, [
    buffers.caseNodes, buffers.styleCases, buffers.caseStability, buffers.scoreInput,
  ], CASES);
  dispatch(pipelines.prepareScoreParams, [
    buffers.scoreParamTable, buffers.branches, buffers.gains, readMeta, buffers.scoreParams,
  ], CASES);
  dispatch(pipelines.score, [
    buffers.scoreInput, buffers.scores, buffers.scoreParams, buffers.terminality,
  ], CASES);
  // Stand-only split readout; the entry rewrites scores[idx] unchanged.
  dispatch(pipelines.split, [
    buffers.scoreInput, buffers.scores, buffers.scoreParams, buffers.terminality,
    buffers.split,
  ], CASES);

  // 10-11: survival blend -> per-skill EV -> top-K beam apply.
  dispatch(pipelines.prepareBlendInputs, [
    buffers.verdictOut, buffers.scores, buffers.blendIn,
  ], CHILD * MAX_BRANCHES);
  dispatch(pipelines.blend, [buffers.blendIn, buffers.blendOut], CHILD * MAX_BRANCHES);
  dispatch(pipelines.beamValue, [
    buffers.branches, buffers.blendOut, buffers.values, buffers.availOut,
  ], CHILD);
  dispatch(pipelines.beamApply, [
    buffers.values, buffers.caseNodes, readMeta, writeMeta,
    writeNodes, buffers.counters, buffers.trace,
  ], BEAM);
  pass.end();

  // 12: one staging copy, one submit.
  encoder.copyBufferToBuffer(writeMeta, 0, buffers.readback, 0, BEAM * META_WORDS * 4);
  encoder.copyBufferToBuffer(
    writeNodes, 0, buffers.readback, BEAM * META_WORDS * 4, BEAM * NS_STATE.WORDS * 4,
  );
  const metaOffset = BEAM * META_WORDS + BEAM * NS_STATE.WORDS;
  encoder.copyBufferToBuffer(buffers.trace, 0, buffers.readback, metaOffset * 4, TRACE_WORDS * 4);
  encoder.copyBufferToBuffer(
    buffers.counters, 0, buffers.readback, (metaOffset + TRACE_WORDS) * 4, 16,
  );
  encoder.copyBufferToBuffer(buffers.scores, 0, buffers.readback, (metaOffset + TRACE_WORDS + 4) * 4, CASES * 4);
  encoder.copyBufferToBuffer(
    buffers.values, 0, buffers.readback, (metaOffset + TRACE_WORDS + 4 + CASES) * 4, CHILD * 4,
  );
  encoder.copyBufferToBuffer(
    buffers.values, VALUES_LAYOUT.available * 4,
    buffers.readback, (metaOffset + TRACE_WORDS + 4 + CASES + CHILD) * 4, CHILD * 4,
  );
  encoder.copyBufferToBuffer(
    buffers.split, 0, buffers.readback,
    (metaOffset + TRACE_WORDS + 4 + CASES + CHILD + CHILD) * 4, CASES * 32,
  );
  const dumpCase = options.dumpCase;
  const dumpOffset = READBACK_WORDS - SCORE_DUMP_WORDS - DUMP_LIMIT_WORDS;
  const specs = options.dumpBuffers ?? [];
  if (dumpCase !== undefined) {
    const c = Math.max(0, Math.min(CASES - 1, Math.floor(dumpCase)));
    encoder.copyBufferToBuffer(
      buffers.scoreInput, c * SCORE_INPUT.WORDS * 4,
      buffers.readback, dumpOffset * 4, SCORE_INPUT.WORDS * 4,
    );
    encoder.copyBufferToBuffer(
      buffers.scoreParams, c * SCORE_PARAM.WORDS * 4,
      buffers.readback, (dumpOffset + SCORE_INPUT.WORDS) * 4, SCORE_PARAM.WORDS * 4,
    );
  }
  let dumpCursor = dumpOffset + SCORE_DUMP_WORDS;
  for (const spec of specs) {
    encoder.copyBufferToBuffer(
      buffers[spec.buffer], spec.offsetWords * 4,
      buffers.readback, dumpCursor * 4, spec.words * 4,
    );
    dumpCursor += spec.words;
  }
  device.queue.submit([encoder.finish()]);
  const encodeSubmitMs = now() - start;

  const mapStart = now();
  await buffers.readback.mapAsync(GPUMapMode.READ);
  const raw = buffers.readback.getMappedRange();
  const words = new Float32Array(raw);
  const meta = new Float32Array(words.slice(0, BEAM * META_WORDS));
  const nodes = new Float32Array(
    words.slice(BEAM * META_WORDS, BEAM * META_WORDS + BEAM * NS_STATE.WORDS),
  );
  const trace = new Float32Array(words.slice(metaOffset, metaOffset + TRACE_WORDS));
  const counters = new Uint32Array(raw.slice((metaOffset + TRACE_WORDS) * 4, (metaOffset + TRACE_WORDS + 4) * 4));
  const scores = new Float32Array(
    words.slice(metaOffset + TRACE_WORDS + 4, metaOffset + TRACE_WORDS + 4 + CASES),
  );
  const values = new Float32Array(
    words.slice(metaOffset + TRACE_WORDS + 4 + CASES, metaOffset + TRACE_WORDS + 4 + CASES + CHILD),
  );
  const availability = new Float32Array(
    words.slice(metaOffset + TRACE_WORDS + 4 + CASES + CHILD,
      metaOffset + TRACE_WORDS + 4 + CASES + CHILD + CHILD),
  );
  const split = new Float32Array(
    words.slice(metaOffset + TRACE_WORDS + 4 + CASES + CHILD + CHILD,
      metaOffset + TRACE_WORDS + 4 + CASES + CHILD + CHILD + CASES * 8),
  );
  const scoreDump = dumpCase === undefined ? undefined : {
    input: new Float32Array(words.slice(dumpOffset, dumpOffset + SCORE_INPUT.WORDS)),
    params: new Float32Array(words.slice(dumpOffset + SCORE_INPUT.WORDS, dumpOffset + SCORE_DUMP_WORDS)),
  };
  const dump = specs.length === 0 ? undefined : new Float32Array(
    words.slice(dumpOffset + SCORE_DUMP_WORDS, dumpCursor),
  );
  buffers.readback.unmap();
  const readbackMs = now() - mapStart;
  return {
    nodes, meta, trace, counters, scores, values, availability, split, scoreDump, dump,
    timing: { encodeSubmitMs, readbackMs, totalMs: encodeSubmitMs + readbackMs },
  };
}

/**
 * Chain two resident levels with ping-pong buffers:
 *   level 1: nodesA/metaA → nodesB/metaB
 *   level 2: nodesB/metaB → nodesA/metaA (level-1 child nodes, device-resident)
 *
 * Both levels use the same per-search buffer set; nothing is re-uploaded and
 * the host never reconstructs a child node between submits. The only host work
 * between the two submits is the level-1 staging map (meta/values needed by the
 * ranker); the child STATES stay on the device in nodesB.
 *
 * Production wiring and the depth>2 guard are deliberately untouched.
 */
export async function runTwoResidentLevels(
  device: GpuDevice,
  pipelines: ResidentPipelines,
  buffers: ResidentBuffers,
  options: {
    readonly depth?: number;
    /**
     * Stand-only hook: rebuild the gain table between submits, keyed by level-2
     * node (beam slot) instead of the root skill index. Level 1 keys gains by
     * (rootSkill, skill, kind); level 2 must key them by (levelNode, skill, kind)
     * because `child = node*64 + skill` uses a beam slot as `node` there. Without
     * this, every slot except node 0 reads an empty region and EV2 ≈ -47656.
     */
    readonly gainsForLevel2?: (level1: ResidentLevelResult) => Float32Array;
    /** Stand-only score-case index to dump from level 2. */
    readonly dumpCaseLevel2?: number;
    /** Stand-only raw buffer slices to dump from level 2. */
    readonly dumpBuffersLevel2?: readonly ResidentDumpSpec[];
    /** Stand-only raw buffer slices to dump from level 1 (root survival terms). */
    readonly dumpBuffersLevel1?: readonly ResidentDumpSpec[];
  } = {},
): Promise<ResidentTwoLevelResult> {
  const depth = options.depth ?? DEPTH_MAX;
  const start = now();
  const level1 = await runOneResidentLevel(device, pipelines, buffers, {
    depth, level: 0, dumpBuffers: options.dumpBuffersLevel1,
  });
  if (options.gainsForLevel2) {
    device.queue.writeBuffer(buffers.gains, 0, options.gainsForLevel2(level1));
  }
  const level2 = await runOneResidentLevel(device, pipelines, buffers, {
    depth, level: 1, dumpCase: options.dumpCaseLevel2,
    dumpBuffers: options.dumpBuffersLevel2,
  });
  return {
    level1,
    level2,
    timing: {
      level1Ms: level1.timing.totalMs,
      level2Ms: level2.timing.totalMs,
      totalMs: now() - start,
    },
  };
}
