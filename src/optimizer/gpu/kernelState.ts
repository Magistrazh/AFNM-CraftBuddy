/**
 * Stateful GPU kernel — layer 1: node frame + payment costs.
 *
 * Mirrors tools/gpu-probe/mirror.mjs `buildFrame`, `expectedPaymentCosts` and
 * the scalar part of `phasePaymentCosts` / cleanup:
 *   - buff stats are folded from the catalog's raw scaling payloads against
 *     the node's live buff slots (stacks and internal state included);
 *   - state-tracked and derived-harmony buffs are stripped, harmony modifiers
 *     and condition multipliers are applied from the frame params;
 *   - payment costs are computed per mastery branch (probability weighted,
 *     identical branches kept verbatim), with cost channels, legacy percentage
 *     multipliers and clamps;
 *   - pool, stability and stabilityPenalty are written back; every other node
 *     word is copied unchanged. Later layers own technique/buff effects.
 *
 * The compact kernel (kernel.ts) and skills.ts are untouched.
 */

import {
  BUFF_CONDITION_MODE_OPS,
  BUFF_CONDITION_OPS,
  BUFF_DEF,
  BUFF_DEF_FLAGS,
  BUFF_EFFECT_OPS,
  BUFF_EFFECT_WORDS,
  HARMONY_CHARGE_CAPACITY,
  HARMONY_COMBO_CAPACITY,
  HARMONY_REACTION_CAPACITY,
  HASH_BLAZE,
  HASH_COMPLETION_GAINED,
  BUFF_LIFETIME,
  BUFF_SCALING_WORDS,
  BUFF_SKILL_CAPACITY,
  BUFF_SLOT_CAPACITY,
  BUFF_SLOT_WORDS,
  BUFF_STAT_WORDS,
  BUFF_TRIGGER_OPS,
  BUFF_TRIGGER_WORDS,
  BUFF_VAR_DYNAMIC,
  BUFF_VAR_OPS,
  MAX_OUTCOMES,
  NS_STATE,
  SKILL_ACTION_TYPE,
  SKILL_BUFF_TYPE,
  SKILL_COST_BRANCH,
  SKILL_COST_BRANCH_CAPACITY,
  SKILL_COST_BRANCH_WORDS,
  SKILL_CREATED_BUFF,
  SKILL_CREATED_BUFF_CAPACITY,
  SKILL_CREATED_BUFF_WORDS,
  SKILL_CREATED_FLAGS,
  SKILL_EFFECT,
  SKILL_EFFECT_CAPACITY,
  SKILL_MASTERY,
  SKILL_MASTERY_CAPACITY,
  SKILL_MASTERY_OPS,
  SKILL_STATE,
  SKILL_STATE_COST_OFF,
  SKILL_STATE_CREATED_OFF,
  SKILL_STATE_EFFECT_OFF,
  SKILL_STATE_FLAGS,
  SKILL_STATE_MASTERY_OFF,
} from './records';

export const STATE_KERNEL_WGSL = `
@group(0) @binding(0) var<storage, read> inState: array<f32>;
@group(0) @binding(1) var<storage, read_write> outState: array<f32>;
@group(0) @binding(2) var<storage, read> catalog: array<f32>;
@group(0) @binding(3) var<storage, read> skills: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<f32>;
// Optional per-effect diagnostic buffer. params[27] enables it; the normal
// stand passes 0 and never reads it. Seven words per record.
@group(0) @binding(5) var<storage, read_write> trace: array<f32>;

const NS_STATE_WORDS: u32 = ${NS_STATE.WORDS}u;
const NODE_COMPLETION: u32 = ${NS_STATE.completion}u;
const NODE_PERFECTION: u32 = ${NS_STATE.perfection}u;
const NODE_POOL: u32 = ${NS_STATE.pool}u;
const NODE_STABILITY: u32 = ${NS_STATE.stability}u;
const NODE_MAX_STABILITY: u32 = ${NS_STATE.maxStability}u;
const NODE_PENALTY: u32 = ${NS_STATE.stabilityPenalty}u;
const NODE_FLOOR_STABILITY: u32 = ${NS_STATE.floorStability}u;
const NODE_FLOOR_MAX: u32 = ${NS_STATE.floorMaxStability}u;
const NODE_SURV_PROB: u32 = ${NS_STATE.survivalProbability}u;
const NODE_SURV_STABILITY: u32 = ${NS_STATE.survivingStability}u;
const NODE_SURV_PENALTY: u32 = ${NS_STATE.survivingPenalty}u;
const NODE_COND_STABILITY: u32 = ${NS_STATE.condStability}u;
const NODE_COND_PENALTY: u32 = ${NS_STATE.condPenalty}u;
const NODE_OUTCOME_OVERFLOW: u32 = ${NS_STATE.outcomeOverflow}u;
const NODE_PROJECTION_DEBUG: u32 = ${NS_STATE.projectionDebug}u;
const NODE_PROJECTION_COUNT: u32 = ${NS_STATE.projectionCount}u;
const NODE_SURVIVAL_COUNT: u32 = ${NS_STATE.survivalCount}u;
const NODE_SURVIVAL_TWIN: u32 = ${NS_STATE.survivalTwin}u;
const NODE_CONTROL: u32 = ${NS_STATE.control}u;
const NODE_INTENSITY: u32 = ${NS_STATE.intensity}u;
const NODE_CRIT_CHANCE: u32 = ${NS_STATE.critChance}u;
const NODE_CRIT_MULT: u32 = ${NS_STATE.critMultiplier}u;
const NODE_POOL_PCT: u32 = ${NS_STATE.poolCostPercentage}u;
const NODE_STAB_PCT: u32 = ${NS_STATE.stabilityCostPercentage}u;
const NODE_BUFFS: u32 = ${NS_STATE.buffs}u;
const NODE_COOLDOWNS: u32 = ${NS_STATE.cooldowns}u;
const NODE_STEP: u32 = ${NS_STATE.step}u;
const NODE_ALIVE: u32 = ${NS_STATE.alive}u;
const NODE_FLAGS: u32 = ${NS_STATE.flags}u;
const NODE_CONDITION: u32 = ${NS_STATE.conditionId}u;
const NODE_SUCCESS_BONUS: u32 = ${NS_STATE.successChanceBonus}u;

// --- final layer: harmony subsystem + derived state fields -------------------
const NODE_H_DERIVED: u32 = ${NS_STATE.harmonyDerivedMask}u;
const NODE_H_TYPE: u32 = ${NS_STATE.harmonyType}u;
const NODE_H_HEAT: u32 = ${NS_STATE.harmonyForgeHeat}u;
const NODE_H_LAST_HEAT: u32 = ${NS_STATE.harmonyForgeLastBuffedHeat}u;
const NODE_H_CHARGES: u32 = ${NS_STATE.harmonyAlchCharges}u;
const NODE_H_COMBO: u32 = ${NS_STATE.harmonyAlchLastCombo}u;
const NODE_H_RESONANCE: u32 = ${NS_STATE.harmonyResonanceType}u;
const NODE_H_RES_STRENGTH: u32 = ${NS_STATE.harmonyResonanceStrength}u;
const NODE_H_PENDING: u32 = ${NS_STATE.harmonyResonancePendingType}u;
const NODE_H_PENDING_COUNT: u32 = ${NS_STATE.harmonyResonancePendingCount}u;
const NODE_H_REACTION: u32 = ${NS_STATE.harmonyReactionMods}u;
const NODE_H_RECOMMEND: u32 = ${NS_STATE.harmonyRecommendMask}u;
const NODE_HARMONY: u32 = ${NS_STATE.harmony}u;
const NODE_COND_GEN_HARMONY: u32 = ${NS_STATE.conditionGenerationHarmony}u;
// The documented export/parity mode keeps the generation word equal to the
// clamped bar. The depth-2 EV patch switches this on for condition branches.
const EV_MODE: bool = false;
const NODE_DEFAULT_BUFF_MULT: u32 = ${NS_STATE.defaultBuffMultiplier}u;
const NODE_C_BUFF_MULT: u32 = ${NS_STATE.controlBuffMultiplier}u;
const NODE_I_BUFF_MULT: u32 = ${NS_STATE.intensityBuffMultiplier}u;
const SKILL_BUFF_MULT: u32 = ${SKILL_STATE.buffMultiplier}u;
const NODE_BONUS: u32 = ${NS_STATE.completionBonus}u;
const NODE_C_BOOST: u32 = ${NS_STATE.completionBoost}u;
const NODE_P_BOOST: u32 = ${NS_STATE.perfectionBoost}u;
const H_FORGE: u32 = 1u;
const H_ALCHEMICAL: u32 = 2u;
const H_RESONANCE: u32 = 3u;
const H_CHARGE_CAP: u32 = ${HARMONY_CHARGE_CAPACITY}u;
const H_COMBO_CAP: u32 = ${HARMONY_COMBO_CAPACITY}u;
const H_REACTION_CAP: u32 = ${HARMONY_REACTION_CAPACITY}u;

const DEF_WORDS: u32 = ${BUFF_DEF.WORDS}u;
const DEF_STAT_COUNT: u32 = ${BUFF_DEF.statCount}u;
const DEF_STAT_OFFSET: u32 = ${BUFF_DEF.statOffset}u;
const DEF_INITIAL_COUNT: u32 = ${BUFF_DEF.initialCount}u;
const DEF_INITIAL_OFFSET: u32 = ${BUFF_DEF.initialOffset}u;
const STAT_WORDS: u32 = ${BUFF_STAT_WORDS}u;
const SCALING_WORDS: u32 = ${BUFF_SCALING_WORDS}u;
const SLOT_WORDS: u32 = ${BUFF_SLOT_WORDS}u;
const SLOT_CAPACITY: u32 = ${BUFF_SLOT_CAPACITY}u;

const SKILL_WORDS: u32 = ${SKILL_STATE.WORDS}u;
const SKILL_CONSUMES: u32 = ${SKILL_STATE.consumesTurn}u;
const SKILL_COOLDOWN: u32 = ${SKILL_STATE.cooldown}u;
const SKILL_BUFF_TYPE: u32 = ${SKILL_STATE.reserved2}u;
const SKILL_BUFF_DURATION: u32 = ${SKILL_STATE.reserved3}u;
const BUFF_TYPE_CONTROL: u32 = ${SKILL_BUFF_TYPE.control}u;
const BUFF_TYPE_INTENSITY: u32 = ${SKILL_BUFF_TYPE.intensity}u;
const SKILL_FLAGS: u32 = ${SKILL_STATE.flags}u;
const SKILL_STABILITY_GAIN: u32 = ${SKILL_STATE.stabilityGain}u;
const SKILL_MAX_CHANGE: u32 = ${SKILL_STATE.maxStabilityChange}u;
const SKILL_BUFF_COST_ID: u32 = ${SKILL_STATE.buffCostId}u;
const SKILL_BUFF_COST_AMOUNT: u32 = ${SKILL_STATE.buffCostAmount}u;
const SKILL_COST_COUNT: u32 = ${SKILL_STATE.costBranchCount}u;
const SKILL_COST_BASE: u32 = ${SKILL_STATE_COST_OFF}u;
const COST_WORDS: u32 = ${SKILL_COST_BRANCH_WORDS}u;
const COST_PROBABILITY: u32 = ${SKILL_COST_BRANCH.probability}u;
const COST_QI_SCALING: u32 = ${SKILL_COST_BRANCH.qiScaling}u;
const COST_STAB_SCALING: u32 = ${SKILL_COST_BRANCH.stabilityScaling}u;
const COST_CREATION_MASK: u32 = ${SKILL_COST_BRANCH.creationMask}u;
const COST_MASTERY_FLAG: u32 = ${SKILL_COST_BRANCH.masteryFlag}u;
const COST_BRANCH_CAPACITY: u32 = ${SKILL_COST_BRANCH_CAPACITY}u;
const CREATED_BASE: u32 = ${SKILL_STATE_CREATED_OFF}u;
const CREATED_WORDS: u32 = ${SKILL_CREATED_BUFF_WORDS}u;
const CREATED_ID: u32 = ${SKILL_CREATED_BUFF.buffId}u;
const CREATED_SCALING: u32 = ${SKILL_CREATED_BUFF.scaling}u;
const CREATED_CHANCE: u32 = ${SKILL_CREATED_BUFF.chance}u;
const CREATED_FLAGS: u32 = ${SKILL_CREATED_BUFF.flags}u;
const CREATED_COND_KIND: u32 = ${SKILL_CREATED_BUFF.condKind}u;
const CREATED_COND_MODE: u32 = ${SKILL_CREATED_BUFF.condMode}u;
const CREATED_COND_TARGET: u32 = ${SKILL_CREATED_BUFF.condTarget}u;
const CREATED_COND_PARAM: u32 = ${SKILL_CREATED_BUFF.condParam}u;
const CREATED_COND_FACTOR: u32 = ${SKILL_CREATED_BUFF.condFactor}u;
const CREATED_CAPACITY: u32 = ${SKILL_CREATED_BUFF_CAPACITY}u;

const FLAG_HAS_EFFECTS: u32 = ${SKILL_STATE_FLAGS.hasEffects}u;
const FLAG_DISCIPLINED_TOUCH: u32 = ${SKILL_STATE_FLAGS.isDisciplinedTouch}u;
const FLAG_PREVENT_DECAY: u32 = ${SKILL_STATE_FLAGS.preventsMaxStabilityDecay}u;
const CREATED_COST_RELEVANT: u32 = ${SKILL_CREATED_FLAGS.costRelevant}u;
const CREATED_COST_SUPPRESSED: u32 = ${SKILL_CREATED_FLAGS.costSuppressedAtBuild}u;

// --- layer 2: success roll, mastery frame, technique effects ----------------
const SKILL_SUCCESS: u32 = ${SKILL_STATE.successChance}u;
const SKILL_EFFECT_COUNT: u32 = ${SKILL_STATE.effectCount}u;
const SKILL_MASTERY_COUNT: u32 = ${SKILL_STATE.masteryCount}u;
const SKILL_CREATED_COUNT: u32 = ${SKILL_STATE.createdBuffCount}u;
const SKILL_QI_RESTORE: u32 = ${SKILL_STATE.qiRestore}u;
const E_WORDS: u32 = ${SKILL_EFFECT.WORDS}u;
const E_KIND: u32 = ${SKILL_EFFECT.kind}u;
const E_SCALING: u32 = ${SKILL_EFFECT.scaling}u;
const E_CHANCE: u32 = ${SKILL_EFFECT.chance}u;
const E_FLAGS: u32 = ${SKILL_EFFECT.flags}u;
const M_WORDS: u32 = ${SKILL_MASTERY.WORDS}u;
const M_KIND: u32 = ${SKILL_MASTERY.kind}u;
const M_SCALING: u32 = ${SKILL_MASTERY.scaling}u;
const M_CHANCE: u32 = ${SKILL_MASTERY.chance}u;
const M_EFFECT_KIND: u32 = ${SKILL_MASTERY.effectKind}u;
const M_EFFECT_REF: u32 = ${SKILL_MASTERY.effectRef}u;
const M_ENTRY: u32 = ${SKILL_MASTERY.entry}u;
const M_ENTRY_CHANCE: u32 = ${SKILL_MASTERY.entryChance}u;
const M_UPGRADE_ADD: u32 = ${SKILL_MASTERY.upgradeAdditive}u;
const M_UPGRADE_MUL: u32 = ${SKILL_MASTERY.upgradeMultiplier}u;
const M_UPGRADE_KEY_HASH: u32 = ${SKILL_MASTERY.upgradeKeyHash}u;
const M_OP_EFFECT: u32 = ${SKILL_MASTERY_OPS.effect}u;
const M_OP_UPGRADE: u32 = ${SKILL_MASTERY_OPS.upgrade}u;
const M_OP_CONTROL: u32 = ${SKILL_MASTERY_OPS.control}u;
const M_OP_INTENSITY: u32 = ${SKILL_MASTERY_OPS.intensity}u;
const M_OP_CRIT_CHANCE: u32 = ${SKILL_MASTERY_OPS.critchance}u;
const M_OP_CRIT_MULT: u32 = ${SKILL_MASTERY_OPS.critmultiplier}u;
const EFF_COMPLETION: u32 = ${BUFF_EFFECT_OPS.completion}u;
const EFF_PERFECTION: u32 = ${BUFF_EFFECT_OPS.perfection}u;
const EFF_STABILITY: u32 = ${BUFF_EFFECT_OPS.stability}u;
const EFF_MAX_STABILITY: u32 = ${BUFF_EFFECT_OPS.maxStability}u;
const EFF_POOL: u32 = ${BUFF_EFFECT_OPS.pool}u;
const EFF_CREATE_BUFF: u32 = ${BUFF_EFFECT_OPS.createBuff}u;
const OUTCOME_CAPACITY: u32 = ${MAX_OUTCOMES}u;
const P_MAX_COMPLETION: u32 = 25u;
const P_MAX_PERFECTION: u32 = 26u;

const VAR_NONE: u32 = 0u;
const VAR_FLAT: u32 = ${BUFF_VAR_OPS.flat}u;
const VAR_CONTROL: u32 = ${BUFF_VAR_OPS.control}u;
const VAR_INTENSITY: u32 = ${BUFF_VAR_OPS.intensity}u;
const VAR_STACKS: u32 = ${BUFF_VAR_OPS.stacks}u;
const VAR_MAX_POOL: u32 = ${BUFF_VAR_OPS.maxpool}u;
const VAR_MAX_STABILITY: u32 = ${BUFF_VAR_OPS.maxstability}u;
const VAR_COMPLETION: u32 = ${BUFF_VAR_OPS.completion}u;
const VAR_PERFECTION: u32 = ${BUFF_VAR_OPS.perfection}u;
const VAR_MAX_COMPLETION: u32 = ${BUFF_VAR_OPS.maxcompletion}u;
const VAR_MAX_PERFECTION: u32 = ${BUFF_VAR_OPS.maxperfection}u;
const VAR_CRIT_CHANCE: u32 = ${BUFF_VAR_OPS.critchance}u;
const VAR_CRIT_MULT: u32 = ${BUFF_VAR_OPS.critmultiplier}u;
const VAR_OVERCRIT: u32 = ${BUFF_VAR_OPS.overcrit}u;
const VAR_OVERCRIT_CAP: u32 = ${BUFF_VAR_OPS.overcritcapbonus}u;
const VAR_POOL: u32 = ${BUFF_VAR_OPS.pool}u;
const VAR_STABILITY: u32 = ${BUFF_VAR_OPS.stability}u;
const VAR_POOL_FLAT: u32 = ${BUFF_VAR_OPS.poolcostflat}u;
const VAR_SUCCESS_BONUS: u32 = ${BUFF_VAR_OPS.successchancebonus}u;
const VAR_POOL_PCT: u32 = ${BUFF_VAR_OPS.poolcostpercentage}u;
const VAR_STAB_PCT: u32 = ${BUFF_VAR_OPS.stabilitycostpercentage}u;
const VAR_QI_REDUCTION: u32 = ${BUFF_VAR_OPS.qicostreduction}u;
const VAR_STAB_REDUCTION: u32 = ${BUFF_VAR_OPS.stabilitycostreduction}u;
const VAR_COMPLETION_BOOST: u32 = ${BUFF_VAR_OPS.completionboost}u;
const VAR_PERFECTION_BOOST: u32 = ${BUFF_VAR_OPS.perfectionboost}u;
const VAR_QI_BOOST: u32 = ${BUFF_VAR_OPS.qiboost}u;
const VAR_STABILITY_BOOST: u32 = ${BUFF_VAR_OPS.stabilityboost}u;
const VAR_QI_PENALTY: u32 = ${BUFF_VAR_OPS.qicostpenalty}u;
const VAR_STABILITY_PENALTY: u32 = ${BUFF_VAR_OPS.stabilitycostpenalty}u;
const VAR_DYNAMIC: u32 = ${BUFF_VAR_DYNAMIC}u;

// --- layer 3: buff effects + live-owner buff slots ---------------------------
// Buff effect entry (records.ts encodeEffect, BUFF_EFFECT_WORDS):
//   kind, 12 scaling words, condOp, condParam, condMode, condTarget,
//   chance, buffTarget, stateKey, flags, mode.
const BE_KIND: u32 = 0u;
const BE_SCALING: u32 = 1u;
const BE_COND_OP: u32 = ${1 + BUFF_SCALING_WORDS}u;
const BE_COND_PARAM: u32 = ${2 + BUFF_SCALING_WORDS}u;
const BE_COND_MODE: u32 = ${3 + BUFF_SCALING_WORDS}u;
const BE_COND_TARGET: u32 = ${4 + BUFF_SCALING_WORDS}u;
const BE_CHANCE: u32 = ${5 + BUFF_SCALING_WORDS}u;
const BE_BUFF_TARGET: u32 = ${6 + BUFF_SCALING_WORDS}u;
const BE_WORDS: u32 = ${BUFF_EFFECT_WORDS}u;
// Trigger entry: kind, eventHash, effectOffset, effectCount.
const BT_KIND: u32 = 0u;
const BT_EFFECT_OFFSET: u32 = 2u;
const BT_EFFECT_COUNT: u32 = 3u;
const BT_WORDS: u32 = ${BUFF_TRIGGER_WORDS}u;
const BT_ON_ACTION: u32 = ${BUFF_TRIGGER_OPS.onAction}u;
const BT_ON_FUSION: u32 = ${BUFF_TRIGGER_OPS.onFusion}u;
const BT_ON_REFINE: u32 = ${BUFF_TRIGGER_OPS.onRefine}u;
const BT_ON_STABILIZE: u32 = ${BUFF_TRIGGER_OPS.onStabilize}u;
const BT_ON_SUPPORT: u32 = ${BUFF_TRIGGER_OPS.onSupport}u;
const BT_EVENT: u32 = ${BUFF_TRIGGER_OPS.event}u;
const EFF_NEGATE: u32 = ${BUFF_EFFECT_OPS.negate}u;
const EFF_ADD_STACK: u32 = ${BUFF_EFFECT_OPS.addStack}u;
const EFF_CONSUME_BUFF: u32 = ${BUFF_EFFECT_OPS.consumeBuff}u;
const COND_BUFF: u32 = ${BUFF_CONDITION_OPS.buff}u;
const COND_MODE_MORE: u32 = ${BUFF_CONDITION_MODE_OPS.more}u;
const COND_MODE_LESS: u32 = ${BUFF_CONDITION_MODE_OPS.less}u;
const DEF_FLAGS: u32 = ${BUFF_DEF.flags}u;
const DEF_FLAG_DERIVED: u32 = ${BUFF_DEF_FLAGS.derivedHarmony}u;
const SKILL_ACTION_TYPE: u32 = ${SKILL_STATE.actionType}u;
const ACTION_FUSION: u32 = ${SKILL_ACTION_TYPE.fusion}u;
const ACTION_REFINE: u32 = ${SKILL_ACTION_TYPE.refine}u;
const ACTION_STABILIZE: u32 = ${SKILL_ACTION_TYPE.stabilize}u;
const ACTION_SUPPORT: u32 = ${SKILL_ACTION_TYPE.support}u;
const CREATED_FROM_MASTERY: u32 = ${SKILL_CREATED_FLAGS.fromMastery}u;
const LIVE_CAP: u32 = ${BUFF_SLOT_CAPACITY + SKILL_CREATED_BUFF_CAPACITY}u;
const BE_STATE_KEY: u32 = ${7 + BUFF_SCALING_WORDS}u;
const EFF_SET_STATE: u32 = ${BUFF_EFFECT_OPS.setState}u;
const HASH_STATE_BLAZE: u32 = ${HASH_BLAZE}u;
const HASH_EVENT_COMPLETION: u32 = ${HASH_COMPLETION_GAINED}u;
const BAR_EVENT_CAP: u32 = 32u;

struct LiveSlot {
  id: f32,
  stacks: f32,
  turns: f32,
  v0: f32,
  v1: f32,
  alive: u32,
};

/// The catalog marks NAME candidates once at search start; whether the
/// subsystem is active is a per-node fact carried in harmonyDerivedMask.
fn isDerivedDef(id: f32, nodeBase: u32) -> bool {
  if (id < 0.0) { return true; }
  if ((u32(catalog[u32(id) * DEF_WORDS + DEF_FLAGS]) & DEF_FLAG_DERIVED) == 0u) { return false; }
  // params[6] is P_HARMONY (HARMONY_OPS); the mask bits are declared in
  // records.ts derivedHarmonyActiveBit.
  var bit = 0u;
  let ht = u32(params[6u]);
  if (ht == 1u) { bit = 1u; }        // forge
  else if (ht == 3u) { bit = 2u; }   // resonance
  else if (ht == 4u) { bit = 4u; }   // eccentricDecree
  else if (ht == 2u) { bit = 8u; }   // alchemical
  return bit != 0u && (u32(inState[nodeBase + NODE_H_DERIVED]) & bit) != 0u;
}

fn liveFind(live: ptr<function, array<LiveSlot, LIVE_CAP>>, count: u32, id: f32) -> i32 {
  for (var i = 0u; i < count; i = i + 1u) {
    if ((*live)[i].alive != 0u && (*live)[i].id == id) { return i32(i); }
  }
  return -1;
}

fn liveSetStacks(live: ptr<function, array<LiveSlot, LIVE_CAP>>, slot: u32, stacks: f32, maxStacks: f32) {
  var s = max(0.0, stacks);
  if (maxStacks > 0.0) { s = min(s, maxStacks); }
  (*live)[slot].stacks = s;
  if (s <= 0.0) { (*live)[slot].alive = 0u; }
}

fn liveUpsert(live: ptr<function, array<LiveSlot, LIVE_CAP>>, count: ptr<function, u32>,
              id: f32, delta: f32, turns: f32) {
  if (id < 0.0 || !(delta > 0.0)) { return; }
  let d = floor(delta);
  if (!(d > 0.0)) { return; }
  let maxStacks = catalog[u32(id) * DEF_WORDS + ${BUFF_DEF.maxStacks}u];
  let found = liveFind(live, *count, id);
  if (found >= 0) {
    let slot = u32(found);
    liveSetStacks(live, slot, (*live)[slot].stacks + d, maxStacks);
    return;
  }
  if (*count >= LIVE_CAP) { return; }
  (*live)[*count].id = id;
  (*live)[*count].turns = turns;
  (*live)[*count].v0 = 0.0;
  (*live)[*count].v1 = 0.0;
  (*live)[*count].alive = 1u;
  liveSetStacks(live, *count, d, maxStacks);
  *count = *count + 1u;
}

fn liveAdjust(live: ptr<function, array<LiveSlot, LIVE_CAP>>, count: ptr<function, u32>,
              id: f32, delta: f32) {
  if (delta == 0.0 || id < 0.0) { return; }
  let found = liveFind(live, *count, id);
  if (found < 0) { return; }
  let slot = u32(found);
  if (delta > 0.0) {
    let maxStacks = catalog[u32(id) * DEF_WORDS + ${BUFF_DEF.maxStacks}u];
    liveSetStacks(live, slot, (*live)[slot].stacks + floor(delta), maxStacks);
  } else {
    // skills.ts:4737-4749: remove the floored count, delete at zero.
    liveSetStacks(live, slot, (*live)[slot].stacks - floor(-delta), 0.0);
  }
}

fn liveConsume(live: ptr<function, array<LiveSlot, LIVE_CAP>>, count: ptr<function, u32>,
               id: f32, delta: f32) {
  if (delta > 0.0) { liveAdjust(live, count, id, -floor(delta)); }
}

fn buffChance(entry: u32, ownerStacks: f32) -> f32 {
  let op = u32(catalog[entry + BE_COND_OP]);
  let pre = catalog[entry + BE_CHANCE];
  // A buff: self condition reads the owner's live stack count, which a
  // preceding self addStack may already have changed (skills.ts:5404).
  if (op == COND_BUFF && catalog[entry + BE_COND_TARGET] < -1.5) {
    let threshold = catalog[entry + BE_COND_PARAM];
    let mode = u32(catalog[entry + BE_COND_MODE]);
    if (mode == COND_MODE_MORE) { return select(0.0, 1.0, ownerStacks >= threshold); }
    if (mode == COND_MODE_LESS) { return select(0.0, 1.0, ownerStacks < threshold); }
    return select(0.0, 1.0, ownerStacks == threshold);
  }
  return pre;
}

fn heldStacks(nodeBase: u32, id: f32) -> f32 {
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    let entry = nodeBase + NODE_BUFFS + slot * SLOT_WORDS;
    if (inState[entry] == id) { return inState[entry + 1u]; }
  }
  return 0.0;
}

/// A created buff's condition on a named buff depends on the node's live map.
/// records.ts bakes the condition's opcode/threshold here; only the gate is
/// recomputed per node (the entry's static factor is kept separately).
fn createdChance(ce: u32, nodeBase: u32) -> f32 {
  if (u32(skills[ce + CREATED_COND_KIND]) != COND_BUFF) {
    return skills[ce + CREATED_CHANCE];
  }
  let threshold = skills[ce + CREATED_COND_PARAM];
  let held = heldStacks(nodeBase, skills[ce + CREATED_COND_TARGET]);
  let mode = u32(skills[ce + CREATED_COND_MODE]);
  var gate = 0.0;
  if (mode == COND_MODE_MORE) { gate = select(0.0, 1.0, held >= threshold); }
  else if (mode == COND_MODE_LESS) { gate = select(0.0, 1.0, held < threshold); }
  else { gate = select(0.0, 1.0, held == threshold); }
  return gate * skills[ce + CREATED_COND_FACTOR];
}

// skills.ts:4812-4873 + 5399-5463: a bar applied from a buff effect reads a
// SEPARATE crit frame. It starts from the pre-action state baseline and folds
// stack-only crit stats of every buff in the engine's newBuffs (post-creation,
// post-tick), then adds harmony crit. This is NOT finalFrame.critFactor: the
// technique frame deliberately uses the pre-action buff list, while the buff
// frame includes buffs created this action and the extra stacks a creation
// adds to an already-held buff (pattern-step-1#184 focus: 23/185 -> 39/255,
// expected factor 1.6045 instead of 1.396). Overcrit / cap bonus / chain growth
// are zero in the verification corpus, so the stack-only expectation reduces
// to 1 - c + c * multiplier/100.
fn foldCritStats(id: f32, stacks: f32, nodeBase: u32, f: ptr<function, Frame>,
                 chance: ptr<function, f32>, mult: ptr<function, f32>) {
  let defBase = u32(id) * DEF_WORDS;
  let sc = u32(catalog[defBase + DEF_STAT_COUNT]);
  let statBase = u32(catalog[defBase + DEF_STAT_OFFSET]);
  for (var k = 0u; k < sc; k = k + 1u) {
    let e = statBase + k * STAT_WORDS;
    let op = u32(catalog[e]);
    if (op == VAR_CRIT_CHANCE || op == VAR_CRIT_MULT) {
      let raw = evalScalingAt(e + 1u, f, stacks, 0.0, 0.0, u32(id), nodeBase, false);
      if (op == VAR_CRIT_CHANCE) { (*chance) = (*chance) + raw; }
      else { (*mult) = (*mult) + raw; }
    }
  }
}

fn buffCritFactor(nodeBase: u32,
                  live: ptr<function, array<LiveSlot, LIVE_CAP>>,
                  liveCount: u32,
                  effectStacks: ptr<function, array<f32, SLOT_CAPACITY>>,
                  effectAlive: ptr<function, array<u32, SLOT_CAPACITY>>,
                  effectIds: ptr<function, array<f32, SLOT_CAPACITY>>,
                  effectViewCount: u32,
                  baseStacks: ptr<function, array<f32, SLOT_CAPACITY>>,
                  preExistingCount: u32,
                  f: ptr<function, Frame>) -> f32 {
  var chance = inState[nodeBase + NODE_CRIT_CHANCE];
  var mult = inState[nodeBase + NODE_CRIT_MULT];
  // Effect-view slots: pre-existing owners (stacks = creation delta + tick
  // delta + initial) and mastery extras. live holds post-creation stacks,
  // effectStacks the in-tick addStack result; their sum minus the initial
  // snapshot is the live stack count the engine's buff frame reads.
  for (var s = 0u; s < effectViewCount; s = s + 1u) {
    if ((*effectAlive)[s] == 0u) { continue; }
    let id = (*effectIds)[s];
    if (isDerivedDef(id, nodeBase)) { continue; }
    var stacks = (*effectStacks)[s];
    if (s < preExistingCount && s < liveCount && (*live)[s].id == id) {
      stacks = (*live)[s].stacks + (*effectStacks)[s] - (*baseStacks)[s];
    }
    if (stacks > 0.0) { foldCritStats(id, stacks, nodeBase, f, &chance, &mult); }
  }
  // Creations made by this action that are not already in the effect view.
  for (var s = preExistingCount; s < liveCount; s = s + 1u) {
    if ((*live)[s].alive == 0u) { continue; }
    let id = (*live)[s].id;
    if (isDerivedDef(id, nodeBase)) { continue; }
    var inView = false;
    for (var v = 0u; v < effectViewCount; v = v + 1u) {
      if ((*effectIds)[v] == id) { inView = true; break; }
    }
    if (inView) { continue; }
    let stacks = (*live)[s].stacks;
    if (stacks > 0.0) { foldCritStats(id, stacks, nodeBase, f, &chance, &mult); }
  }
  chance = chance + params[P_H_CRIT];
  let c = clamp(chance / 100.0, 0.0, 1.0);
  return 1.0 - c + c * (mult / 100.0);
}

// Frame parameters (storage buffer, f32 words; see scripts/gpu-harness.ts):
//  0 skill index, 1 condition id, 2/3 condition control/intensity deltas,
//  4 max pool, 5 initial max stability, 6 harmony opcode,
//  7/8 harmony control/intensity multipliers, 9 harmony crit chance bonus,
//  10/11 harmony qi/stability cost reductions, 12 derived-buff mask,
//  13..16 derived buff ids, 17..20 state-tracked buff ids,
//  23/24 target completion/perfection, 25/26 hard completion/perfection caps,
//  27 trace enable flag, 29 harmony successChanceBonus,
//  30 condition chance bonus (kind 'chance' effects).
const P_TRACE: u32 = 27u;
const TRACE_STRIDE: u32 = 7u;
const TRACE_RECORDS: u32 = 128u;
// Mastery upgrades belong to the ACTION, not to the buff catalog. Held buff
// effects carry only an upgradeKey hash (records.ts encodeScaling word 8), so
// they must be resolved against the skill currently being applied.
var<private> activeSkill: u32 = 0u;
// Record layout: [phase, kind, weight-or-first, a, b, c, d].
fn tr(slot: u32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) {
  if (params[P_TRACE] > 0.5 && slot < TRACE_RECORDS) {
    let o = slot * TRACE_STRIDE;
    trace[o] = a;
    trace[o + 1u] = b;
    trace[o + 2u] = c;
    trace[o + 3u] = d;
    trace[o + 4u] = e;
    trace[o + 5u] = f;
    trace[o + 6u] = f32(slot);
  }
}
const P_SKILL: u32 = 0u;
const P_CONDITION: u32 = 1u;
const P_COND_CONTROL: u32 = 2u;
const P_COND_INTENSITY: u32 = 3u;
const P_MAX_POOL: u32 = 4u;
const P_INITIAL_MAX: u32 = 5u;
const P_HARMONY: u32 = 6u;
const P_H_CTRL: u32 = 7u;
const P_H_INT: u32 = 8u;
const P_H_CRIT: u32 = 9u;
const P_H_QI_RED: u32 = 10u;
const P_H_STAB_RED: u32 = 11u;
const P_DERIVED_MASK: u32 = 12u;
const P_DERIVED_IDS: u32 = 13u;
const P_TRACKED_IDS: u32 = 17u;
const P_TARGET_COMPLETION: u32 = 23u;
const P_TARGET_PERFECTION: u32 = 24u;
// skills.ts:2478 — the harmony stat modifiers add their successChanceBonus to
// the action variables (alchemical Reaction 0.25, resonance strength*0.03).
const P_H_SUCCESS: u32 = 29u;
// skills.ts:2090-2162 — condition chance effects add their bonus to the same
// variable; the export corpus only has control/intensity/pool/stability, so
// this is zero there, but the frame is not condition-free by construction.
const P_COND_CHANCE: u32 = 30u;
// mirror.mjs:2957 — only sublime, turn-consuming actions with an active model advance.
const P_SUBLIME: u32 = 28u;
// S4 fatal-detection dispatch: 0 = no override (EV/export), 1 = force success,
// 2 = force failure. safety.ts:229-238 needs a full transition at chance 0.
const P_FORCE_SUCCESS: u32 = 31u;

struct Frame {
  control: f32,
  intensity: f32,
  critchance: f32,
  critmultiplier: f32,
  poolPctRaw: f32,
  stabPctRaw: f32,
  qiReduction: f32,
  qiPenalty: f32,
  stabReduction: f32,
  stabPenalty: f32,
  poolFlatRaw: f32,
  boostCompletion: f32,
  boostPerfection: f32,
  boostStability: f32,
  boostQi: f32,
  controlMult: f32,
  intensityMult: f32,
  hasPoolPct: f32,
  hasStabPct: f32,
  critFactor: f32,
  // Sum of buff-stat successChanceBonus (skills.ts:1905-2025). Only the
  // depth-2 EV patch reads it; export mode keeps appliedSuccess = 1.
  successBonus: f32,
};

fn floorTo(v: f32) -> f32 {
  // mirror.mjs:1284 floors round(v * 1e10) / 1e10; f32 cannot hold 1e10 for
  // large costs, so snap small values only and floor the rest directly.
  if (abs(v) < 1.0e5) { return floor(floor(v * 1.0e5 + 0.5) / 1.0e5); }
  return floor(v);
}
// f32 products of decimal game constants land just below the integer the
// model's JS doubles produce (0.9*90 -> 80.99999 instead of 81). Snap by a
// relative epsilon before flooring; the engine (skills.ts:2003-2011,
// gameTypes.ts:820-911) computes these in doubles.
fn floorDoubled(v: f32) -> f32 {
  return floor(v + max(1.0e-6, abs(v) * 1.0e-6));
}
fn ceilDoubled(v: f32) -> f32 {
  return ceil(v - max(1.0e-6, abs(v) * 1.0e-6));
}
fn roundAway(v: f32, p: f32) -> f32 {
  return sign(v) * floor(abs(v) * p + 0.5) / p;
}
fn roundScaling(v: f32) -> f32 {
  if (v > 10.0) { return floorDoubled(v); }
  if (v < -10.0) { return ceilDoubled(v); }
  let p = select(100.0, 10000.0, abs(v) < 0.01);
  return roundAway(v, p);
}
fn combineCostStat(reduction: bool, current: f32, contribution: f32) -> f32 {
  if (reduction) {
    return 100.0 - ((100.0 - min(100.0, current)) * (100.0 - min(100.0, contribution))) / 100.0;
  }
  return ((100.0 + current) * (100.0 + contribution)) / 100.0 - 100.0;
}
fn costMultiplier(reduction: f32, penalty: f32) -> f32 {
  return max(0.0, 100.0 - reduction) * max(0.0, 100.0 + penalty) / 1.0e4;
}

// Dynamic scaling keys are 24-bit hashes of the buff's initialState keys; the
// node stores their values in v0/v1 in catalog order (records.ts:2597-2617).
fn dynamicVar(hash: f32, v0: f32, v1: f32, def: u32) -> f32 {
  let defBase = def * DEF_WORDS;
  let count = u32(catalog[defBase + DEF_INITIAL_COUNT]);
  let base = u32(catalog[defBase + DEF_INITIAL_OFFSET]);
  for (var i = 0u; i < count; i = i + 1u) {
    if (catalog[base + i * 2u] == hash) {
      if (i == 0u) { return v0; }
      if (i == 1u) { return v1; }
      return 0.0;
    }
  }
  return 0.0;
}

fn varValue(op: u32, hash: f32, f: ptr<function, Frame>, stacks: f32,
            v0: f32, v1: f32, def: u32, nodeBase: u32) -> f32 {
  switch (op) {
    case VAR_FLAT: { return 1.0; }
    case VAR_CONTROL: { return (*f).control * (*f).controlMult; }
    case VAR_INTENSITY: { return (*f).intensity * (*f).intensityMult; }
    case VAR_STACKS: { return stacks; }
    case VAR_MAX_POOL: { return params[P_MAX_POOL]; }
    case VAR_MAX_STABILITY: { return inState[nodeBase + NODE_MAX_STABILITY]; }
    case VAR_COMPLETION: { return inState[nodeBase + NODE_COMPLETION]; }
    case VAR_PERFECTION: { return inState[nodeBase + NODE_PERFECTION]; }
    case VAR_MAX_COMPLETION: { return params[P_TARGET_COMPLETION]; }
    case VAR_MAX_PERFECTION: { return params[P_TARGET_PERFECTION]; }
    case VAR_CRIT_CHANCE: { return (*f).critchance; }
    case VAR_CRIT_MULT: { return (*f).critmultiplier; }
    case VAR_OVERCRIT: { return inState[nodeBase + ${NS_STATE.overcrit}u]; }
    case VAR_OVERCRIT_CAP: { return inState[nodeBase + ${NS_STATE.overcritCapBonus}u]; }
    case VAR_POOL: { return inState[nodeBase + NODE_POOL]; }
    case VAR_STABILITY: { return inState[nodeBase + NODE_STABILITY]; }
    case VAR_POOL_FLAT: { return inState[nodeBase + ${NS_STATE.poolCostFlat}u]; }
    case VAR_SUCCESS_BONUS: { return inState[nodeBase + ${NS_STATE.successChanceBonus}u]; }
    case VAR_POOL_PCT: { return inState[nodeBase + NODE_POOL_PCT]; }
    case VAR_STAB_PCT: { return inState[nodeBase + NODE_STAB_PCT]; }
    case VAR_QI_REDUCTION: { return inState[nodeBase + ${NS_STATE.qiCostReduction}u]; }
    case VAR_STAB_REDUCTION: { return inState[nodeBase + ${NS_STATE.stabilityCostReduction}u]; }
    case VAR_COMPLETION_BOOST: { return (*f).boostCompletion; }
    case VAR_PERFECTION_BOOST: { return (*f).boostPerfection; }
    case VAR_QI_BOOST: { return (*f).boostQi; }
    case VAR_STABILITY_BOOST: { return (*f).boostStability; }
    case VAR_QI_PENALTY: { return (*f).qiPenalty; }
    case VAR_STABILITY_PENALTY: { return (*f).stabPenalty; }
    case VAR_DYNAMIC: { return dynamicVar(hash, v0, v1, def); }
    default: { return 0.0; }
  }
}

// Buff stats and skill costs use the same 12-word payload encoding but live
// in two different storage buffers.
fn scalingWord(base: u32, fromSkills: bool) -> f32 {
  if (fromSkills) { return skills[base]; }
  return catalog[base];
}

// Catalog/skill scaling payload: value, stat op, scaling op, eqn factor,
// max cap, max-on, scaling cap, scaling-on, upgrade add/mul, stat/scaling hash.
fn evalScalingAt(sc: u32, f: ptr<function, Frame>, stacks: f32,
                 v0: f32, v1: f32, def: u32, nodeBase: u32,
                 fromSkills: bool) -> f32 {
  var v = scalingWord(sc, fromSkills);
  if (!fromSkills) {
    let upgradeHash = scalingWord(sc + 8u, false);
    if (upgradeHash != 0.0) {
      let mCount = u32(skills[activeSkill * SKILL_WORDS + SKILL_MASTERY_COUNT]);
      let mBase = ${SKILL_STATE_MASTERY_OFF}u + activeSkill * ${SKILL_MASTERY_CAPACITY}u * M_WORDS;
      for (var i = 0u; i < mCount; i = i + 1u) {
        let m = mBase + i * M_WORDS;
        if (u32(skills[m + M_KIND]) == M_OP_UPGRADE &&
            skills[m + M_UPGRADE_KEY_HASH] == upgradeHash) {
          v = (v + skills[m + M_UPGRADE_ADD]) * skills[m + M_UPGRADE_MUL];
          break;
        }
      }
    }
  }
  let statOp = u32(scalingWord(sc + 1u, fromSkills));
  let scalingOp = u32(scalingWord(sc + 2u, fromSkills));
  if (statOp != VAR_NONE) {
    v = v * varValue(statOp, scalingWord(sc + 10u, fromSkills), f, stacks, v0, v1, def, nodeBase);
  }
  if (scalingOp != VAR_NONE) {
    var sv = varValue(scalingOp, scalingWord(sc + 11u, fromSkills), f, stacks, v0, v1, def, nodeBase);
    if (scalingWord(sc + 7u, fromSkills) > 0.5) {
      let cap = scalingWord(sc + 6u, fromSkills);
      if (cap < 0.0) { sv = max(sv, cap); } else { sv = min(sv, cap); }
    }
    v = v * sv;
  }
  v = v * scalingWord(sc + 3u, fromSkills);
  v = roundScaling(v);
  if (scalingWord(sc + 5u, fromSkills) > 0.5) {
    let cap = scalingWord(sc + 4u, fromSkills);
    if (cap < 0.0) { v = max(v, cap); } else { v = min(v, cap); }
  }
  return v;
}

fn foldBuff(f: ptr<function, Frame>, id: u32, stacks: f32,
            v0: f32, v1: f32, nodeBase: u32) {
  let defBase = id * DEF_WORDS;
  let count = u32(catalog[defBase + DEF_STAT_COUNT]);
  let statBase = u32(catalog[defBase + DEF_STAT_OFFSET]);
  // skills.ts:1924-1991 evaluates every buff stat against the PRE-buff
  // variable frame (vars), accumulating results into adjustedVars. Folding
  // with the running frame compounds a second control/intensity stat onto the
  // first buff's contribution (e.g. two mastery control buffs: 0.1*272 twice
  // becomes 0.1*191 + 0.1*210). Keep the baseline for stat evaluation only.
  var evalFrame: Frame = (*f);
  evalFrame.control = inState[nodeBase + NODE_CONTROL];
  evalFrame.intensity = inState[nodeBase + NODE_INTENSITY];
  evalFrame.controlMult = 1.0;
  evalFrame.intensityMult = 1.0;
  for (var k = 0u; k < count; k = k + 1u) {
    let entry = statBase + k * STAT_WORDS;
    let op = u32(catalog[entry]);
    let raw = evalScalingAt(entry + 1u, &evalFrame, stacks, v0, v1, id, nodeBase, false);
    if (op == ${BUFF_VAR_OPS.completionboost}u) {
      (*f).boostCompletion = (*f).boostCompletion + raw;
    } else if (op == VAR_PERFECTION_BOOST) {
      (*f).boostPerfection = (*f).boostPerfection + raw;
    } else if (op == VAR_QI_BOOST) {
      (*f).boostQi = (*f).boostQi + raw;
    } else if (op == VAR_STABILITY_BOOST) {
      (*f).boostStability = (*f).boostStability + raw;
    } else if (op == VAR_QI_PENALTY) {
      (*f).qiPenalty = combineCostStat(false, (*f).qiPenalty, raw);
    } else if (op == VAR_STABILITY_PENALTY) {
      (*f).stabPenalty = combineCostStat(false, (*f).stabPenalty, raw);
    } else if (op == ${BUFF_VAR_OPS.poolcostpercentage}u) {
      tr(100u + id, 11.0, f32(id), raw, (*f).poolPctRaw, 0.0, 0.0);
      // skills.ts:2003-2011 floors after each multiplication.
      (*f).poolPctRaw = floorDoubled(((*f).poolPctRaw / 100.0) * (raw / 100.0) * 100.0);
      (*f).hasPoolPct = 1.0;
    } else if (op == ${BUFF_VAR_OPS.stabilitycostpercentage}u) {
      (*f).stabPctRaw = floorDoubled(((*f).stabPctRaw / 100.0) * (raw / 100.0) * 100.0);
      (*f).hasStabPct = 1.0;
    } else if (op == ${BUFF_VAR_OPS.qicostreduction}u) {
      (*f).qiReduction = combineCostStat(true, (*f).qiReduction, raw);
    } else if (op == ${BUFF_VAR_OPS.stabilitycostreduction}u) {
      (*f).stabReduction = combineCostStat(true, (*f).stabReduction, raw);
    } else if (op == ${BUFF_VAR_OPS.poolcostflat}u) {
      (*f).poolFlatRaw = (*f).poolFlatRaw + raw;
    } else if (op == VAR_CONTROL) {
      tr(80u + id, 9.0, f32(id), stacks, raw, (*f).control, floorTo(raw));
      (*f).control = (*f).control + floorTo(raw);
    } else if (op == VAR_INTENSITY) {
      (*f).intensity = (*f).intensity + floorTo(raw);
    } else if (op == VAR_CRIT_CHANCE) {
      (*f).critchance = (*f).critchance + raw;
    } else if (op == VAR_SUCCESS_BONUS) {
      // skills.ts:2016-2025 folds buff successChanceBonus stats additively.
      (*f).successBonus = (*f).successBonus + raw;
    } else if (op == VAR_CRIT_MULT) {
      (*f).critmultiplier = (*f).critmultiplier + raw;
    }
  }
}

fn isStrippedBuff(id: f32, nodeBase: u32) -> bool {
  if (id < 0.0) { return true; }
  // records.ts flags every derived-harmony definition, including the
  // forge heat variants (mirror.mjs:1446-1460). The
  // exact-id params below only cover the canonical name of each system.
  if (isDerivedDef(id, nodeBase)) { return true; }
  let mask = u32(params[P_DERIVED_MASK]);
  for (var i = 0u; i < 4u; i = i + 1u) {
    if (params[P_TRACKED_IDS + i] == id) { return true; }
    if ((mask & (1u << i)) != 0u && params[P_DERIVED_IDS + i] == id) {
      return true;
    }
  }
  return false;
}

fn defHasFlag(id: f32, flag: u32) -> bool {
  if (id < 0.0) { return false; }
  return (u32(catalog[u32(id) * DEF_WORDS + DEF_FLAGS]) & flag) != 0u;
}

// skills.ts:1913-1929 — hasExplicitControlBuff / hasExplicitIntensityBuff scan
// every resolved active buff definition, *including* state-tracked ones such as
// Completion Bonus. Only the stat folding excludes those definitions; the gate
// on the legacy turn-counter multiplier does not. A stripped buff therefore
// still suppresses the multiplier: dropping it here made the mastery frame of
// invasive_refine use 272*1.08-136=157.76 instead of the engine's 272-136=136,
// which moved the 10% perfection mastery from 18 to 21.
fn liveHasStatBuff(nodeBase: u32, flag: u32) -> bool {
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    let entry = nodeBase + NODE_BUFFS + slot * SLOT_WORDS;
    let id = inState[entry];
    if (!(inState[entry + 1u] > 0.0)) { continue; }
    if (defHasFlag(id, flag)) { return true; }
  }
  return false;
}

/** skills.ts:1924-1940 for frames built after phaseMasteryFrame: a buff a
 *  previous mastery entry created is already in activeBuffs there. */
fn masteryHasStatBuff(nodeBase: u32, ms: ptr<function, MasteryState>, flag: u32) -> bool {
  if (liveHasStatBuff(nodeBase, flag)) { return true; }
  for (var i = 0u; i < (*ms).extraCount; i = i + 1u) {
    if (defHasFlag((*ms).extras[i].id, flag)) { return true; }
  }
  return false;
}

fn buildFrame(nodeBase: u32, creationMask: u32, skillIndex: u32) -> Frame {
  var f: Frame;
  f.control = inState[nodeBase + NODE_CONTROL];
  f.intensity = inState[nodeBase + NODE_INTENSITY];
  f.critchance = inState[nodeBase + NODE_CRIT_CHANCE];
  f.critmultiplier = inState[nodeBase + NODE_CRIT_MULT];
  f.poolPctRaw = 100.0;
  f.stabPctRaw = 100.0;
  f.qiReduction = inState[nodeBase + ${NS_STATE.qiCostReduction}u];
  f.qiPenalty = inState[nodeBase + ${NS_STATE.qiCostPenalty}u];
  f.stabReduction = inState[nodeBase + ${NS_STATE.stabilityCostReduction}u];
  f.stabPenalty = inState[nodeBase + ${NS_STATE.stabilityCostPenalty}u];
  f.poolFlatRaw = inState[nodeBase + ${NS_STATE.poolCostFlat}u];
  f.boostCompletion = 0.0;
  f.boostPerfection = inState[nodeBase + ${NS_STATE.completionBonus}u] * 10.0;
  f.boostStability = 0.0;
  f.boostQi = 0.0;
  // Condition effects belong to the node's own condition id, not to a
  // caller-supplied default: a mismatched node gets the neutral multiplier.
  let conditionMatch = select(
    0.0,
    1.0,
    inState[nodeBase + NODE_CONDITION] == params[P_CONDITION],
  );
  f.controlMult = 1.0 + params[P_COND_CONTROL] * conditionMatch;
  f.intensityMult = 1.0 + params[P_COND_INTENSITY] * conditionMatch;
  f.hasPoolPct = 0.0;
  f.hasStabPct = 0.0;
  f.successBonus = 0.0;

  // skills.ts:1924-1940 — a turn-counter control/intensity buff multiplies the
  // base stat BEFORE any buff stat is folded, and only when no live buff
  // carries that stat explicitly. The payment frame runs before masteries, so
  // no created buff can suppress it here.
  if (inState[nodeBase + ${NS_STATE.controlBuffTurns}u] > 0.0 &&
      !liveHasStatBuff(nodeBase, ${BUFF_DEF_FLAGS.hasControlStat}u)) {
    f.control = f.control * inState[nodeBase + ${NS_STATE.controlBuffMultiplier}u];
  }
  if (inState[nodeBase + ${NS_STATE.intensityBuffTurns}u] > 0.0 &&
      !liveHasStatBuff(nodeBase, ${BUFF_DEF_FLAGS.hasIntensityStat}u)) {
    f.intensity = f.intensity * inState[nodeBase + ${NS_STATE.intensityBuffMultiplier}u];
  }

  // mirror.mjs:1553-1567 — live buffs first, then mastery creations.
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    let entry = nodeBase + NODE_BUFFS + slot * SLOT_WORDS;
    let id = inState[entry];
    if (isStrippedBuff(id, nodeBase)) { continue; }
    foldBuff(&f, u32(id), inState[entry + 1u], inState[entry + 3u], inState[entry + 4u], nodeBase);
  }
  let createdBase = CREATED_BASE + skillIndex * CREATED_CAPACITY * CREATED_WORDS;
  for (var i = 0u; i < CREATED_CAPACITY; i = i + 1u) {
    if ((creationMask & (1u << i)) == 0u) { continue; }
    let entry = createdBase + i * CREATED_WORDS;
    if ((u32(skills[entry + CREATED_FLAGS]) & CREATED_COST_RELEVANT) == 0u) { continue; }
    let id = skills[entry + CREATED_ID];
    if (id < 0.0) { continue; }
    // mirror.mjs:2229 floors the creation's raw stack value; the encoded
    // payload may carry upgrades, so read only its value word here.
    foldBuff(&f, u32(id), floor(skills[entry + CREATED_SCALING]), 0.0, 0.0, nodeBase);
  }

  // mirror.mjs:1640-1649 — harmony modifiers (state baseline, not buffed).
  if (params[P_H_CTRL] != 1.0) {
    f.control = f.control + (params[P_H_CTRL] - 1.0) * inState[nodeBase + NODE_CONTROL];
  }
  if (params[P_H_INT] != 1.0) {
    f.intensity = f.intensity + (params[P_H_INT] - 1.0) * inState[nodeBase + NODE_INTENSITY];
  }
  f.critchance = f.critchance + params[P_H_CRIT];
  if (params[P_H_QI_RED] != 0.0) {
    f.qiReduction = combineCostStat(true, f.qiReduction, params[P_H_QI_RED]);
  }
  if (params[P_H_STAB_RED] != 0.0) {
    f.stabReduction = combineCostStat(true, f.stabReduction, params[P_H_STAB_RED]);
  }
  let p = clamp(f.critchance / 100.0, 0.0, 1.0);
  f.critFactor = 1.0 - p + p * (f.critmultiplier / 100.0);
  return f;
}

fn costFor(f: ptr<function, Frame>, entry: u32, nodeBase: u32) -> vec2<f32> {
  let qiCost = evalScalingAt(entry + COST_QI_SCALING, f, 0.0, 0.0, 0.0, 0u, nodeBase, true);
  let stabCost = evalScalingAt(entry + COST_STAB_SCALING, f, 0.0, 0.0, 0.0, 0u, nodeBase, true);
  let flat = max(0.0, floorTo((*f).poolFlatRaw));
  let qiFactor = costMultiplier((*f).qiReduction, (*f).qiPenalty);
  let stabFactor = costMultiplier((*f).stabReduction, (*f).stabPenalty);
  // mirror.mjs:1666-1689 — legacy percentages only apply when a held buff
  // actually carries the stat; the divided-by baseline is the raw state value.
  // skills.ts:1209-1220 normalizeRuntimeCostPercentage: a buff-derived 0 is a
  // neutral 100, so the mirror's value-or-100 fallback must be reproduced here.
  var rawPoolPct = floorTo((*f).poolPctRaw);
  if (rawPoolPct == 0.0) { rawPoolPct = 100.0; }
  var rawStabPct = floorTo((*f).stabPctRaw);
  if (rawStabPct == 0.0) { rawStabPct = 100.0; }
  let poolPct = select(
    1.0,
    rawPoolPct / max(1.0e-10, inState[nodeBase + NODE_POOL_PCT]),
    (*f).hasPoolPct > 0.5,
  );
  let stabPct = select(
    1.0,
    rawStabPct / max(1.0e-10, inState[nodeBase + NODE_STAB_PCT]),
    (*f).hasStabPct > 0.5,
  );
  tr(6u, 10.0, qiCost, stabCost, flat, qiFactor, poolPct);
  return vec2<f32>(
    max(0.0, floorTo((qiCost + flat) * qiFactor * poolPct)),
    floorTo(stabCost * stabFactor * stabPct),
  );
}

fn hasHeldBuff(nodeBase: u32, id: f32) -> bool {
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    if (inState[nodeBase + NODE_BUFFS + slot * SLOT_WORDS] == id) { return true; }
  }
  return false;
}

/// records.ts drops cost-relevant creations that are already held in the
/// initial state. The catalog is built once, so the drop has to be applied
/// against each node's own live buff map here.
fn activeCreationMask(nodeBase: u32, skillIndex: u32, costEntry: u32, mask: u32) -> u32 {
  var result = mask;
  let createdBase = CREATED_BASE + skillIndex * CREATED_CAPACITY * CREATED_WORDS;
  let branchCount = u32(skills[skillIndex * SKILL_WORDS + SKILL_COST_COUNT]);
  // A single resolved branch still applies every included creation
  // unconditionally, even when varies is set by a DIFFERENT (held) creation.
  // records.ts drops creations held in the START state; a later node that no
  // longer holds one has to get its cost stats back. Only certainty-1
  // creations can be re-added without inventing a new probability branch.
  let singleBranch = branchCount <= 1u;
  for (var i = 0u; i < CREATED_CAPACITY; i++) {
    let ce = createdBase + i * CREATED_WORDS;
    let flags = u32(skills[ce + CREATED_FLAGS]);
    let costRelevant = (flags & CREATED_COST_RELEVANT) != 0u;
    let id = skills[ce + CREATED_ID];
    if (id < 0.0) { continue; }
    let held = hasHeldBuff(nodeBase, id);
    if (held) { result = result & ~(1u << i); }
  }
  if (singleBranch) {
    for (var i = 0u; i < CREATED_CAPACITY; i++) {
      let ce = createdBase + i * CREATED_WORDS;
      let flags = u32(skills[ce + CREATED_FLAGS]);
      let id = skills[ce + CREATED_ID];
      if (id < 0.0 || hasHeldBuff(nodeBase, id)) { continue; }
      let suppressed = (flags & CREATED_COST_RELEVANT) != 0u &&
        (flags & CREATED_COST_SUPPRESSED) != 0u;
      if (suppressed && skills[ce + CREATED_CHANCE] > 0.999) {
        result = result | (1u << i);
      }
    }
  }
  return result;
}

fn expectedPaymentCosts(nodeBase: u32, skillIndex: u32) -> vec2<f32> {
  let header = skillIndex * SKILL_WORDS;
  let branchCount = u32(skills[header + SKILL_COST_COUNT]);
  let costBase = SKILL_COST_BASE + skillIndex * COST_BRANCH_CAPACITY * COST_WORDS;
  if (branchCount == 0u) {
    var frame = buildFrame(nodeBase, 0u, skillIndex);
    return costFor(&frame, costBase, nodeBase);
  }
  var first = vec2<f32>(0.0, 0.0);
  var haveFirst = false;
  var different = false;
  var qi = 0.0;
  var stability = 0.0;
  for (var i = 0u; i < COST_BRANCH_CAPACITY; i = i + 1u) {
    if (i >= branchCount) { break; }
    let entry = costBase + i * COST_WORDS;
    let probability = skills[entry + COST_PROBABILITY];
    let mask = activeCreationMask(nodeBase, skillIndex, entry, u32(skills[entry + COST_CREATION_MASK]));
    var frame = buildFrame(nodeBase, mask, skillIndex);
    let cost = costFor(&frame, entry, nodeBase);
    if (!haveFirst) {
      first = cost;
      haveFirst = true;
    } else if (cost.x != first.x || cost.y != first.y) {
      different = true;
    }
    qi = qi + probability * cost.x;
    stability = stability + probability * cost.y;
  }
  // mirror.mjs:2286 — identical branch costs stay verbatim; averaging them
  // manufactures f32 roundoff that costs a whole stability point in cleanup.
  if (!different) { return first; }
  return vec2<f32>(qi, stability);
}

// --- layer 2: ordered stability outcomes, mastery frame, numeric effects -----

struct Outcome {
  stability: f32,
  penaltyDelta: f32,
  probability: f32,
};

// --- S4: full StabilityProjection (stability.ts:27-268) ----------------------
// The scalar EV path above stays untouched so guided parity is not perturbed.
// This parallel distribution is what decides the floor, P(s>0) and the
// conditional means. penalty is an ABSOLUTE penalty, as in the engine's
// StabilityOutcome, so finish() can use clamp(ceil(penalty + decay), 0, max).
struct BranchOutcome {
  stability: f32,
  penalty: f32,
  probability: f32,
  succeeded: u32, // 0 = unset, 1 = technique success, 2 = technique failure
};

struct Projection {
  ev: array<BranchOutcome, OUTCOME_CAPACITY>,
  evCount: u32,
  sv: array<BranchOutcome, OUTCOME_CAPACITY>,
  svCount: u32,
  hasSv: u32,
  overflow: u32,
};

// Refuse loudly on overflow: stability.ts has no capacity limit, so a silent
// merge here would fabricate a floor. The flag is exported and the stand gates
// on it (A6). All helpers take one struct pointer plus a plain u32 selector — WGSL
// rejects two pointer arguments derived from the same struct (alias rule).
// The selector is passed as a plain u32, not a pointer.
fn pCountVal(p: ptr<function, Projection>, which: u32) -> u32 {
  if (which == 1u) { return (*p).svCount; }
  return (*p).evCount;
}
fn pSetCount(p: ptr<function, Projection>, which: u32, n: u32) {
  if (which == 1u) { (*p).svCount = n; }
  else { (*p).evCount = n; }
}
fn pRead(p: ptr<function, Projection>, which: u32, i: u32) -> BranchOutcome {
  if (which == 1u) { return (*p).sv[i]; }
  return (*p).ev[i];
}
fn pWrite(p: ptr<function, Projection>, which: u32, i: u32, o: BranchOutcome) {
  if (which == 1u) { (*p).sv[i] = o; }
  else { (*p).ev[i] = o; }
}
fn appendLocal(arr: ptr<function, array<BranchOutcome, OUTCOME_CAPACITY>>,
               cnt: ptr<function, u32>, overflowFlag: ptr<function, u32>, o: BranchOutcome) {
  if (o.probability <= 0.0) { return; }
  var n = *cnt;
  for (var i = 0u; i < n; i = i + 1u) {
    let old = (*arr)[i];
    if (old.stability == o.stability && old.penalty == o.penalty &&
        old.succeeded == o.succeeded) {
      var merged = old;
      merged.probability = old.probability + o.probability;
      (*arr)[i] = merged;
      return;
    }
  }
  if (n >= OUTCOME_CAPACITY) {
    (*overflowFlag) = 1u;
    return;
  }
  (*arr)[n] = o;
  *cnt = n + 1u;
}
fn pAppend(p: ptr<function, Projection>, which: u32, o: BranchOutcome) {
  if (o.probability <= 0.0) { return; }
  let n = pCountVal(p, which);
  for (var i = 0u; i < n; i = i + 1u) {
    let old = pRead(p, which, i);
    if (old.stability == o.stability && old.penalty == o.penalty &&
        old.succeeded == o.succeeded) {
      var merged = old;
      merged.probability = old.probability + o.probability;
      pWrite(p, which, i, merged);
      return;
    }
  }
  if (n >= OUTCOME_CAPACITY) {
    (*p).overflow = 1u;
    return;
  }
  pWrite(p, which, n, o);
  pSetCount(p, which, n + 1u);
}

fn pInit(p: ptr<function, Projection>, stability: f32, penalty: f32) {
  (*p).ev[0] = BranchOutcome(stability, penalty, 1.0, 0u);
  (*p).evCount = 1u;
  (*p).svCount = 0u;
  (*p).hasSv = 0u;
  (*p).overflow = 0u;
}

// stability.ts:118-159
fn pChange(p: ptr<function, Projection>, which: u32, stability: f32, penalty: f32,
           probability: f32, successOnly: bool) {
  if (probability <= 0.0 || (stability == 0.0 && penalty == 0.0)) { return; }
  let count = pCountVal(p, which);
  if (probability >= 1.0) {
    for (var i = 0u; i < count; i = i + 1u) {
      let o = pRead(p, which, i);
      if (successOnly && o.succeeded == 2u) { continue; }
      var next = o;
      next.stability = o.stability + stability;
      next.penalty = o.penalty + penalty;
      pWrite(p, which, i, next);
    }
    return;
  }
  var nextArr: array<BranchOutcome, OUTCOME_CAPACITY>;
  var nextCount = 0u;
  var spill = 0u;
  for (var i = 0u; i < count; i = i + 1u) {
    let o = pRead(p, which, i);
    if (successOnly && o.succeeded == 2u) {
      appendLocal(&nextArr, &nextCount, &spill, o);
      continue;
    }
    appendLocal(&nextArr, &nextCount, &spill,
      BranchOutcome(o.stability, o.penalty, o.probability * (1.0 - probability), o.succeeded));
    appendLocal(&nextArr, &nextCount, &spill,
      BranchOutcome(o.stability + stability, o.penalty + penalty,
                    o.probability * probability, o.succeeded));
  }
  for (var i = 0u; i < nextCount; i = i + 1u) { pWrite(p, which, i, nextArr[i]); }
  pSetCount(p, which, nextCount);
  (*p).overflow = (*p).overflow | spill;
}
fn pChangeBoth(p: ptr<function, Projection>, stability: f32, penalty: f32,
               probability: f32, successOnly: bool) {
  pChange(p, 0u, stability, penalty, probability, successOnly);
  if ((*p).hasSv == 1u) {
    pChange(p, 1u, stability, penalty, probability, successOnly);
  }
}

// stability.ts:80-103, 105-108
fn pBeginOne(p: ptr<function, Projection>, which: u32, probability: f32) {
  if (probability >= 1.0) { return; }
  let count = pCountVal(p, which);
  var nextArr: array<BranchOutcome, OUTCOME_CAPACITY>;
  var nextCount = 0u;
  var spill = 0u;
  for (var i = 0u; i < count; i = i + 1u) {
    let o = pRead(p, which, i);
    if (probability > 0.0) {
      appendLocal(&nextArr, &nextCount, &spill,
        BranchOutcome(o.stability, o.penalty, o.probability * probability, 1u));
    }
    if (probability < 1.0) {
      appendLocal(&nextArr, &nextCount, &spill,
        BranchOutcome(o.stability, o.penalty, o.probability * (1.0 - probability), 2u));
    }
  }
  for (var i = 0u; i < nextCount; i = i + 1u) { pWrite(p, which, i, nextArr[i]); }
  pSetCount(p, which, nextCount);
  (*p).overflow = (*p).overflow | spill;
}
fn pBeginTechnique(p: ptr<function, Projection>, probability: f32) {
  pBeginOne(p, 0u, probability);
  if ((*p).hasSv == 1u) { pBeginOne(p, 1u, probability); }
}
fn pEndTechnique(p: ptr<function, Projection>) {
  for (var w = 0u; w < 2u; w = w + 1u) {
    if (w == 1u && (*p).hasSv == 0u) { continue; }
    let count = pCountVal(p, w);
    for (var i = 0u; i < count; i = i + 1u) {
      var o = pRead(p, w, i);
      o.succeeded = 0u;
      pWrite(p, w, i, o);
    }
  }
}

// stability.ts:67-77 — the compatibility EV stays scalar, the survival copy
// keeps the realized roll. The copy is taken from the current EV distribution.
fn pChangeExpected(p: ptr<function, Projection>, expected: f32, realized: f32,
                   probability: f32) {
  if ((*p).hasSv == 0u) {
    for (var i = 0u; i < (*p).evCount; i = i + 1u) { (*p).sv[i] = (*p).ev[i]; }
    (*p).svCount = (*p).evCount;
    (*p).hasSv = 1u;
  }
  pChange(p, 0u, expected, 0.0, 1.0, false);
  pChange(p, 1u, realized, 0.0, probability, false);
}

// stability.ts:177-190 — each payment outcome is its own branch; no averaging.
fn pPayScalar(p: ptr<function, Projection>, cost: f32) {
  pChangeBoth(p, -cost, 0.0, 1.0, false);
}
fn pPayBranches(p: ptr<function, Projection>,
                probs: ptr<function, array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>>,
                costs: ptr<function, array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>>,
                count: u32) {
  if (count == 0u) { return; }
  pPayArr(p, 0u, probs, costs, count);
  if ((*p).hasSv == 1u) { pPayArr(p, 1u, probs, costs, count); }
}
fn pPayArr(p: ptr<function, Projection>, which: u32,
           probs: ptr<function, array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>>,
           costs: ptr<function, array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>>,
           count: u32) {
  let sourceCount = pCountVal(p, which);
  var nextArr: array<BranchOutcome, OUTCOME_CAPACITY>;
  var nextCount = 0u;
  var spill = 0u;
  for (var i = 0u; i < sourceCount; i = i + 1u) {
    let o = pRead(p, which, i);
    for (var j = 0u; j < count; j = j + 1u) {
      appendLocal(&nextArr, &nextCount, &spill,
        BranchOutcome(o.stability - (*costs)[j], o.penalty,
                      o.probability * (*probs)[j], o.succeeded));
    }
  }
  for (var i = 0u; i < nextCount; i = i + 1u) { pWrite(p, which, i, nextArr[i]); }
  pSetCount(p, which, nextCount);
  (*p).overflow = (*p).overflow | spill;
}
fn pRestoreMaximum(p: ptr<function, Projection>) {
  for (var w = 0u; w < 2u; w = w + 1u) {
    if (w == 1u && (*p).hasSv == 0u) { continue; }
    let count = pCountVal(p, w);
    for (var i = 0u; i < count; i = i + 1u) {
      var o = pRead(p, w, i);
      o.penalty = 0.0;
      pWrite(p, w, i, o);
    }
  }
}

// stability.ts:192-267. Returns EV stability/penalty from the main
// distribution, the floor/P(s>0) from the surviving distribution, and the raw
// conditional sums. The host divides the sums in f64 and applies snapRoundoff
// (stability.ts:5-8): 1e-9 is below an f32 ulp, so doing it in the shader
// would drop the guard and move a following floor/ceil by a whole point.
struct SurvivorStats {
  minStability: f32,
  minMaximum: f32,
  survivalProbability: f32,
  survivingStability: f32,
  survivingPenalty: f32,
};
struct ProjectionResult {
  evStability: f32,
  evPenalty: f32,
  stats: SurvivorStats,
};
fn pSurvivorStats(p: ptr<function, Projection>, which: u32, initialMaximum: f32,
                  decayF: f32, cleanup: bool) -> SurvivorStats {
  var minStability = 1.0e30;
  var minMaximum = 1.0e30;
  var surviveProbability = 0.0;
  var survivingStability = 0.0;
  var survivingPenalty = 0.0;
  let count = pCountVal(p, which);
  for (var i = 0u; i < count; i = i + 1u) {
    let o = pRead(p, which, i);
    var penalty = o.penalty;
    var stability = o.stability;
    if (cleanup) {
      penalty = clamp(ceil(penalty + decayF), 0.0, initialMaximum);
      stability = floor(max(0.0, min(initialMaximum - penalty, stability)));
    }
    minStability = min(minStability, stability);
    minMaximum = min(minMaximum, max(0.0, initialMaximum - penalty));
    if (stability > 0.0) {
      surviveProbability = surviveProbability + o.probability;
      survivingStability = survivingStability + o.probability * stability;
      survivingPenalty = survivingPenalty + o.probability * penalty;
    }
  }
  return SurvivorStats(minStability, minMaximum, clamp(surviveProbability, 0.0, 1.0),
                       survivingStability, survivingPenalty);
}
fn pFinish(p: ptr<function, Projection>, initialMaximum: f32, decayF: f32,
           cleanup: bool) -> ProjectionResult {
  var evStability = 0.0;
  var evPenalty = 0.0;
  for (var i = 0u; i < (*p).evCount; i = i + 1u) {
    let o = (*p).ev[i];
    var penalty = o.penalty;
    var stability = o.stability;
    if (cleanup) {
      penalty = clamp(ceil(penalty + decayF), 0.0, initialMaximum);
      stability = floor(max(0.0, min(initialMaximum - penalty, stability)));
    }
    evStability = evStability + o.probability * stability;
    evPenalty = evPenalty + o.probability * penalty;
  }
  var stats = pSurvivorStats(p, 0u, initialMaximum, decayF, cleanup);
  if ((*p).hasSv == 1u) {
    stats = pSurvivorStats(p, 1u, initialMaximum, decayF, cleanup);
  }
  return ProjectionResult(evStability, evPenalty, stats);
}

struct BuffStack {
  id: f32,
  stacks: f32,
};

struct MasteryState {
  controlBonus: f32,
  intensityBonus: f32,
  critChanceBonus: f32,
  critMultiplierBonus: f32,
  overrideCount: u32,
  overrides: array<BuffStack, ${SKILL_CREATED_BUFF_CAPACITY}u>,
  extraCount: u32,
  extras: array<BuffStack, ${SKILL_CREATED_BUFF_CAPACITY}u>,
};

fn appendOutcome(next: ptr<function, array<Outcome, OUTCOME_CAPACITY>>,
                 count: ptr<function, u32>, o: Outcome) {
  if ((*count) >= OUTCOME_CAPACITY) { return; }
  for (var i = 0u; i < (*count); i = i + 1u) {
    let old = (*next)[i];
    if (old.stability == o.stability && old.penaltyDelta == o.penaltyDelta) {
      var merged = old;
      merged.probability = old.probability + o.probability;
      (*next)[i] = merged;
      return;
    }
  }
  (*next)[(*count)] = o;
  *count = (*count) + 1u;
}

// stability.ts:134-158 — every chance-gated application branches; identical
// (stability, penalty) outcomes merge by summed probability.
fn addBranchOutcome(outcomes: ptr<function, array<Outcome, OUTCOME_CAPACITY>>,
                    count: ptr<function, u32>, stability: f32, penalty: f32,
                    probability: f32) {
  if (probability <= 0.0 || (stability == 0.0 && penalty == 0.0)) { return; }
  var next: array<Outcome, OUTCOME_CAPACITY>;
  var nextCount = 0u;
  for (var i = 0u; i < (*count); i = i + 1u) {
    let o = (*outcomes)[i];
    if (probability < 1.0) {
      appendOutcome(&next, &nextCount,
        Outcome(o.stability, o.penaltyDelta, o.probability * (1.0 - probability)));
    }
    appendOutcome(&next, &nextCount, Outcome(
      o.stability + stability, o.penaltyDelta + penalty,
      o.probability * probability));
  }
  for (var i = 0u; i < nextCount; i = i + 1u) { (*outcomes)[i] = next[i]; }
  *count = nextCount;
}

// S4: the engine's StabilityProjection.pay() splits on every probabilistic
// stability cost (stability.ts:177-190). expectedPaymentCosts above keeps only
// the EV for the scalar path; this function exposes the same branches so the
// projection can keep them. Identical costs stay separate (as in the engine).
fn paymentCostBranches(nodeBase: u32, skillIndex: u32,
                       probs: ptr<function, array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>>,
                       costs: ptr<function, array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>>,
                       count: ptr<function, u32>) {
  let header = skillIndex * SKILL_WORDS;
  let branchCount = u32(skills[header + SKILL_COST_COUNT]);
  let costBase = SKILL_COST_BASE + skillIndex * COST_BRANCH_CAPACITY * COST_WORDS;
  if (branchCount == 0u) {
    var frame = buildFrame(nodeBase, 0u, skillIndex);
    (*costs)[0] = costFor(&frame, costBase, nodeBase).y;
    (*probs)[0] = 1.0;
    *count = 1u;
    return;
  }
  var n = 0u;
  for (var i = 0u; i < COST_BRANCH_CAPACITY; i = i + 1u) {
    if (i >= branchCount || n >= ${SKILL_COST_BRANCH_CAPACITY}u) { break; }
    let entry = costBase + i * COST_WORDS;
    let probability = skills[entry + COST_PROBABILITY];
    if (probability <= 0.0) { continue; }
    let mask = activeCreationMask(nodeBase, skillIndex, entry, u32(skills[entry + COST_CREATION_MASK]));
    var frame = buildFrame(nodeBase, mask, skillIndex);
    (*costs)[n] = costFor(&frame, entry, nodeBase).y;
    (*probs)[n] = probability;
    n = n + 1u;
  }
  *count = n;
}

// skills.ts:3732 — a mastery phase has consumesTurn:false, so finish() applies
// no cleanup: its chance weight becomes a scalar expectation (stability.ts:192).
fn addScalarOutcome(outcomes: ptr<function, array<Outcome, OUTCOME_CAPACITY>>,
                    count: ptr<function, u32>, stability: f32, penalty: f32,
                    probability: f32) {
  if (probability <= 0.0 || (stability == 0.0 && penalty == 0.0)) { return; }
  for (var i = 0u; i < (*count); i = i + 1u) {
    var o = (*outcomes)[i];
    o.stability = o.stability + stability * probability;
    o.penaltyDelta = o.penaltyDelta + penalty * probability;
    (*outcomes)[i] = o;
  }
}

fn overrideStacks(ms: ptr<function, MasteryState>, id: f32) -> f32 {
  for (var i = 0u; i < (*ms).overrideCount; i = i + 1u) {
    if ((*ms).overrides[i].id == id) { return (*ms).overrides[i].stacks; }
  }
  return -1.0;
}

fn setOverrideStacks(ms: ptr<function, MasteryState>, id: f32, stacks: f32) {
  for (var i = 0u; i < (*ms).overrideCount; i = i + 1u) {
    if ((*ms).overrides[i].id == id) {
      (*ms).overrides[i].stacks = stacks;
      return;
    }
  }
  if ((*ms).overrideCount < ${SKILL_CREATED_BUFF_CAPACITY}u) {
    (*ms).overrides[(*ms).overrideCount].id = id;
    (*ms).overrides[(*ms).overrideCount].stacks = stacks;
    (*ms).overrideCount = (*ms).overrideCount + 1u;
  }
}

// mirror.mjs:2388-2397 — makeFrame re-folds the post-upgrade frame after every
// mastery group: live buffs at their upserted stacks, then mastery creations,
// then the control/intensity/crit bonus phases, then the crit factor.
fn buildEffectFrame(nodeBase: u32, ms: ptr<function, MasteryState>) -> Frame {
  var f: Frame;
  f.control = inState[nodeBase + NODE_CONTROL];
  f.intensity = inState[nodeBase + NODE_INTENSITY];
  f.critchance = inState[nodeBase + NODE_CRIT_CHANCE];
  f.critmultiplier = inState[nodeBase + NODE_CRIT_MULT];
  f.poolPctRaw = 100.0;
  f.stabPctRaw = 100.0;
  f.qiReduction = inState[nodeBase + ${NS_STATE.qiCostReduction}u];
  f.qiPenalty = inState[nodeBase + ${NS_STATE.qiCostPenalty}u];
  f.stabReduction = inState[nodeBase + ${NS_STATE.stabilityCostReduction}u];
  f.stabPenalty = inState[nodeBase + ${NS_STATE.stabilityCostPenalty}u];
  f.poolFlatRaw = inState[nodeBase + ${NS_STATE.poolCostFlat}u];
  f.boostCompletion = 0.0;
  f.boostPerfection = inState[nodeBase + ${NS_STATE.completionBonus}u] * 10.0;
  f.boostStability = 0.0;
  f.boostQi = 0.0;
  let conditionMatch = select(
    0.0, 1.0,
    inState[nodeBase + NODE_CONDITION] == params[P_CONDITION]);
  f.controlMult = 1.0 + params[P_COND_CONTROL] * conditionMatch;
  f.intensityMult = 1.0 + params[P_COND_INTENSITY] * conditionMatch;
  f.hasPoolPct = 0.0;
  f.hasStabPct = 0.0;
  f.successBonus = 0.0;
  // skills.ts:1924-1940 — same rule as buildFrame, but mastery-created buffs
  // are part of the resolved active buffs once an earlier entry has run.
  if (inState[nodeBase + ${NS_STATE.controlBuffTurns}u] > 0.0 &&
      !masteryHasStatBuff(nodeBase, ms, ${BUFF_DEF_FLAGS.hasControlStat}u)) {
    f.control = f.control * inState[nodeBase + ${NS_STATE.controlBuffMultiplier}u];
  }
  if (inState[nodeBase + ${NS_STATE.intensityBuffTurns}u] > 0.0 &&
      !masteryHasStatBuff(nodeBase, ms, ${BUFF_DEF_FLAGS.hasIntensityStat}u)) {
    f.intensity = f.intensity * inState[nodeBase + ${NS_STATE.intensityBuffMultiplier}u];
  }
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    let entry = nodeBase + NODE_BUFFS + slot * SLOT_WORDS;
    let id = inState[entry];
    if (isStrippedBuff(id, nodeBase)) { continue; }
    let over = overrideStacks(ms, id);
    let stacks = select(inState[entry + 1u], over, over >= 0.0);
    foldBuff(&f, u32(id), stacks, inState[entry + 3u], inState[entry + 4u], nodeBase);
  }
  for (var i = 0u; i < (*ms).extraCount; i = i + 1u) {
    let s = (*ms).extras[i];
    foldBuff(&f, u32(s.id), s.stacks, 0.0, 0.0, nodeBase);
  }
  if (params[P_H_CTRL] != 1.0) {
    f.control = f.control + (params[P_H_CTRL] - 1.0) * inState[nodeBase + NODE_CONTROL];
  }
  if (params[P_H_INT] != 1.0) {
    f.intensity = f.intensity + (params[P_H_INT] - 1.0) * inState[nodeBase + NODE_INTENSITY];
  }
  f.critchance = f.critchance + params[P_H_CRIT];
  if (params[P_H_QI_RED] != 0.0) {
    f.qiReduction = combineCostStat(true, f.qiReduction, params[P_H_QI_RED]);
  }
  if (params[P_H_STAB_RED] != 0.0) {
    f.stabReduction = combineCostStat(true, f.stabReduction, params[P_H_STAB_RED]);
  }
  f.control = f.control * (1.0 + (*ms).controlBonus);
  f.intensity = f.intensity * (1.0 + (*ms).intensityBonus);
  f.critchance = f.critchance + (*ms).critChanceBonus;
  f.critmultiplier = f.critmultiplier + (*ms).critMultiplierBonus;
  let p = clamp(f.critchance / 100.0, 0.0, 1.0);
  f.critFactor = 1.0 - p + p * (f.critmultiplier / 100.0);
  return f;
}

fn applyBoost(value: f32, boost: f32) -> f32 {
  // numericApplications.ts:13-16 — only positive amounts, and the boost path
  // ends in safeFloor (utils/largeNumbers.ts:109: Math.floor), not the 1e-5
  // scaling rounding.
  if (value > 0.0 && boost != 0.0) { return floor(value * (1.0 + boost / 100.0)); }
  return value;
}

fn boostedBar(critted: f32, boost: f32, floorUnboosted: bool) -> f32 {
  // mirror.mjs:2020-2025 — boost multiplies the UNFLOORED critted amount.
  // The engine applies safeFloor (an integer floor) to the result:
  // numericApplications.ts:13-16 and skills.ts:3105-3116. The crit factor is
  // computed in doubles there; f32 can land one ulp below the integer
  // (1500 * 1.096 -> 1643.99993). Give it less than one ulp of relative slack
  // before flooring, the same f64-emulation idea as floorDoubled for costs.
  // floorDoubled's 1e-6 is several ulps at bar-sized values and would push
  // genuinely-below products over the line.
  if (boost != 0.0 && critted > 0.0) {
    return floor(critted * (1.0 + boost / 100.0) + max(1.0e-7, abs(critted) * 1.0e-7));
  }
  if (floorUnboosted) { return floor(critted + max(1.0e-7, abs(critted) * 1.0e-7)); }
  return critted;
}

fn successRoll(skillChance: f32, bonus: f32) -> f32 {
  // skills.ts:4489-4499 — clamp(base + successChanceBonus).
  return clamp(skillChance + bonus, 0.0, 1.0);
}

fn recordBarEvent(events: ptr<function, array<f32, BAR_EVENT_CAP>>,
                  count: ptr<function, u32>, amount: f32) {
  if (!(amount > 0.0) || (*count) >= BAR_EVENT_CAP) { return; }
  (*events)[(*count)] = amount;
  *count = (*count) + 1u;
}

fn applyNumeric(kind: u32, sc: u32, weight: f32, zeroFlag: bool,
                masteryPhase: bool, nodeBase: u32, f: ptr<function, Frame>,
                pool: ptr<function, f32>, maxPool: f32,
                outcomes: ptr<function, array<Outcome, OUTCOME_CAPACITY>>,
                outcomeCount: ptr<function, u32>,
                barCompRaw: ptr<function, f32>, barPerfRaw: ptr<function, f32>,
                 events: ptr<function, array<f32, BAR_EVENT_CAP>>,
                 eventCount: ptr<function, u32>,
                 proj: ptr<function, Projection>, effectChance: f32,
                 successOnly: bool) {
  let value = evalScalingAt(sc, f, 0.0, 0.0, 0.0, 0u, nodeBase, true);
  if (kind == EFF_STABILITY) {
    if (masteryPhase) {
      addScalarOutcome(outcomes, outcomeCount, value, 0.0, weight);
      pChangeBoth(proj, value, 0.0, effectChance, successOnly);
    } else {
      addBranchOutcome(outcomes, outcomeCount, value, 0.0, weight);
      // skills.ts:5168-5172 — changeTechnique keeps the success gate separate.
      pChangeBoth(proj, value, 0.0, effectChance, successOnly);
    }
  } else if (kind == EFF_MAX_STABILITY) {
    // skills.ts:4569/5245/5973 — the delta enters the penalty, not max directly.
    if (masteryPhase) {
      addScalarOutcome(outcomes, outcomeCount, 0.0, -value, weight);
      pChangeBoth(proj, 0.0, -value, effectChance, successOnly);
    } else {
      addBranchOutcome(outcomes, outcomeCount, 0.0, -value, weight);
      pChangeBoth(proj, 0.0, -value, effectChance, successOnly);
    }
  } else if (kind == EFF_POOL) {
    // skills.ts:5295-5306: applyGainBoost floors the restored amount, the
    // condition factor is applied afterwards without another floor.
    let amount = applyBoost(value, (*f).boostQi) * weight;
    *pool = clamp(*pool + amount, 0.0, maxPool);
  } else if (kind == EFF_COMPLETION || kind == EFF_PERFECTION) {
    // Technique path only: mastery bars are grouped and floored per entry in
    // the phaseMasteryFrame loop (skills.ts:4300-4341).
    // mirror.mjs:2115-2121 — technique drops a negative result from a
    // positive-valued amount; buff applications (later layer) do not.
    let effective = select(value, 0.0, value < 0.0 && zeroFlag);
    let raw = effective * weight;
    if (kind == EFF_COMPLETION) {
      *barCompRaw = (*barCompRaw) + raw;
      // skills.ts:5143-5151 — each positive application fires its own
      // completionGained event; blaze keys on the largest single one.
      recordBarEvent(events, eventCount,
        boostedBar(raw * (*f).critFactor, (*f).boostCompletion, true));
    } else {
      *barPerfRaw = (*barPerfRaw) + raw;
    }
  }
}

// --- final layer: completion bonus, harmony advance, derived frame fields ----

// mirror.mjs:130-152 (outcome.ts:45-72). f32 emulation uses *13/10 because
// *1.3 rounds down one ulp for some targets (verified for all corpus targets).
const BAND_MAX_ITER: u32 = 64u;
fn bandStep(current: f32) -> f32 {
  return floor(current * 13.0 / 10.0);
}
fn bandGuaranteed(value: f32, barTarget: f32) -> f32 {
  if (barTarget <= 0.0) { return 0.0; }
  var current = barTarget;
  var remaining = max(0.0, value);
  var guaranteed = 0.0;
  for (var i = 0u; i < BAND_MAX_ITER; i = i + 1u) {
    if (!(remaining > 0.0 && current > 0.0 && remaining >= current)) { break; }
    remaining = remaining - current;
    guaranteed = guaranteed + 1.0;
    current = bandStep(current);
  }
  return guaranteed;
}
fn bandProgress(value: f32, barTarget: f32) -> f32 {
  if (barTarget <= 0.0) { return 0.0; }
  var current = barTarget;
  var remaining = max(0.0, value);
  var progress = 0.0;
  for (var i = 0u; i < BAND_MAX_ITER; i = i + 1u) {
    if (!(remaining > 0.0 && current > 0.0 && remaining >= current)) { break; }
    remaining = remaining - current;
    progress = progress + 1.0;
    current = bandStep(current);
  }
  if (current > 0.0) { progress = progress + remaining / current; }
  return progress;
}

fn techniqueBit(id: f32) -> u32 {
  if (id < 1.0 || id > 4.0) { return 0u; }
  return 1u << (u32(id) - 1u);
}

// mirror.mjs:2921-2928 — six three-charge recipes, ids already sorted.
fn alchemyCombo(id: u32, slot: u32) -> f32 {
  var combos = array<f32, 18>(
    1.0, 2.0, 4.0,
    1.0, 2.0, 2.0,
    1.0, 1.0, 2.0,
    1.0, 2.0, 3.0,
    2.0, 2.0, 4.0,
    2.0, 3.0, 4.0,
  );
  return combos[id * 3u + slot];
}

fn comboMatch(c: ptr<function, array<f32, H_CHARGE_CAP>>,
              start: u32, id: u32) -> bool {
  for (var slot = 0u; slot < 3u; slot = slot + 1u) {
    if ((*c)[start + slot] != alchemyCombo(id, slot)) { return false; }
  }
  return true;
}

fn comboModIndex(id: u32) -> u32 {
  // Reaction word indices: control, intensity, crit, qi, stability, success.
  var indices = array<u32, 6>(4u, 1u, 0u, 2u, 3u, 5u);
  return indices[id];
}
fn comboModValue(id: u32) -> f32 {
  var values = array<f32, 6>(25.0, 1.25, 1.25, 25.0, 25.0, 0.25);
  return values[id];
}

// mirror.mjs:2930-2947 — allowed next charges from all recipes containing the
// live prefix, returned as the recommendation bitmask.
fn nextChargeMask(c: ptr<function, array<f32, H_CHARGE_CAP>>,
                  count: u32) -> u32 {
  var mask = 0u;
  if (count == 0u || count >= 3u) { return mask; }
  for (var id = 0u; id < 6u; id = id + 1u) {
    var left = array<f32, 3>(
      alchemyCombo(id, 0u), alchemyCombo(id, 1u), alchemyCombo(id, 2u));
    var missing = false;
    for (var i = 0u; i < count; i = i + 1u) {
      var found = -1;
      for (var k = 0u; k < 3u; k = k + 1u) {
        if (left[k] == (*c)[i]) { found = i32(k); break; }
      }
      if (found < 0) { missing = true; break; }
      left[u32(found)] = 0.0;
    }
    if (missing) { continue; }
    for (var k = 0u; k < 3u; k = k + 1u) {
      if (left[k] > 0.0) { mask = mask | techniqueBit(left[k]); }
    }
  }
  return mask;
}

// True Bifang Flame: a triggered setState(key='blaze') on completionGained.
fn hasBlazeTrigger(id: f32) -> bool {
  if (id < 0.0) { return false; }
  let defBase = u32(id) * DEF_WORDS;
  let tb = u32(catalog[defBase + ${BUFF_DEF.triggerOffset}u]);
  let tc = u32(catalog[defBase + ${BUFF_DEF.triggerCount}u]);
  let eb = u32(catalog[defBase + ${BUFF_DEF.effectOffset}u]);
  for (var t = 0u; t < tc; t = t + 1u) {
    let trig = tb + t * BT_WORDS;
    if (u32(catalog[trig]) != BT_EVENT ||
        u32(catalog[trig + 1u]) != HASH_EVENT_COMPLETION) { continue; }
    let start = u32(catalog[trig + BT_EFFECT_OFFSET]);
    let count = u32(catalog[trig + BT_EFFECT_COUNT]);
    for (var k = 0u; k < count; k = k + 1u) {
      let e = eb + (start + k) * BE_WORDS;
      if (u32(catalog[e + BE_KIND]) == EFF_SET_STATE &&
          u32(catalog[e + BE_STATE_KEY]) == HASH_STATE_BLAZE) { return true; }
    }
  }
  return false;
}

// mirror.mjs:1517-1638 — only completionBoost/perfectionBoost are derived in
// the post-action frame; qi/stability boosts are unchanged in the corpus.
fn outputBoosts(nodeBase: u32, bonus: f32) -> vec2<f32> {
  var f: Frame;
  f.control = inState[nodeBase + NODE_CONTROL];
  f.intensity = inState[nodeBase + NODE_INTENSITY];
  f.critchance = inState[nodeBase + NODE_CRIT_CHANCE];
  f.critmultiplier = inState[nodeBase + NODE_CRIT_MULT];
  f.poolPctRaw = 100.0;
  f.stabPctRaw = 100.0;
  f.qiReduction = inState[nodeBase + ${NS_STATE.qiCostReduction}u];
  f.qiPenalty = inState[nodeBase + ${NS_STATE.qiCostPenalty}u];
  f.stabReduction = inState[nodeBase + ${NS_STATE.stabilityCostReduction}u];
  f.stabPenalty = inState[nodeBase + ${NS_STATE.stabilityCostPenalty}u];
  f.poolFlatRaw = inState[nodeBase + ${NS_STATE.poolCostFlat}u];
  f.boostCompletion = 0.0;
  f.boostPerfection = bonus * 10.0;
  f.boostStability = 0.0;
  f.boostQi = 0.0;
  f.controlMult = 1.0;
  f.intensityMult = 1.0;
  f.hasPoolPct = 0.0;
  f.hasStabPct = 0.0;
  f.successBonus = 0.0;
  f.critFactor = 1.0;
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    let entry = nodeBase + NODE_BUFFS + slot * SLOT_WORDS;
    let id = outState[entry];
    if (isStrippedBuff(id, nodeBase)) { continue; }
    foldBuff(&f, u32(id), outState[entry + 1u],
      outState[entry + 3u], outState[entry + 4u], nodeBase);
  }
  return vec2<f32>(f.boostCompletion, f.boostPerfection);
}

@compute @workgroup_size(64)
fn applyStateKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = gid.x;
  let nodeCount = arrayLength(&outState) / NS_STATE_WORDS;
  if (node >= nodeCount) { return; }
  let base = node * NS_STATE_WORDS;
  for (var k = 0u; k < NS_STATE_WORDS; k = k + 1u) {
    outState[base + k] = inState[base + k];
  }

  let skillIndex = u32(params[P_SKILL]);
  activeSkill = skillIndex;
  let header = skillIndex * SKILL_WORDS;
  let cost = expectedPaymentCosts(base, skillIndex);
  tr(0u, 0.0, cost.x, cost.y, inState[base + NODE_POOL],
    inState[base + NODE_POOL_PCT], f32(u32(skills[header + SKILL_COST_COUNT])));
  let flags = u32(skills[header + SKILL_FLAGS]);
  let hasEffects = (flags & FLAG_HAS_EFFECTS) != 0u;
  let preventsDecay = (flags & FLAG_PREVENT_DECAY) != 0u;
  // skills.ts:4400-4401 / 4610-4663: items without an explicit override do
  // not consume a turn and therefore do not tick step, cooldowns or durations.
  let consumesTurn = skills[header + SKILL_CONSUMES] > 0.5;
  let disciplinedTouch = (flags & FLAG_DISCIPLINED_TOUCH) != 0u;
  let stabilityGain = select(skills[header + SKILL_STABILITY_GAIN], 0.0, hasEffects);

  var outcomes: array<Outcome, OUTCOME_CAPACITY>;
  var outcomeCount = 1u;
  outcomes[0] = Outcome(
    inState[base + NODE_STABILITY] - cost.y + stabilityGain,
    -skills[header + SKILL_MAX_CHANGE], 1.0);

  // S4: parallel full distribution. The scalar path above is not touched.
  var proj: Projection;
  pInit(&proj, inState[base + NODE_STABILITY], inState[base + NODE_PENALTY]);
  var payProbs: array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>;
  var payCosts: array<f32, ${SKILL_COST_BRANCH_CAPACITY}u>;
  var payCount = 0u;
  paymentCostBranches(base, skillIndex, &payProbs, &payCosts, &payCount);
  if (payCount > 0u) { pPayBranches(&proj, &payProbs, &payCosts, payCount); }
  // skills.ts:4556-4575 — max-stability change and full restore happen before
  // masteries and before the success roll.
  if (skills[header + SKILL_MAX_CHANGE] != 0.0) {
    pChangeBoth(&proj, 0.0, -skills[header + SKILL_MAX_CHANGE], 1.0, false);
  }
  if (skills[header + ${SKILL_STATE.restoresMaxStabilityToFull}u] > 0.5) {
    pRestoreMaximum(&proj);
  }

  var pool = inState[base + NODE_POOL];
  let maxPool = params[P_MAX_POOL];
  var barCompRaw = 0.0;
  var barPerfRaw = 0.0;
  var dCompletion = 0.0;
  var dPerfection = 0.0;
  // Mastery bars are their own transitions: the technique bar clamps against
  // the state after them but before the buff-effect gains (skills.ts:5807-5815
  // adds those last and hard-caps the sum).
  var masteryComp = 0.0;
  var masteryPerf = 0.0;
  var barEvents: array<f32, BAR_EVENT_CAP>;
  var barEventCount = 0u;
  var mastery: MasteryState;
  mastery.controlBonus = 0.0;
  mastery.intensityBonus = 0.0;
  mastery.critChanceBonus = 0.0;
  mastery.critMultiplierBonus = 0.0;
  mastery.overrideCount = 0u;
  mastery.extraCount = 0u;

  // skills.ts:4331-4344 — "Runtime captures success after setup payment and
  // listeners, before masteries": the roll uses the buff frame as it stands
  // before phaseMasteryFrame, so a mastery-created buff does not raise this
  // action's own success. mastery is still empty here.
  let preMasterySuccessBonus = buildEffectFrame(base, &mastery).successBonus;
  // skills.ts:4300-4341 — one transition per mastery entry: the entry's effects
  // share one evaluation frame and the entry's own condition chance is the
  // transition's successChance, applied AFTER the critted gain is floored.
  var lastEntry: i32 = -1;
  var lastBarEntry: i32 = -1;
  var lastProjEntry: i32 = -1;
  var projEntryOpen = false;
  var entryFrame: Frame;

  // --- phaseMasteryFrame (mirror.mjs:2333-2459) ---
  let mCount = u32(skills[header + SKILL_MASTERY_COUNT]);
  let mBase = ${SKILL_STATE_MASTERY_OFF}u + skillIndex * ${SKILL_MASTERY_CAPACITY}u * M_WORDS;
  let createdBase = CREATED_BASE + skillIndex * CREATED_CAPACITY * CREATED_WORDS;
  for (var i = 0u; i < mCount; i = i + 1u) {
    let m = mBase + i * M_WORDS;
    let kind = u32(skills[m + M_KIND]);
    let chance = skills[m + M_CHANCE];
    // Upgrades are baked into the encoded payloads by records.ts.
    if (kind == M_OP_UPGRADE) { continue; }
    if (kind == M_OP_CONTROL) {
      mastery.controlBonus = mastery.controlBonus + skills[m + M_SCALING] * chance;
      continue;
    }
    if (kind == M_OP_INTENSITY) {
      mastery.intensityBonus = mastery.intensityBonus + skills[m + M_SCALING] * chance;
      continue;
    }
    if (kind == M_OP_CRIT_CHANCE) {
      mastery.critChanceBonus = mastery.critChanceBonus + skills[m + M_SCALING] * chance;
      continue;
    }
    if (kind == M_OP_CRIT_MULT) {
      mastery.critMultiplierBonus = mastery.critMultiplierBonus + skills[m + M_SCALING] * chance;
      continue;
    }
    if (kind != M_OP_EFFECT) { continue; }
    let effKind = u32(skills[m + M_EFFECT_KIND]);
    let entryId = i32(skills[m + M_ENTRY]);
    let entryChance = skills[m + M_ENTRY_CHANCE];
    if (entryId != lastEntry) {
      lastEntry = entryId;
      // Built at the entry's first record, before any buff this entry creates
      // joins mastery: a mid-technique creation only affects later work.
      entryFrame = buildEffectFrame(base, &mastery);
    }
    // records.ts encodes a createBuff mastery entry with effectRef >= 0 and
    // effectKind = 0 (addMastery leaves effectKind at its default there), so
    // the created-buff reference, not effKind, identifies that branch.
    let createdRefRaw = skills[m + M_EFFECT_REF];
    if (createdRefRaw >= 0.0) {
      let createdRef = u32(createdRefRaw);
      if (createdRef >= CREATED_CAPACITY) { continue; }
      let ce = createdBase + createdRef * CREATED_WORDS;
      let id = skills[ce + CREATED_ID];
      if (id < 0.0) { continue; }
      // skills.ts:3763-3764 — floor(value x factor x effectChance); zero joins nothing.
      let stacks = floor(skills[ce + CREATED_SCALING] * createdChance(ce, base));
      if (!(stacks > 0.0)) { continue; }
      var held = -1.0;
      for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
        let entry = base + NODE_BUFFS + slot * SLOT_WORDS;
        if (inState[entry] == id) { held = inState[entry + 1u]; break; }
      }
      let over = overrideStacks(&mastery, id);
      if (over >= 0.0) { held = over; }
      if (held >= 0.0) {
        let merged = held + stacks;
        let maxStacks = catalog[u32(id) * DEF_WORDS + ${BUFF_DEF.maxStacks}u];
        let capped = select(merged, min(merged, maxStacks), maxStacks > 0.0);
        setOverrideStacks(&mastery, id, max(0.0, capped));
      } else {
        var found = false;
        for (var e = 0u; e < mastery.extraCount; e = e + 1u) {
          if (mastery.extras[e].id == id) {
            mastery.extras[e].stacks = mastery.extras[e].stacks + stacks;
            found = true;
            break;
          }
        }
        if (!found && mastery.extraCount < CREATED_CAPACITY) {
          mastery.extras[mastery.extraCount].id = id;
          mastery.extras[mastery.extraCount].stacks = stacks;
          mastery.extraCount = mastery.extraCount + 1u;
        }
      }
      continue;
    }
    if (effKind == EFF_COMPLETION || effKind == EFF_PERFECTION) {
      // skills.ts:3235-3262 sums same-bar effects with only their own condition
      // factor; skills.ts:3105-3116 then floors the critted, boosted sum and
      // weights it by the transition's successChance (the entry condition,
      // skills.ts:4338-4341), flooring again.
      let tbC = dCompletion;
      let tbP = dPerfection;
      if (entryId != lastBarEntry) {
        lastBarEntry = entryId;
        var groupComp = 0.0;
        var groupPerf = 0.0;
        for (var j = i; j < mCount; j = j + 1u) {
          let mj = mBase + j * M_WORDS;
          if (u32(skills[mj + M_KIND]) != M_OP_EFFECT) { break; }
          if (i32(skills[mj + M_ENTRY]) != entryId) { break; }
          if (skills[mj + M_EFFECT_REF] >= 0.0) { continue; }
          let kj = u32(skills[mj + M_EFFECT_KIND]);
          if (kj != EFF_COMPLETION && kj != EFF_PERFECTION) { continue; }
          // chance = entry factor x effect factor; only the effect factor
          // multiplies the amount inside the sum.
          let effectChance = select(1.0, skills[mj + M_CHANCE] / entryChance, entryChance > 0.0);
          let scaled = evalScalingAt(mj + M_SCALING, &entryFrame, 0.0, 0.0, 0.0, 0u, base, true);
          let zeroFlag = (u32(skills[mj + ${SKILL_MASTERY.flags}u]) & 1u) != 0u;
          let effective = select(scaled, 0.0, scaled < 0.0 && zeroFlag);
          if (kj == EFF_COMPLETION) {
            groupComp = groupComp + effective * effectChance;
          } else {
            groupPerf = groupPerf + effective * effectChance;
          }
        }
        // skills.ts:3078-3093 — the engine clamps the critted gain to the
        // remaining headroom BEFORE the transition's success factor. Each
        // mastery entry is its own transition, so its current value already
        // includes the earlier entries.
        if (groupComp != 0.0) {
          var amount = boostedBar(groupComp * entryFrame.critFactor,
            entryFrame.boostCompletion, true);
          let remaining = max(0.0, params[P_MAX_COMPLETION] -
            inState[base + NODE_COMPLETION] - dCompletion);
          if (amount > 0.0) { amount = min(amount, remaining); }
          dCompletion = dCompletion + floor(amount * entryChance);
          masteryComp = masteryComp + floor(amount * entryChance);
          recordBarEvent(&barEvents, &barEventCount, amount);
        }
        if (groupPerf != 0.0) {
          var amount = boostedBar(groupPerf * entryFrame.critFactor,
            entryFrame.boostPerfection, true);
          let remaining = max(0.0, params[P_MAX_PERFECTION] -
            inState[base + NODE_PERFECTION] - dPerfection);
          if (amount > 0.0) { amount = min(amount, remaining); }
          dPerfection = dPerfection + floor(amount * entryChance);
          masteryPerf = masteryPerf + floor(amount * entryChance);
        }
      }
      tr(8u + i, 1.0, f32(effKind), chance,
        dCompletion - tbC, dPerfection - tbP, 0.0);
      continue;
    }
    let zeroFlag = (u32(skills[m + ${SKILL_MASTERY.flags}u]) & 1u) != 0u;
    let tbC = dCompletion; let tbP = dPerfection; let tbPool = pool;
    // skills.ts:4371-4380/5252 — a mastery entry is its own transition with
    // its own success chance; stability effects only land on its success
    // branch. Keep that split in the projection (the scalar EV path is
    // untouched).
    var projEffectChance = chance;
    var projSuccessOnly = false;
    if (effKind == EFF_STABILITY || effKind == EFF_MAX_STABILITY) {
      if (entryId != lastProjEntry) {
        if (projEntryOpen) { pEndTechnique(&proj); }
        pBeginTechnique(&proj, entryChance);
        lastProjEntry = entryId;
        projEntryOpen = true;
      }
      projEffectChance = select(1.0, chance / max(entryChance, 1.0e-9), entryChance > 0.0);
      projSuccessOnly = true;
    }
    applyNumeric(effKind, m + M_SCALING, chance, zeroFlag, true, base, &entryFrame,
      &pool, maxPool, &outcomes, &outcomeCount,
      &barCompRaw, &barPerfRaw, &barEvents, &barEventCount,
      &proj, projEffectChance, projSuccessOnly);
    tr(8u + i, 1.0, f32(effKind), chance, dCompletion - tbC, dPerfection - tbP, pool - tbPool);
  }

  if (projEntryOpen) { pEndTechnique(&proj); }
  // Payment is taken after the mastery phases (mirror.mjs:2683).
  pool = clamp(pool - cost.x, 0.0, maxPool);

  // --- phaseSuccessRoll: derived clamp recorded, export override applied -----
  // skills.ts:3171-3181 — total success is clamp(skill.successChance +
  // scalingVars.successChanceBonus). That frame already folds live buff
  // successChanceBonus stats (skills.ts:1905-2025), the harmony bonus
  // (skills.ts:2478) and condition chance bonuses (skills.ts:2090-2162); the
  // node's own word is only state.successChanceBonus (records.ts:94-98).
  let derivedSuccess = successRoll(
    skills[header + SKILL_SUCCESS],
    inState[base + NODE_SUCCESS_BONUS] + preMasterySuccessBonus +
      params[P_H_SUCCESS] + params[P_COND_CHANCE]);
  // mirror.mjs:2466/2476 — the corpus calls applySkill({successChance: 1}), so
  // the derived roll is recorded but the export override is the applied weight.
  let appliedSuccess = 1.0;
  // S4 fatal-detection override (safety.ts:229-238): a second dispatch forces
  // the technique roll to 0 or 1 while leaving every other phase identical.
  // 0/no override keeps the export corpus semantics; the depth-2 patch makes
  // appliedSuccess = derivedSuccess before this point.
  var appliedWeight = appliedSuccess;
  let forceSuccess = u32(params[P_FORCE_SUCCESS]);
  if (forceSuccess == 1u) { appliedWeight = 1.0; }
  else if (forceSuccess == 2u) { appliedWeight = 0.0; }

  // skills.ts:4582-4597 — scalar stability gain on techniques with no
  // stability effects. EV keeps the expected gain; the survival twin keeps the
  // realized roll (stability.ts:67-77).
  if (!hasEffects && stabilityGain != 0.0) {
    if (appliedWeight < 1.0) {
      pChangeExpected(&proj, stabilityGain * appliedWeight, stabilityGain, appliedWeight);
    } else {
      pChangeBoth(&proj, stabilityGain, 0.0, 1.0, false);
    }
  }

  // --- phaseApplyEffects (mirror.mjs:2481-2505) ---
  let effectCount = u32(skills[header + SKILL_EFFECT_COUNT]);
  let effectBase = ${SKILL_STATE_EFFECT_OFF}u + skillIndex * ${SKILL_EFFECT_CAPACITY}u * E_WORDS;
  var finalFrame = buildEffectFrame(base, &mastery);
  tr(5u, 7.0, finalFrame.control, finalFrame.intensity,
    finalFrame.controlMult, finalFrame.intensityMult, finalFrame.critFactor);
  // stability.ts:5252 — one success roll shared by every technique effect.
  pBeginTechnique(&proj, appliedWeight);
  for (var i = 0u; i < effectCount; i = i + 1u) {
    let e = effectBase + i * E_WORDS;
    let kind = u32(skills[e + E_KIND]);
    // skills.ts:3235-3262 sums bar effects with only the condition factor;
    // the success chance lands AFTER the critted gain is floored
    // (skills.ts:3105-3116: safeFloor(gainWithCrit) then x expectedFactor).
    let barEffect = kind == EFF_COMPLETION || kind == EFF_PERFECTION;
    let weight = select(
      skills[e + E_CHANCE] * appliedWeight,
      skills[e + E_CHANCE],
      barEffect);
    if (weight <= 0.0) { continue; }
    let zeroFlag = (u32(skills[e + E_FLAGS]) & 1u) != 0u;
    let tbC = dCompletion; let tbP = dPerfection; let tbPool = pool;
    applyNumeric(kind, e + E_SCALING, weight, zeroFlag, false, base, &finalFrame,
      &pool, maxPool, &outcomes, &outcomeCount,
      &barCompRaw, &barPerfRaw, &barEvents, &barEventCount,
      &proj, skills[e + E_CHANCE], true);
    tr(24u + i, 2.0, f32(kind), weight, dCompletion - tbC, dPerfection - tbP, pool - tbPool);
  }
  pEndTechnique(&proj);
  let qiRestore = skills[header + SKILL_QI_RESTORE];
  // skills.ts:4523-4537 — a scalar restoresQi action goes through
  // applyGainBoost(qiRestore, qiBoost) before it is added to the pool.
  if (qiRestore != 0.0) {
    pool = clamp(pool + applyBoost(qiRestore, finalFrame.boostQi), 0.0, maxPool);
  }
  tr(1u, 8.0, qiRestore, pool, barCompRaw, barPerfRaw, 0.0);

  // --- phase 5: buff effects + live-owner buff slots -------------------------
  var live: array<LiveSlot, LIVE_CAP>;
  var liveCount = 0u;
  var effectStacks: array<f32, SLOT_CAPACITY>;
  var effectAlive: array<u32, SLOT_CAPACITY>;
  var effectIds: array<f32, SLOT_CAPACITY>;
  // Initial pre-existing stacks, before this action's creations/addStack ticks;
  // buffCritFactor needs them to reconstruct the engine's live newBuffs stack
  // count (created stacks + tick delta + initial).
  var baseStacks: array<f32, SLOT_CAPACITY>;
  var effectViewCount = 0u;

  // skills.ts:4297-4322 pays skill.buffCost in a mastery phase BEFORE the
  // technique effects and before the per-turn buff tick. The owner loses the
  // floored stacks and is deleted at zero (skills.ts:4741-4749). The kernel
  // previously copied the input slots unchanged, so draw_the_temper left
  // banked_heat alive and a later availability pass still saw the cost
  // stack. buffCostAmount < 0 is consumeAll: spend the owner's live stacks
  // (skills.ts:4310-4313).
  let buffCostId = skills[header + SKILL_BUFF_COST_ID];
  var buffCostRemoval = floor(skills[header + SKILL_BUFF_COST_AMOUNT]);
  if (buffCostId >= 0.0 && buffCostRemoval < 0.0) {
    buffCostRemoval = 0.0;
    for (var costSlot = 0u; costSlot < SLOT_CAPACITY; costSlot = costSlot + 1u) {
      let costEntry = base + NODE_BUFFS + costSlot * SLOT_WORDS;
      if (inState[costEntry] == buffCostId) {
        buffCostRemoval = floor(inState[costEntry + 1u]);
        break;
      }
    }
  }

  // Node slots are in the engine's active-buff map order. Derived harmony
  // buffs are stripped from that map (mirror.mjs:2750-2753).
  for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
    let entry = base + NODE_BUFFS + slot * SLOT_WORDS;
    let id = inState[entry];
    if (id < 0.0) { continue; }
    var stacks = inState[entry + 1u];
    if (buffCostId >= 0.0 && id == buffCostId && buffCostRemoval > 0.0) {
      stacks = stacks - buffCostRemoval;
      // skills.ts:4745-4749 deletes the buff when the remaining stacks hit 0.
      if (stacks <= 0.0) { continue; }
    }
    live[liveCount].id = id;
    live[liveCount].stacks = stacks;
    live[liveCount].turns = inState[entry + 2u];
    live[liveCount].v0 = inState[entry + 3u];
    live[liveCount].v1 = inState[entry + 4u];
    live[liveCount].alive = select(1u, 0u, isDerivedDef(id, base));
    liveCount = liveCount + 1u;
  }
  let preExistingCount = liveCount;
  // skills.ts:4610-4628 ticks the two global duration counters. A counter
  // survives the last buff carrying its lifetime kind, so it lives in its own
  // node word rather than being inferred from buff slots.
  var controlTurns = inState[base + ${NS_STATE.controlBuffTurns}u];
  var intensityTurns = inState[base + ${NS_STATE.intensityBuffTurns}u];
  if (consumesTurn) {
    if (controlTurns > 0.0) { controlTurns = controlTurns - 1.0; }
    if (intensityTurns > 0.0) { intensityTurns = intensityTurns - 1.0; }
  }
  if (disciplinedTouch) { controlTurns = 0.0; intensityTurns = 0.0; }
  let grantType = u32(skills[header + SKILL_BUFF_TYPE]);
  // skills.ts:4626-4643 — a buff-granting technique replaces the matching
  // multiplier with its own override, else the config default.
  let skillBuffMult = skills[header + SKILL_BUFF_MULT];
  let defaultBuffMult = inState[base + NODE_DEFAULT_BUFF_MULT];
  if (grantType == BUFF_TYPE_CONTROL) {
    controlTurns = skills[header + SKILL_BUFF_DURATION];
    if (skillBuffMult != 0.0 && skillBuffMult != 1.0) {
      outState[base + NODE_C_BUFF_MULT] = skillBuffMult;
    } else if (defaultBuffMult != 0.0) {
      outState[base + NODE_C_BUFF_MULT] = defaultBuffMult;
    }
  } else if (grantType == BUFF_TYPE_INTENSITY) {
    intensityTurns = skills[header + SKILL_BUFF_DURATION];
    if (skillBuffMult != 0.0 && skillBuffMult != 1.0) {
      outState[base + NODE_I_BUFF_MULT] = skillBuffMult;
    } else if (defaultBuffMult != 0.0) {
      outState[base + NODE_I_BUFF_MULT] = defaultBuffMult;
    }
  }
  // Numeric view: ctx.liveStacks — pre-action stacks with mastery upserts.
  for (var slot = 0u; slot < preExistingCount; slot = slot + 1u) {
    let id = live[slot].id;
    effectIds[slot] = id;
    let over = overrideStacks(&mastery, id);
    effectStacks[slot] = select(live[slot].stacks, over, over >= 0.0);
    baseStacks[slot] = effectStacks[slot];
    effectAlive[slot] = select(1u, 0u, isDerivedDef(id, base));
    effectViewCount = effectViewCount + 1u;
  }
  for (var i = 0u; i < mastery.extraCount; i = i + 1u) {
    if (effectViewCount >= SLOT_CAPACITY) { break; }
    let e = mastery.extras[i];
    effectIds[effectViewCount] = e.id;
    effectStacks[effectViewCount] = select(e.stacks, 0.0, isDerivedDef(e.id, base));
    effectAlive[effectViewCount] = select(1u, 0u, isDerivedDef(e.id, base));
    effectViewCount = effectViewCount + 1u;
  }

  // Map rebuild (mirror.mjs:2842-2854): mastery creations first, then the
  // technique's creations. SKILL_CREATED_BUFF.chance already carries the
  // condition factor used by applyBuffStackEffects.
  let createdCount = u32(skills[header + SKILL_CREATED_COUNT]);
  for (var round = 1u; round <= 2u; round = round + 1u) {
    for (var i = 0u; i < createdCount; i = i + 1u) {
      let ce = createdBase + i * CREATED_WORDS;
      let fromMastery = (u32(skills[ce + CREATED_FLAGS]) & CREATED_FROM_MASTERY) != 0u;
      if ((round == 1u) != fromMastery) { continue; }
      let id = skills[ce + CREATED_ID];
      if (id < 0.0 || isDerivedDef(id, base)) { continue; }
      // skills.ts:5264 and 5338-5346 — a technique's createBuff stacks are
      // scaled by (actionSuccessChance * conditionResult.probability) before
      // upsertBuffIntoMap floors them. createdChance only carries the
      // condition factor, so the action's success weight must be folded in
      // here too. Without it a 0.55 technique creates Heightening 1x1 and
      // Luck Enhancement 1x5*0.3 on the device while the engine floors both
      // to zero; those phantom buffs then inflate the NEXT action's crit
      // multiplier and success bonus. Mastery entries use their own chance.
      let chance = createdChance(ce, base) *
        select(appliedWeight, 1.0, fromMastery);
      if (!(chance > 0.0)) { continue; }
      let delta = floor(skills[ce + CREATED_SCALING] * chance);
      if (!(delta > 0.0)) { continue; }
      let turns = select(0.0, floor(catalog[u32(id) * DEF_WORDS + ${BUFF_DEF.baseTurns}u]),
        catalog[u32(id) * DEF_WORDS + ${BUFF_DEF.lifetimeKind}u] > 0.5);
          liveUpsert(&live, &liveCount, id, delta, turns);
    }
  }

  var actionTrigger: u32 = 0u;
  let actionType = u32(skills[header + SKILL_ACTION_TYPE]);
  if (actionType == ACTION_FUSION) { actionTrigger = BT_ON_FUSION; }
  else if (actionType == ACTION_REFINE) { actionTrigger = BT_ON_REFINE; }
  else if (actionType == ACTION_STABILIZE) { actionTrigger = BT_ON_STABILIZE; }
  else if (actionType == ACTION_SUPPORT) { actionTrigger = BT_ON_SUPPORT; }

  // Numeric effects iterate the PRE-ACTION buffs only (mirror.mjs:2538). The
  // live-owner rule (skills.ts:5399-5405) re-reads the owner before EVERY
  // effect; a self addStack to zero or a negate skips all later effects.
  for (var slot = 0u; slot < preExistingCount; slot = slot + 1u) {
    if (effectAlive[slot] == 0u || isDerivedDef(effectIds[slot], base)) { continue; }
    let id = effectIds[slot];
    let defBase = u32(id) * DEF_WORDS;
    let trigBase = u32(catalog[defBase + ${BUFF_DEF.triggerOffset}u]);
    let trigCount = u32(catalog[defBase + ${BUFF_DEF.triggerCount}u]);
    let effBase = u32(catalog[defBase + ${BUFF_DEF.effectOffset}u]);
    // Engine order (skills.ts:5783-5790): action trigger first, then per-turn.
    for (var round = 0u; round < 2u; round = round + 1u) {
      let want = select(actionTrigger, BT_ON_ACTION, round == 1u);
      if (want == 0u) { continue; }
      for (var t = 0u; t < trigCount; t = t + 1u) {
        let tb = trigBase + t * BT_WORDS;
        if (u32(catalog[tb + BT_KIND]) != want) { continue; }
        let start = u32(catalog[tb + BT_EFFECT_OFFSET]);
        let cnt = u32(catalog[tb + BT_EFFECT_COUNT]);
        for (var k = 0u; k < cnt; k = k + 1u) {
          if (effectAlive[slot] == 0u) { break; }
          let ee = effBase + (start + k) * BE_WORDS;
          let kind = u32(catalog[ee + BE_KIND]);
          if (kind == 0u) { continue; }
          let ownerStacks = effectStacks[slot];
          let weight = buffChance(ee, ownerStacks);
          if (!(weight > 0.0)) { continue; }
          // skills.ts:5413 overlays the owner's live stacks before each effect.
          let value = evalScalingAt(ee + BE_SCALING, &finalFrame, ownerStacks,
            0.0, 0.0, u32(id), base, false);
          let tbC = dCompletion; let tbP = dPerfection; let tbPool = pool;
          if (kind == EFF_ADD_STACK && catalog[ee + BE_BUFF_TARGET] < 0.0) {
            // skills.ts:5586-5597: delta = stacks * chance, floored after add
            // (or floored removal); zero stacks delete the owner mid-tick.
            let delta = value * weight;
            if (delta != 0.0) {
              let maxStacks = catalog[defBase + ${BUFF_DEF.maxStacks}u];
              let next = select(ownerStacks + floor(delta), ownerStacks - floor(-delta), delta < 0.0);
              var capped = max(0.0, next);
              if (maxStacks > 0.0) { capped = min(capped, maxStacks); }
              effectStacks[slot] = capped;
              if (capped <= 0.0) { effectAlive[slot] = 0u; }
            }
            continue;
          }
          if (kind == EFF_NEGATE) {
            effectAlive[slot] = 0u;
            effectStacks[slot] = 0.0;
            continue;
          }
          if (kind == EFF_STABILITY) {
            // mirror.mjs:2079-2091 — every buff stability application branches.
            addBranchOutcome(&outcomes, &outcomeCount, value, 0.0, weight);
            // skills.ts:5531-5535 — buff change() is NOT success-gated.
            pChangeBoth(&proj, value, 0.0, weight, false);
          } else if (kind == EFF_MAX_STABILITY) {
            addBranchOutcome(&outcomes, &outcomeCount, 0.0, -value, weight);
            pChangeBoth(&proj, 0.0, -value, weight, false);
          } else if (kind == EFF_POOL) {
            // applyPoolDelta clamps at each application (skills.ts:4937).
            let amount = applyBoost(value * weight, finalFrame.boostQi);
            pool = clamp(pool + amount, 0.0, maxPool);
          } else if (kind == EFF_COMPLETION || kind == EFF_PERFECTION) {
            // Buff path has NO negative-result guard (skills.ts:5434-5443) and
            // crits its own raw amount with a separate buff crit factor, then
            // boosts with a floor only when active (numericApplications.ts:13-38).
            let raw = value * weight;
            var crit = 1.0;
            if (raw > 0.0) {
              crit = buffCritFactor(base, &live, liveCount, &effectStacks, &effectAlive,
                &effectIds, effectViewCount, &baseStacks, preExistingCount, &finalFrame);
            }
            let critted = raw * crit;
            if (kind == EFF_COMPLETION) {
              let amount = boostedBar(critted, finalFrame.boostCompletion, false);
              dCompletion = dCompletion + amount;
              recordBarEvent(&barEvents, &barEventCount, amount);
            } else {
              dPerfection = dPerfection + boostedBar(critted, finalFrame.boostPerfection, false);
            }
          }
          tr(64u + slot * 4u + k, 3.0, f32(kind), weight,
            dCompletion - tbC, dPerfection - tbP, pool - tbPool);
        }
      }
    }
  }

  // Buff-stack mutations on the output map (mirror.mjs:2836-2873). The
  // engine snapshots its pre-existing list BEFORE the mastery/skill creations, so a
  // buff created this turn does not tick its own effects in the same action.
  var mi = 0u;
  loop {
    if (mi >= preExistingCount) { break; }
    let slot = mi;
    mi = mi + 1u;
    if (live[slot].alive == 0u || isDerivedDef(live[slot].id, base)) { continue; }
    let id = live[slot].id;
    let defBase = u32(id) * DEF_WORDS;
    let trigBase = u32(catalog[defBase + ${BUFF_DEF.triggerOffset}u]);
    let trigCount = u32(catalog[defBase + ${BUFF_DEF.triggerCount}u]);
    let effBase = u32(catalog[defBase + ${BUFF_DEF.effectOffset}u]);
    for (var round = 0u; round < 2u; round = round + 1u) {
      let want = select(actionTrigger, BT_ON_ACTION, round == 1u);
      if (want == 0u) { continue; }
      for (var t = 0u; t < trigCount; t = t + 1u) {
        let tb = trigBase + t * BT_WORDS;
        if (u32(catalog[tb + BT_KIND]) != want) { continue; }
        let start = u32(catalog[tb + BT_EFFECT_OFFSET]);
        let cnt = u32(catalog[tb + BT_EFFECT_COUNT]);
        for (var k = 0u; k < cnt; k = k + 1u) {
          if (live[slot].alive == 0u) { break; }
          let ee = effBase + (start + k) * BE_WORDS;
          let kind = u32(catalog[ee + BE_KIND]);
          if (kind == 0u) { continue; }
          let ownerStacks = live[slot].stacks;
          let weight = buffChance(ee, ownerStacks);
          if (!(weight > 0.0)) { continue; }
          let buffTargetId = catalog[ee + BE_BUFF_TARGET];
          let value = evalScalingAt(ee + BE_SCALING, &finalFrame, ownerStacks,
            0.0, 0.0, u32(id), base, false);
          let delta = value * weight;
          if (kind == EFF_ADD_STACK) {
            liveAdjust(&live, &liveCount, select(id, buffTargetId, buffTargetId >= 0.0), delta);
          } else if (kind == EFF_NEGATE) {
            live[slot].alive = 0u;
            live[slot].stacks = 0.0;
          } else if (kind == EFF_CREATE_BUFF) {
            if (delta < 0.0) { liveAdjust(&live, &liveCount, buffTargetId, delta); }
            else { liveUpsert(&live, &liveCount, buffTargetId, delta, 0.0); }
          } else if (kind == EFF_CONSUME_BUFF) {
            liveConsume(&live, &liveCount, buffTargetId, delta);
          }
        }
      }
    }
  }

  // --- mirror.mjs:2688-2700, 2954-3054: harmony runs after the technique and
  // its per-turn buff effects, before the aggregate bar boost and cleanup.
  let hType = u32(inState[base + NODE_H_TYPE]);
  let canAdvance = params[P_SUBLIME] > 0.5 && consumesTurn &&
    (hType == H_FORGE || hType == H_ALCHEMICAL || hType == H_RESONANCE);
  if (canAdvance) {
    let action = u32(skills[header + SKILL_ACTION_TYPE]);
    var comboDone = false;
    var hDelta = 0.0;
    var hStability = 0.0;
    var hPool = 0.0;
    var hPenalty = 0.0;

    if (hType == H_FORGE) {
      var heat = clamp(inState[base + NODE_H_HEAT], 0.0, 10.0);
      var lastHeat = inState[base + NODE_H_LAST_HEAT];
      // mirror.mjs:2969 — absent lastBuffedHeat is seeded from the current
      // heat unless heat is the non-refreshing value 1.
      if (lastHeat < -0.5 && heat != 1.0) { lastHeat = heat; }
      heat = clamp(heat + select(-1.0, 2.0, action == ACTION_FUSION), 0.0, 10.0);
      if (heat != 1.0) { lastHeat = heat; }
      outState[base + NODE_H_HEAT] = heat;
      outState[base + NODE_H_LAST_HEAT] = lastHeat;
      outState[base + NODE_H_RECOMMEND] = select(14.0, 1.0, heat <= 4.0);
      // mirror.mjs:2975-2978 / getForgeHeatBand (1386-1392).
      if (heat >= 4.0 && heat <= 6.0) { hDelta = 10.0; }
      else if ((heat >= 2.0 && heat <= 3.0) || (heat >= 7.0 && heat <= 9.0)) { hDelta = -10.0; }
      else if (heat == 0.0 || heat == 10.0) { hDelta = -20.0; }
    } else if (hType == H_ALCHEMICAL) {
      var charges: array<f32, H_CHARGE_CAP>;
      var nCharges = 0u;
      for (var i = 0u; i < H_CHARGE_CAP; i = i + 1u) {
        let v = inState[base + NODE_H_CHARGES + i];
        if (v >= 0.0 && nCharges < H_CHARGE_CAP) {
          charges[nCharges] = v;
          nCharges = nCharges + 1u;
        }
      }
      if (action > 0u && nCharges < H_CHARGE_CAP) {
        charges[nCharges] = f32(action);
        nCharges = nCharges + 1u;
      }
      // aa.charges.sort() (mirror.mjs:2985); ids are already name-sorted.
      for (var i = 1u; i < nCharges; i = i + 1u) {
        var key = charges[i];
        var j = i;
        while (j > 0u && charges[j - 1u] > key) {
          charges[j] = charges[j - 1u];
          j = j - 1u;
        }
        charges[j] = key;
      }
      // Partial charges keep the carried reaction modifiers (mirror.mjs:2983).
      var reaction: array<f32, H_REACTION_CAP>;
      for (var i = 0u; i < H_REACTION_CAP; i = i + 1u) {
        reaction[i] = inState[base + NODE_H_REACTION + i];
      }
      var comboFound = -1;
      if (nCharges < 3u) {
        outState[base + NODE_H_RECOMMEND] = f32(nextChargeMask(&charges, nCharges));
      } else {
        comboDone = true;
        for (var i = 0u; i < H_REACTION_CAP; i = i + 1u) {
          reaction[i] = select(0.0, 1.0, i < 2u);
        }
        for (var id = 0u; id < 6u; id = id + 1u) {
          if (comboMatch(&charges, nCharges - 3u, id)) { comboFound = i32(id); }
        }
        if (comboFound >= 0) {
          hDelta = 20.0;
          let id = u32(comboFound);
          reaction[comboModIndex(id)] = comboModValue(id);
        } else {
          hDelta = -20.0;
          reaction[0] = 0.75;
        }
        for (var i = 0u; i < 3u; i = i + 1u) {
          outState[base + NODE_H_COMBO + i] = charges[nCharges - 3u + i];
          charges[nCharges - 3u + i] = -1.0;
        }
        nCharges = 0u;
        outState[base + NODE_H_RECOMMEND] = 0.0;
      }
      // lastCombo survives from the node unless a combo completed above.
      for (var i = 0u; i < H_CHARGE_CAP; i = i + 1u) {
        let v = select(-1.0, charges[i], i < nCharges);
        outState[base + NODE_H_CHARGES + i] = v;
      }
      for (var i = 0u; i < H_REACTION_CAP; i = i + 1u) {
        outState[base + NODE_H_REACTION + i] = reaction[i];
      }
    } else {
      var resType = inState[base + NODE_H_RESONANCE];
      var strength = inState[base + NODE_H_RES_STRENGTH];
      var pendingType = inState[base + NODE_H_PENDING];
      var pending = inState[base + NODE_H_PENDING_COUNT];
      if (resType < 0.0) {
        resType = f32(action);
        strength = 1.0;
        pendingType = -1.0;
        pending = 0.0;
      } else if (resType == f32(action)) {
        strength = strength + 1.0;
        pendingType = -1.0;
        pending = 0.0;
        hDelta = 3.0 * strength;
      } else {
        let isContinuing = pendingType == f32(action);
        let second = isContinuing && pending == 1.0;
        if (!second) {
          hDelta = -9.0;
          // mirror.mjs:3024-3028 — a non-second type change also costs 3 stability.
          hStability = -3.0;
          strength = max(0.0, strength - 1.0);
        }
        if (isContinuing) {
          pending = pending + 1.0;
          if (pending >= 2.0) {
            resType = f32(action);
            pendingType = -1.0;
            pending = 0.0;
          }
        } else {
          pendingType = f32(action);
          pending = 1.0;
        }
      }
      outState[base + NODE_H_RESONANCE] = resType;
      outState[base + NODE_H_RES_STRENGTH] = strength;
      outState[base + NODE_H_PENDING] = pendingType;
      outState[base + NODE_H_PENDING_COUNT] = pending;
      outState[base + NODE_H_RECOMMEND] = f32(techniqueBit(resType));
    }

    // The derived-subsystem gate is live state: forgeWorks exists, a resonance
    // has been seeded, or an alchemical combo has installed reaction modifiers.
    var derivedMask = u32(inState[base + NODE_H_DERIVED]);
    if (hType == H_FORGE && inState[base + NODE_H_LAST_HEAT] >= 0.0) { derivedMask = derivedMask | 1u; }
    if (hType == H_RESONANCE && outState[base + NODE_H_RESONANCE] >= 0.0) { derivedMask = derivedMask | 2u; }
    if (hType == H_ALCHEMICAL && comboDone) { derivedMask = derivedMask | 8u; }
    outState[base + NODE_H_DERIVED] = f32(derivedMask);

    outState[base + NODE_HARMONY] =
      clamp(inState[base + NODE_HARMONY] + hDelta, -100.0, 100.0);
    // skills.ts:5916-5926: generation harmony is UNCLAMPED and derives from
    // the previous clamped bar, unlike the display harmony above.
    // Export snapshots carry no generation harmony: records.ts falls back to
    // the clamped bar, so parity mode must do the same.
    outState[base + NODE_COND_GEN_HARMONY] = outState[base + NODE_HARMONY];
    if (EV_MODE) {
      outState[base + NODE_COND_GEN_HARMONY] =
        inState[base + NODE_HARMONY] + hDelta;
    }
    addScalarOutcome(&outcomes, &outcomeCount, hStability, hPenalty, 1.0);
    // skills.ts:5928-5955 — direct harmony state changes are scalar changes.
    pChangeBoth(&proj, hStability, hPenalty, 1.0, false);
    pool = clamp(pool + hPool, 0.0, maxPool);
    tr(120u, 12.0, f32(hType), f32(action), hDelta, hStability, pool);
  }

  // Compact surviving buffs into the node's slots; deleted/derived entries
  // shift later slots left, new buffs append at the end (mirror.mjs:2872).
  var outSlot = 0u;
  for (var i = 0u; i < liveCount; i = i + 1u) {
    if (live[i].alive == 0u) { continue; }
    let dst = base + NODE_BUFFS + outSlot * SLOT_WORDS;
    let kind = catalog[u32(live[i].id) * DEF_WORDS + ${BUFF_DEF.lifetimeKind}u];
    var turns = 0.0;
    if (kind == f32(${BUFF_LIFETIME.control}u)) { turns = controlTurns; }
    else if (kind == f32(${BUFF_LIFETIME.intensity}u)) { turns = intensityTurns; }
    outState[dst] = live[i].id;
    outState[dst + 1u] = live[i].stacks;
    outState[dst + 2u] = turns;
    outState[dst + 3u] = live[i].v0;
    outState[dst + 4u] = live[i].v1;
    outSlot = outSlot + 1u;
  }
  for (var i = outSlot; i < SLOT_CAPACITY; i = i + 1u) {
    let dst = base + NODE_BUFFS + i * SLOT_WORDS;
    outState[dst] = -1.0;
    outState[dst + 1u] = 0.0;
    outState[dst + 2u] = 0.0;
    outState[dst + 3u] = 0.0;
    outState[dst + 4u] = 0.0;
  }

  // mirror.mjs:2705-2714 — aggregate technique gains crit/boost/floor once.
  tr(2u, 4.0, barCompRaw, barPerfRaw, finalFrame.critFactor,
    finalFrame.boostCompletion, finalFrame.boostPerfection);
  // skills.ts:3105-3116 expectedProgressGain: floor the critted, boosted bar
  // FIRST, then weight it by the success chance and floor again. Weighting the
  // raw amount first (weight = chance x success above) moves the floor
  // boundary and loses a point on every fractional bar gain.
  // skills.ts:3078-3093 — clamp to the remaining headroom first; only then
  // weight by the success chance. Weighting first would erase the failure risk
  // of any gain that overshoots the bar.
  // The technique transition's current value is the state after the mastery
  // phases; buff-effect gains are added afterwards and only the final sum is
  // hard-capped (skills.ts:5807-5815), so they must not shrink this headroom.
  let mainCompHeadroom = max(0.0, params[P_MAX_COMPLETION] -
    inState[base + NODE_COMPLETION] - masteryComp);
  let mainPerfHeadroom = max(0.0, params[P_MAX_PERFECTION] -
    inState[base + NODE_PERFECTION] - masteryPerf);
  var mainComp = boostedBar(barCompRaw * finalFrame.critFactor,
    finalFrame.boostCompletion, true);
  var mainPerf = boostedBar(barPerfRaw * finalFrame.critFactor,
    finalFrame.boostPerfection, true);
  if (mainComp > 0.0) { mainComp = min(mainComp, mainCompHeadroom); }
  if (mainPerf > 0.0) { mainPerf = min(mainPerf, mainPerfHeadroom); }
  dCompletion = dCompletion + floor(mainComp * appliedWeight);
  dPerfection = dPerfection + floor(mainPerf * appliedWeight);
  tr(3u, 5.0, dCompletion, dPerfection, pool, 0.0, 0.0);

  // --- cleanup: caps, per-outcome penalty mean and stability floor ----------
  let compHeadroom = max(0.0, params[P_MAX_COMPLETION] - inState[base + NODE_COMPLETION]);
  let perfHeadroom = max(0.0, params[P_MAX_PERFECTION] - inState[base + NODE_PERFECTION]);
  if (dCompletion > compHeadroom) { dCompletion = compHeadroom; }
  if (dPerfection > perfHeadroom) { dPerfection = perfHeadroom; }

  let decay = select(1.0, 0.0, preventsDecay);
  let prevPenalty = inState[base + NODE_PENALTY];
  let initialMax = params[P_INITIAL_MAX];
  let maxStab = inState[base + NODE_MAX_STABILITY];
  var expectedStability = 0.0;
  var expectedPenalty = 0.0;
  for (var i = 0u; i < outcomeCount; i = i + 1u) {
    let o = outcomes[i];
    let penalty = clamp(ceil(prevPenalty + o.penaltyDelta + decay), 0.0, initialMax);
    let cleaned = floor(max(0.0, min(maxStab - (penalty - prevPenalty), o.stability)));
    expectedStability = expectedStability + o.probability * cleaned;
    expectedPenalty = expectedPenalty + o.probability * penalty;
  }
  let dMaxStability = prevPenalty - expectedPenalty;
  tr(4u, 6.0, dCompletion, dPerfection, compHeadroom, perfHeadroom, expectedStability);

  // S4: finish the parallel projection. The floor and P(s>0) come from the
  // surviving distribution (stability.ts:242-255); the conditional division is
  // deliberately left as raw f32 here and redone in f64 by the host.
  let fin = pFinish(&proj, initialMax, decay, consumesTurn);
  let pSurvive = fin.stats.survivalProbability;
  outState[base + NODE_FLOOR_STABILITY] = fin.stats.minStability;
  outState[base + NODE_FLOOR_MAX] = fin.stats.minMaximum;
  outState[base + NODE_SURV_PROB] = pSurvive;
  outState[base + NODE_SURV_STABILITY] = fin.stats.survivingStability;
  outState[base + NODE_SURV_PENALTY] = fin.stats.survivingPenalty;
  outState[base + NODE_COND_STABILITY] =
    select(0.0, fin.stats.survivingStability / max(pSurvive, 1.0e-30), pSurvive > 0.0);
  outState[base + NODE_COND_PENALTY] =
    select(0.0, fin.stats.survivingPenalty / max(pSurvive, 1.0e-30), pSurvive > 0.0);
  outState[base + NODE_OUTCOME_OVERFLOW] = f32(proj.overflow);
  outState[base + NODE_PROJECTION_COUNT] = f32(proj.evCount);
  outState[base + NODE_SURVIVAL_COUNT] = f32(proj.svCount);
  outState[base + NODE_SURVIVAL_TWIN] = f32(proj.hasSv);
  let dumpWhich = select(0u, 1u, proj.hasSv == 1u);
  let dumpCount = pCountVal(&proj, dumpWhich);
  for (var i = 0u; i < dumpCount && i < OUTCOME_CAPACITY; i = i + 1u) {
    let o = pRead(&proj, dumpWhich, i);
    var penalty = o.penalty;
    var stability = o.stability;
    if (consumesTurn) {
      penalty = clamp(ceil(penalty + decay), 0.0, initialMax);
      stability = floor(max(0.0, min(initialMax - penalty, stability)));
    }
    let d = base + NODE_PROJECTION_DEBUG + i * 3u;
    outState[d] = stability;
    outState[d + 1u] = penalty;
    outState[d + 2u] = o.probability;
  }

  outState[base + NODE_COMPLETION] = inState[base + NODE_COMPLETION] + dCompletion;
  outState[base + NODE_PERFECTION] = inState[base + NODE_PERFECTION] + dPerfection;
  outState[base + NODE_POOL] = pool;
  outState[base + NODE_STABILITY] = expectedStability;
  outState[base + NODE_MAX_STABILITY] = maxStab + dMaxStability;
  outState[base + NODE_PENALTY] = expectedPenalty;

  // mirror.mjs:3308-3317 — every turn-consuming action rebuilds Completion
  // Bonus from the post-action bar against targetCompletion.
  var newBonus = inState[base + NODE_BONUS];
  if (consumesTurn && params[P_TARGET_COMPLETION] > 0.0) {
    newBonus = max(0.0, bandGuaranteed(outState[base + NODE_COMPLETION],
      params[P_TARGET_COMPLETION]) - 1.0);
    outState[base + NODE_BONUS] = newBonus;
  }
  let boosts = outputBoosts(base, newBonus);
  outState[base + NODE_C_BOOST] = boosts.x;
  outState[base + NODE_P_BOOST] = boosts.y;

  // skills.ts:1125-1141, 5686-5710 — True Bifang Flame remembers the largest
  // single completionGained band-progress percentage since the last action.
  if (barEventCount > 0u && params[P_TARGET_COMPLETION] > 0.0) {
    var running = inState[base + NODE_COMPLETION];
    var bestPercent = 0.0;
    for (var i = 0u; i < barEventCount; i = i + 1u) {
      let afterBar = running + barEvents[i];
      let percent = (bandProgress(afterBar, params[P_TARGET_COMPLETION]) -
        bandProgress(running, params[P_TARGET_COMPLETION])) * 100.0;
      bestPercent = max(bestPercent, percent);
      running = afterBar;
    }
    let newBlaze = floor(bestPercent + 1.0e-5);
    for (var slot = 0u; slot < SLOT_CAPACITY; slot = slot + 1u) {
      let entry = base + NODE_BUFFS + slot * SLOT_WORDS;
      let id = outState[entry];
      if (!hasBlazeTrigger(id)) { continue; }
      outState[entry + 3u] = max(outState[entry + 3u], newBlaze);
    }
    tr(121u, 13.0, bestPercent, newBlaze, f32(barEventCount), 0.0, 0.0);
  }

  // --- post-action turn bookkeeping (skills.ts:4610-4663, 6031-6056) ---
  for (var cdSlot = 0u; cdSlot < ${BUFF_SKILL_CAPACITY}u; cdSlot = cdSlot + 1u) {
    let value = inState[base + NODE_COOLDOWNS + cdSlot];
    if (value <= 0.0) { continue; }
    if (!consumesTurn) { outState[base + NODE_COOLDOWNS + cdSlot] = value; }
    else if (value > 1.0) { outState[base + NODE_COOLDOWNS + cdSlot] = value - 1.0; }
    else { outState[base + NODE_COOLDOWNS + cdSlot] = 0.0; }
  }
  if (consumesTurn) {
    let skillCooldown = skills[header + SKILL_COOLDOWN];
    if (skillCooldown > 0.0) {
      outState[base + NODE_COOLDOWNS + skillIndex] = skillCooldown;
    }
  }
  outState[base + ${NS_STATE.controlBuffTurns}u] = controlTurns;
  outState[base + ${NS_STATE.intensityBuffTurns}u] = intensityTurns;
  outState[base + NODE_STEP] =
    inState[base + NODE_STEP] + select(0.0, 1.0, consumesTurn);
  outState[base + NODE_ALIVE] = select(0.0, 1.0, expectedStability > 0.0);
  var nodeFlags = 0u;
  if (outState[base + NODE_COMPLETION] >= params[P_TARGET_COMPLETION] &&
      outState[base + NODE_PERFECTION] >= params[P_TARGET_PERFECTION]) {
    nodeFlags = nodeFlags | 1u;
  }
  if (outState[base + NODE_POOL] <= 0.0) { nodeFlags = nodeFlags | 2u; }
  if (outState[base + NODE_STABILITY] <= 0.0) { nodeFlags = nodeFlags | 4u; }
  outState[base + NODE_FLAGS] = f32(nodeFlags);
}
`;

/** Compile the state kernel and fail loudly on any WGSL error. */
export async function createStateKernelPipeline(
  device: GpuDevice,
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: STATE_KERNEL_WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `WGSL: ${errors.map((error) => `${error.lineNum}:${error.message}`).join(' | ')}`,
    );
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'applyStateKernel' },
  });
}
