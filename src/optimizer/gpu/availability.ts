/**
 * S3 availability kernel.
 *
 * Port of `src/optimizer/skills.ts`:
 *   canApplySkill                 3550-3663
 *   getAvailableSkills            6073-6099
 *   getBlockedSkillReasons        6153-6301
 *   checkConditionRequirement     3509-3520
 *   getEffectiveQiCost            3525-4532
 *   getEffectiveStabilityCost     3537-3544
 *
 * Nothing in the engine is edited. Every value that the engine can only
 * produce on the host (the effective qi cost of `calculateEffectiveActionCosts`,
 * item stock, droplet stock, pills used this turn and the game-native
 * `canUseAction` precheck) crosses the boundary as a **named input** in the
 * per-candidate parameter table. The device, not the host, makes the
 * available/blocked decision from those numbers and the device node words.
 *
 * Two outputs are computed side by side because the engine's own diagnostic
 * function is not the exact complement of `canApplySkill`: droplets and
 * buff-requirement/buff-cost gates block availability but have no entry in
 * `getBlockedSkillReasons`. Reproducing that asymmetry is part of parity.
 */

import {
  BUFF_SLOT_CAPACITY,
  BUFF_SLOT_WORDS,
  NS_STATE,
  conditionKindOf,
} from './records';

/** Per-candidate skill record (host-encoded constants, no decisions). */
export const AVAIL_SKILL = {
  dropletCost: 0,
  toxicityCost: 1,
  conditionKind: 2,
  cooldownSkillIndex: 3,
  buffRequirementId: 4,
  buffRequirementAmount: 5,
  buffCostId: 6,
  buffCostRequired: 7,
  isItem: 8,
  reagentOnlyAtStepZero: 9,
  flags: 10,
  WORDS: 11,
} as const;

export const AVAIL_SKILL_FLAGS = {
  hasConditionRequirement: 1,
  /** Negative probe N6: remove one technique from the device's set. */
  disabled: 2,
} as const;

/** Per-candidate named inputs. These are numbers, never booleans decided by
 *  the stand in place of the device. */
export const AVAIL_PARAM = {
  qiDroplets: 0,
  itemRemaining: 1,
  consumedPillsThisTurn: 2,
  pillsPerRound: 3,
  /** Game-native `canUseAction`: 1 allow, 0 block. */
  nativePrecheck: 4,
  /** 1 when the host actually consulted a native provider. */
  nativePrecheckPresent: 5,
  /** `calculateEffectiveActionCosts(...).qiCost`, host-derived number. */
  effectiveQiCost: 6,
  /** `config.maxToxicity || 0`, config data. */
  maxToxicity: 7,
  /** `conditionKindOf(currentCondition)`, canonical condition kind. */
  currentConditionKind: 8,
  WORDS: 9,
} as const;

/** Per-candidate device readout. */
export const AVAIL_OUT = {
  available: 0,
  /** `getBlockedSkillReasons` category code, 0 when not blocked. */
  reason: 1,
  /** Which `canApplySkill` predicates fired (bit per predicate). */
  availableTrace: 2,
  /** Which `getBlockedSkillReasons` predicates fired. */
  reasonTrace: 3,
  WORDS: 4,
} as const;

export const AVAIL_TRACE = {
  stability: 1 << 0,
  droplet: 1 << 1,
  cooldown: 1 << 2,
  condition: 1 << 3,
  buffRequirement: 1 << 4,
  buffCost: 1 << 5,
  item: 1 << 6,
  qi: 1 << 7,
  toxicity: 1 << 8,
  native: 1 << 9,
  disabled: 1 << 10,
} as const;

/** Engine `SkillBlockedReason['reason']` -> numeric code. */
export const AVAIL_REASON = {
  none: 0,
  stability: 1,
  cooldown: 2,
  condition: 3,
  qi: 4,
  toxicity: 5,
} as const;

export function reasonCodeOf(reason: string): number {
  switch (reason) {
    case 'stability': return AVAIL_REASON.stability;
    case 'cooldown': return AVAIL_REASON.cooldown;
    case 'condition': return AVAIL_REASON.condition;
    case 'qi': return AVAIL_REASON.qi;
    case 'toxicity': return AVAIL_REASON.toxicity;
    default: return AVAIL_REASON.none;
  }
}

export const REASON_NAMES = ['none', 'stability', 'cooldown', 'condition', 'qi', 'toxicity'] as const;

/** Canonical condition names for device kind codes (conditions.ts aliases). */
export const CONDITION_KIND_NAMES = [
  'neutral', 'positive', 'negative', 'veryPositive', 'veryNegative', 'other',
] as const;

export function conditionKindCode(condition: string | undefined): number {
  return conditionKindOf(condition);
}

export function conditionNameOfKind(kind: number): string {
  return CONDITION_KIND_NAMES[kind] ?? 'other';
}

export const AVAILABILITY_KERNEL_WGSL = `
@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read> skills: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const NS_STATE_WORDS: u32 = ${NS_STATE.WORDS}u;
const NODE_STABILITY: u32 = ${NS_STATE.stability}u;
const NODE_POOL: u32 = ${NS_STATE.pool}u;
const NODE_STEP: u32 = ${NS_STATE.step}u;
const NODE_CONDITION: u32 = ${NS_STATE.conditionId}u;
const NODE_BUFFS: u32 = ${NS_STATE.buffs}u;
const NODE_COOLDOWNS: u32 = ${NS_STATE.cooldowns}u;
const NODE_TOXICITY: u32 = ${NS_STATE.toxicity}u;
const BUFF_SLOT_WORDS: u32 = ${BUFF_SLOT_WORDS}u;
const BUFF_SLOT_CAPACITY: u32 = ${BUFF_SLOT_CAPACITY}u;

const S_DROPLET: u32 = ${AVAIL_SKILL.dropletCost}u;
const S_TOXICITY: u32 = ${AVAIL_SKILL.toxicityCost}u;
const S_CONDITION: u32 = ${AVAIL_SKILL.conditionKind}u;
const S_COOLDOWN_INDEX: u32 = ${AVAIL_SKILL.cooldownSkillIndex}u;
const S_BUFF_REQ_ID: u32 = ${AVAIL_SKILL.buffRequirementId}u;
const S_BUFF_REQ_AMOUNT: u32 = ${AVAIL_SKILL.buffRequirementAmount}u;
const S_BUFF_COST_ID: u32 = ${AVAIL_SKILL.buffCostId}u;
const S_BUFF_COST_REQUIRED: u32 = ${AVAIL_SKILL.buffCostRequired}u;
const S_IS_ITEM: u32 = ${AVAIL_SKILL.isItem}u;
const S_REAGENT_STEP0: u32 = ${AVAIL_SKILL.reagentOnlyAtStepZero}u;
const S_FLAGS: u32 = ${AVAIL_SKILL.flags}u;
const S_WORDS: u32 = ${AVAIL_SKILL.WORDS}u;

const F_HAS_CONDITION: u32 = ${AVAIL_SKILL_FLAGS.hasConditionRequirement}u;
const F_DISABLED: u32 = ${AVAIL_SKILL_FLAGS.disabled}u;

const P_DROPLETS: u32 = ${AVAIL_PARAM.qiDroplets}u;
const P_ITEM_REMAINING: u32 = ${AVAIL_PARAM.itemRemaining}u;
const P_CONSUMED_PILLS: u32 = ${AVAIL_PARAM.consumedPillsThisTurn}u;
const P_PILLS_PER_ROUND: u32 = ${AVAIL_PARAM.pillsPerRound}u;
const P_NATIVE: u32 = ${AVAIL_PARAM.nativePrecheck}u;
const P_NATIVE_PRESENT: u32 = ${AVAIL_PARAM.nativePrecheckPresent}u;
const P_EFFECTIVE_QI: u32 = ${AVAIL_PARAM.effectiveQiCost}u;
const P_MAX_TOXICITY: u32 = ${AVAIL_PARAM.maxToxicity}u;
const P_CONDITION_KIND: u32 = ${AVAIL_PARAM.currentConditionKind}u;
const P_WORDS: u32 = ${AVAIL_PARAM.WORDS}u;

const OUT_WORDS: u32 = ${AVAIL_OUT.WORDS}u;
const R_NONE: f32 = ${AVAIL_REASON.none}.0;
const R_STABILITY: f32 = ${AVAIL_REASON.stability}.0;
const R_COOLDOWN: f32 = ${AVAIL_REASON.cooldown}.0;
const R_CONDITION: f32 = ${AVAIL_REASON.condition}.0;
const R_QI: f32 = ${AVAIL_REASON.qi}.0;
const R_TOXICITY: f32 = ${AVAIL_REASON.toxicity}.0;

const T_STABILITY: f32 = ${AVAIL_TRACE.stability}.0;
const T_DROPLET: f32 = ${AVAIL_TRACE.droplet}.0;
const T_COOLDOWN: f32 = ${AVAIL_TRACE.cooldown}.0;
const T_CONDITION: f32 = ${AVAIL_TRACE.condition}.0;
const T_BUFF_REQ: f32 = ${AVAIL_TRACE.buffRequirement}.0;
const T_BUFF_COST: f32 = ${AVAIL_TRACE.buffCost}.0;
const T_ITEM: f32 = ${AVAIL_TRACE.item}.0;
const T_QI: f32 = ${AVAIL_TRACE.qi}.0;
const T_TOXICITY: f32 = ${AVAIL_TRACE.toxicity}.0;
const T_NATIVE: f32 = ${AVAIL_TRACE.native}.0;
const T_DISABLED: f32 = ${AVAIL_TRACE.disabled}.0;

fn buffStacks(nodeBase: u32, id: f32) -> f32 {
  if (id < 0.0) { return 0.0; }
  for (var slot = 0u; slot < BUFF_SLOT_CAPACITY; slot = slot + 1u) {
    let entry = nodeBase + NODE_BUFFS + slot * BUFF_SLOT_WORDS;
    if (nodes[entry] >= 0.0 && abs(nodes[entry] - id) < 0.5) {
      return nodes[entry + 1u];
    }
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn availability(@builtin(global_invocation_id) gid: vec3<u32>) {
  let candidate = gid.x;
  let count = arrayLength(&params) / P_WORDS;
  if (candidate >= count) { return; }
  let nodeBase = candidate * NS_STATE_WORDS;
  let skillBase = candidate * S_WORDS;
  let paramBase = candidate * P_WORDS;

  let stability = nodes[nodeBase + NODE_STABILITY];
  let pool = nodes[nodeBase + NODE_POOL];
  let step = nodes[nodeBase + NODE_STEP];
  let toxicity = nodes[nodeBase + NODE_TOXICITY];
  let skillCond = skills[skillBase + S_CONDITION];
  let flags = u32(skills[skillBase + S_FLAGS]);
  let isItem = skills[skillBase + S_IS_ITEM] > 0.5;
  let cooldownIndex = skills[skillBase + S_COOLDOWN_INDEX];
  let conditionMismatch = (flags & F_HAS_CONDITION) != 0u &&
    abs(skillCond - params[paramBase + P_CONDITION_KIND]) > 0.5;
  let maxToxicity = params[paramBase + P_MAX_TOXICITY];
  let toxicityOver = maxToxicity > 0.0 && skills[skillBase + S_TOXICITY] > 0.0 &&
    toxicity + skills[skillBase + S_TOXICITY] > maxToxicity;
  let qiShort = pool < params[paramBase + P_EFFECTIVE_QI];
  let nativeBlocked = params[paramBase + P_NATIVE_PRESENT] > 0.5 &&
    params[paramBase + P_NATIVE] < 0.5;
  let pillsPerRound = max(1.0, floor(params[paramBase + P_PILLS_PER_ROUND]));

  var availableTrace = 0.0;
  if (nodes[nodeBase + NODE_STABILITY] <= 0.0) { availableTrace = availableTrace + T_STABILITY; }
  if (skills[skillBase + S_DROPLET] > params[paramBase + P_DROPLETS]) {
    availableTrace = availableTrace + T_DROPLET;
  }
  if (!isItem && cooldownIndex >= 0.0 &&
      nodes[nodeBase + NODE_COOLDOWNS + u32(cooldownIndex)] > 0.0) {
    availableTrace = availableTrace + T_COOLDOWN;
  }
  if (!isItem && conditionMismatch) { availableTrace = availableTrace + T_CONDITION; }
  if (!isItem && skills[skillBase + S_BUFF_REQ_ID] >= 0.0 &&
      buffStacks(nodeBase, skills[skillBase + S_BUFF_REQ_ID]) <
        skills[skillBase + S_BUFF_REQ_AMOUNT]) {
    availableTrace = availableTrace + T_BUFF_REQ;
  }
  if (!isItem && skills[skillBase + S_BUFF_COST_ID] >= 0.0 &&
      buffStacks(nodeBase, skills[skillBase + S_BUFF_COST_ID]) <
        skills[skillBase + S_BUFF_COST_REQUIRED]) {
    availableTrace = availableTrace + T_BUFF_COST;
  }
  if (isItem) {
    let itemKey = params[paramBase + P_ITEM_REMAINING];
    if (itemKey <= 0.0 ||
        (skills[skillBase + S_REAGENT_STEP0] > 0.5 && abs(step) > 0.5) ||
        params[paramBase + P_CONSUMED_PILLS] >= pillsPerRound) {
      availableTrace = availableTrace + T_ITEM;
    }
  }
  if (qiShort) { availableTrace = availableTrace + T_QI; }
  if (toxicityOver) { availableTrace = availableTrace + T_TOXICITY; }
  if (!isItem && nativeBlocked) { availableTrace = availableTrace + T_NATIVE; }
  if ((flags & F_DISABLED) != 0u) { availableTrace = availableTrace + T_DISABLED; }

  // Diagnostic mirror of skills.ts:6153-6301. It deliberately does NOT check
  // droplets, buffRequirement or buffCost, and it labels an exhausted item as
  // 'qi' — reproducing the engine's own reasons rather than fixing them.
  var reason = R_NONE;
  var reasonTrace = 0.0;
  if (stability <= 0.0) {
    reason = R_STABILITY;
    reasonTrace = T_STABILITY;
  } else if (!isItem && cooldownIndex >= 0.0 &&
             nodes[nodeBase + NODE_COOLDOWNS + u32(cooldownIndex)] > 0.0) {
    reason = R_COOLDOWN;
    reasonTrace = T_COOLDOWN;
  } else if (!isItem && conditionMismatch) {
    reason = R_CONDITION;
    reasonTrace = T_CONDITION;
  } else if (isItem && params[paramBase + P_ITEM_REMAINING] <= 0.0) {
    reason = R_QI;
    reasonTrace = T_ITEM;
  } else if (isItem && params[paramBase + P_CONSUMED_PILLS] >= pillsPerRound) {
    reason = R_CONDITION;
    reasonTrace = T_ITEM;
  } else if (isItem && skills[skillBase + S_REAGENT_STEP0] > 0.5 && abs(step) > 0.5) {
    reason = R_CONDITION;
    reasonTrace = T_ITEM;
  } else if (qiShort) {
    reason = R_QI;
    reasonTrace = T_QI;
  } else if (toxicityOver) {
    reason = R_TOXICITY;
    reasonTrace = T_TOXICITY;
  } else if (!isItem && nativeBlocked) {
    reason = R_CONDITION;
    reasonTrace = T_NATIVE;
  }

  out[candidate * OUT_WORDS] = select(1.0, 0.0, availableTrace > 0.5);
  out[candidate * OUT_WORDS + 1u] = reason;
  out[candidate * OUT_WORDS + 2u] = availableTrace;
  out[candidate * OUT_WORDS + 3u] = reasonTrace;
}
`;

/** Compile with compilation-info errors surfaced instead of silent no-op. */
export async function createAvailabilityKernelPipeline(
  device: GpuDevice,
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: AVAILABILITY_KERNEL_WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `availability WGSL: ${errors.map((e) => `${e.lineNum}:${e.message}`).join(' | ')}`,
    );
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'availability' },
  });
}
