/**
 * Condition-transition kernel (S4a).
 *
 * Port of src/optimizer/search/conditions.ts:
 *   getGeneratedConditionDistribution (90-168)
 *   pickBranchConditionDistribution    (170-187)
 *   getMostLikelyCondition              (189-193)
 *   getConditionTransitions             (195-247)
 *
 * Harmony and conditionDiscordance enter the distribution through continuous
 * arithmetic (harmony/100, 0.18/discordance, (harmony+100)/200), so the
 * formula is ported rather than baked into a lookup table. The decision and
 * the exact source lines are recorded in data/gpu-full-port-plan.md (S4a).
 *
 * One invocation handles one node. Output record:
 *   [0] branch count
 *   [1 + b*5 + 0] probability
 *   [1 + b*5 + 1] next condition kind
 *   [1 + b*5 + 2..4] next queue (3 slots)
 *
 * The compact kernel (kernel.ts), skills.ts and conditions.ts are untouched.
 */

import { CONDITION_KIND, CONDITION_QUEUE_SLOTS, NS_STATE } from './records';

export const CONDITION_KERNEL_WGSL = `
@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read_write> branches: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;

const NS_STATE_WORDS: u32 = ${NS_STATE.WORDS}u;
const NODE_QUEUE: u32 = ${NS_STATE.conditionQueue}u;
const QUEUE_SLOTS: u32 = ${CONDITION_QUEUE_SLOTS}u;
const DIST_CAP: u32 = 5u;
const BRANCH_WORDS: u32 = 5u;
const RECORD_WORDS: u32 = 1u + DIST_CAP * BRANCH_WORDS;

const C_NEUTRAL: u32 = ${CONDITION_KIND.neutral}u;
const C_POSITIVE: u32 = ${CONDITION_KIND.positive}u;
const C_NEGATIVE: u32 = ${CONDITION_KIND.negative}u;
const C_VERY_POSITIVE: u32 = ${CONDITION_KIND.veryPositive}u;
const C_VERY_NEGATIVE: u32 = ${CONDITION_KIND.veryNegative}u;

// params layout
const P_HARMONY: u32 = 0u;      // conditionGenerationHarmony (unclamped)
const P_DISCORDANCE: u32 = 1u;  // cfg.conditionDiscordance ?? 0
const P_ENABLE_BRANCH: u32 = 2u;
const P_BRANCH_LIMIT: u32 = 3u;
const P_MIN_PROB: u32 = 4u;
const P_CURRENT_KIND: u32 = 5u;

struct Dist {
  count: u32,
  cond: array<u32, 5>,
  prob: array<f32, 5>,
};

fn clamp01(v: f32) -> f32 {
  return clamp(v, 0.0, 1.0);
}

fn push(d: ptr<function, Dist>, cond: u32, p: f32) {
  if ((*d).count >= DIST_CAP) { return; }
  (*d).cond[(*d).count] = cond;
  (*d).prob[(*d).count] = p;
  (*d).count = (*d).count + 1u;
}

/**
 * normalizeConditionDistribution (conditions.ts:62-88): drop non-positive
 * entries, merge equal conditions, divide by the total, stable-sort descending
 * (JS Array.sort is stable, so ties keep insertion order).
 */
fn normalizeDist(d: ptr<function, Dist>) {
  var mCond: array<u32, 5>;
  var mProb: array<f32, 5>;
  var mCount = 0u;
  for (var i = 0u; i < (*d).count; i = i + 1u) {
    if ((*d).prob[i] <= 0.0) { continue; }
    var found = -1;
    for (var j = 0u; j < mCount; j = j + 1u) {
      if (mCond[j] == (*d).cond[i]) { found = i32(j); }
    }
    if (found < 0) {
      mCond[mCount] = (*d).cond[i];
      mProb[mCount] = (*d).prob[i];
      mCount = mCount + 1u;
    } else {
      mProb[u32(found)] = mProb[u32(found)] + (*d).prob[i];
    }
  }
  var total = 0.0;
  for (var i = 0u; i < mCount; i = i + 1u) { total = total + mProb[i]; }
  if (total <= 0.0) {
    (*d).count = 1u;
    (*d).cond[0] = C_NEUTRAL;
    (*d).prob[0] = 1.0;
    return;
  }

  var outCond: array<u32, 5>;
  var outProb: array<f32, 5>;
  var n = 0u;
  for (var i = 0u; i < mCount; i = i + 1u) {
    let p = mProb[i] / total;
    var pos = n;
    loop {
      if (pos == 0u) { break; }
      if (outProb[pos - 1u] >= p) { break; }
      outProb[pos] = outProb[pos - 1u];
      outCond[pos] = outCond[pos - 1u];
      pos = pos - 1u;
    }
    outProb[pos] = p;
    outCond[pos] = mCond[i];
    n = n + 1u;
  }
  (*d).count = n;
  for (var i = 0u; i < n; i = i + 1u) {
    (*d).cond[i] = outCond[i];
    (*d).prob[i] = outProb[i];
  }
}

fn allNeutral(queue: ptr<function, array<u32, 3>>, qLen: u32) -> bool {
  for (var i = 0u; i < qLen; i = i + 1u) {
    if ((*queue)[i] != C_NEUTRAL) { return false; }
  }
  return true;
}

/** Port of getGeneratedConditionDistribution (conditions.ts:90-168). */
fn generated(
  current: u32,
  queue: ptr<function, array<u32, 3>>,
  qLen: u32,
  harmony: f32,
  discordance: f32,
) -> Dist {
  var d: Dist;
  d.count = 0u;
  var negativeDelta = 0.0;
  var positiveDelta = 0.0;
  if (harmony < 0.0) { negativeDelta = abs(harmony) / 100.0; }
  if (harmony > 0.0) { positiveDelta = abs(harmony) / 100.0; }
  var last = 99u; // "undefined" sentinel: no last condition in the queue
  if (qLen > 0u) { last = (*queue)[qLen - 1u]; }

  if (discordance > 0.0) {
    let activeDiscordance = clamp01(discordance);
    let extreme = activeDiscordance * clamp01(0.18 / discordance);
    let mild = activeDiscordance - extreme;
    let positive = clamp01(0.5 + (harmony / 100.0) * 0.25);
    push(&d, C_NEUTRAL, 1.0 - activeDiscordance);
    push(&d, C_VERY_POSITIVE, extreme / 2.0);
    push(&d, C_VERY_NEGATIVE, extreme / 2.0);
    push(&d, C_POSITIVE, mild * positive);
    push(&d, C_NEGATIVE, mild * (1.0 - positive));
    normalizeDist(&d);
    return d;
  }
  if (last == C_VERY_POSITIVE || last == C_VERY_NEGATIVE) {
    push(&d, C_NEUTRAL, 1.0);
    normalizeDist(&d);
    return d;
  }
  if (last == C_POSITIVE) {
    let up = clamp01(0.3 * positiveDelta);
    push(&d, C_VERY_POSITIVE, up);
    push(&d, C_NEUTRAL, 1.0 - up);
    normalizeDist(&d);
    return d;
  }
  if (last == C_NEGATIVE) {
    let up = clamp01(0.3 * negativeDelta);
    push(&d, C_VERY_NEGATIVE, up);
    push(&d, C_NEUTRAL, 1.0 - up);
    normalizeDist(&d);
    return d;
  }

  var changeProbability = 0.0;
  if (current == C_NEUTRAL && allNeutral(queue, qLen)) {
    changeProbability = 1.0;
  } else {
    var neutralCount = 0u;
    var i = qLen;
    loop {
      if (i == 0u) { break; }
      if ((*queue)[i - 1u] != C_NEUTRAL) { break; }
      neutralCount = neutralCount + 1u;
      i = i - 1u;
    }
    changeProbability = clamp01(
      f32(neutralCount) * (0.15 + 0.15 * max(negativeDelta, positiveDelta)),
    );
  }
  let positiveChance = clamp01((harmony + 100.0) / 200.0);
  push(&d, C_NEUTRAL, 1.0 - changeProbability);
  push(&d, C_POSITIVE, changeProbability * positiveChance);
  push(&d, C_NEGATIVE, changeProbability * (1.0 - positiveChance));
  normalizeDist(&d);
  return d;
}

/** Port of pickBranchConditionDistribution (conditions.ts:170-187). */
fn pickBranch(d: ptr<function, Dist>, branchEnabled: f32, branchLimit: f32, minProb: f32) {
  if (branchEnabled < 0.5) {
    let c = (*d).cond[0];
    (*d).count = 1u;
    (*d).cond[0] = c;
    (*d).prob[0] = 1.0;
    return;
  }
  var kept: Dist;
  kept.count = 0u;
  for (var i = 0u; i < (*d).count; i = i + 1u) {
    // conditions.ts:165-168 compares the f64 product with the f64 threshold.
    // The device computes the same product in f32: at harmony -70 the
    // positive branch is exactly 0.15 in the engine but one f32 ulp below the
    // threshold here, so it was dropped (264 missing branches). Treat values
    // within a few f32 ulps of the threshold as meeting it; this is numeric
    // parity for the threshold comparison, not a wider branch threshold.
    if ((*d).prob[i] + 1.0e-7 >= minProb) {
      push(&kept, (*d).cond[i], (*d).prob[i]);
    }
  }
  let lim = max(1u, u32(floor(branchLimit)));
  var limited: Dist;
  limited.count = 0u;
  if (kept.count > 0u) {
    for (var i = 0u; i < kept.count && i < lim; i = i + 1u) {
      push(&limited, kept.cond[i], kept.prob[i]);
    }
  } else {
    for (var i = 0u; i < (*d).count && i < lim; i = i + 1u) {
      push(&limited, (*d).cond[i], (*d).prob[i]);
    }
  }
  normalizeDist(&limited);
  *d = limited;
}

@compute @workgroup_size(64)
fn conditionTransitions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = gid.x;
  let nodeCount = arrayLength(&nodes) / NS_STATE_WORDS;
  if (node >= nodeCount) { return; }
  let base = node * NS_STATE_WORDS;
  let outBase = node * RECORD_WORDS;

  let harmony = params[P_HARMONY];
  let discordance = params[P_DISCORDANCE];
  let branchEnabled = params[P_ENABLE_BRANCH];
  let branchLimit = params[P_BRANCH_LIMIT];
  let minProb = params[P_MIN_PROB];
  let current = u32(params[P_CURRENT_KIND]);

  var queue: array<u32, 3>;
  var qLen = 0u;
  for (var i = 0u; i < QUEUE_SLOTS; i = i + 1u) {
    queue[i] = u32(nodes[base + NODE_QUEUE + i]);
    qLen = qLen + 1u;
  }

  var d: Dist;
  var shifted: array<u32, 3>;
  var nextCondition: u32 = 0u;

  if (qLen > 0u) {
    nextCondition = queue[0];
    for (var i = 0u; i < QUEUE_SLOTS - 1u; i = i + 1u) {
      shifted[i] = queue[i + 1u];
    }
    d = generated(nextCondition, &shifted, QUEUE_SLOTS - 1u, harmony, discordance);
    pickBranch(&d, branchEnabled, branchLimit, minProb);
  } else {
    var empty: array<u32, 3>;
    d = generated(current, &empty, 0u, harmony, discordance);
    pickBranch(&d, branchEnabled, branchLimit, minProb);
  }

  var count = d.count;
  branches[outBase] = f32(count);
  for (var b = 0u; b < count; b = b + 1u) {
    let o = outBase + 1u + b * BRANCH_WORDS;
    branches[o] = d.prob[b];
    var nextQueue: array<u32, 3>;
    if (qLen > 0u) {
      nextCondition = queue[0];
      nextQueue[0] = shifted[0];
      nextQueue[1] = shifted[1];
      nextQueue[2] = d.cond[b];
    } else {
      nextCondition = d.cond[b];
      var appended: array<u32, 3>;
      let ad = generated(d.cond[b], &appended, 0u, harmony, discordance);
      nextQueue[0] = ad.cond[0];
      nextQueue[1] = 0u;
      nextQueue[2] = 0u;
    }
    branches[o + 1u] = f32(nextCondition);
    branches[o + 2u] = f32(nextQueue[0]);
    branches[o + 3u] = f32(nextQueue[1]);
    branches[o + 4u] = f32(nextQueue[2]);
  }
  // Clear stale records from a previous dispatch with a larger branch count.
  for (var b = count; b < DIST_CAP; b = b + 1u) {
    let o = outBase + 1u + b * BRANCH_WORDS;
    for (var k = 0u; k < BRANCH_WORDS; k = k + 1u) { branches[o + k] = 0.0; }
  }
}
`;

/** Compile the condition kernel and fail loudly on any WGSL error. */
export async function createConditionKernelPipeline(
  device: GpuDevice,
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: CONDITION_KERNEL_WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `WGSL: ${errors.map((error) => `${error.lineNum}:${error.message}`).join(' | ')}`,
    );
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'conditionTransitions' },
  });
}
