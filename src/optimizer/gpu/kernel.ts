/**
 * WGSL scoring + ranked-selection kernel.
 *
 * Ported verbatim from tools/gpu-probe/stage7.html (full-model childDeltas
 * from stage4.html/mirror.mjs plus the local top-K / global top-B beam), with
 * one added entry point (`scoreKernel`) for the S1 per-skill debug readout.
 * The shader contains integer opcodes only — all names and scaling payloads
 * are resolved on the CPU in records.ts.
 */

export const GPU_KERNEL_WGSL = `
@group(0) @binding(0) var<storage, read> skills: array<f32>;
@group(0) @binding(1) var<storage, read> initCombined: array<f32>;
@group(0) @binding(2) var<storage, read_write> inCombined: array<f32>;
@group(0) @binding(3) var<storage, read_write> outCombined: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<f32>;
@group(0) @binding(5) var<storage, read_write> top: array<f32>;
@group(0) @binding(6) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(7) var<storage, read> levelInfo: array<f32>;
@group(0) @binding(8) var<storage, read_write> trace: array<f32>;

const BN: u32 = __BN__u;
const AN: u32 = __AN__u;
const KN: u32 = __KN__u;
const HEAD: u32 = 20u;
const EFF_OFF: u32 = HEAD;
const COST_OFF: u32 = EFF_OFF + 84u;
const MENTRY_OFF: u32 = COST_OFF + 48u;
const FMOD_OFF: u32 = MENTRY_OFF + 112u;
const REC: u32 = FMOD_OFF + 224u;
const NODE: u32 = 38u;
const NS: u32 = 40u;
const CAND: u32 = 8u;
const OUT: u32 = 5u;
const MAX_EFFECTS: u32 = 12u;
const MAX_MENTRIES: u32 = 16u;
const MAX_OUTCOMES: u32 = 16u;

var<workgroup> wgScore: array<f32, 64>;
var<workgroup> wgSkill: array<u32, 64>;
var<workgroup> wgId: array<u32, 64>;
var<workgroup> wgDelta: array<vec4<f32>, 64>;
var<workgroup> wgComp: array<f32, 64>;

var<workgroup> wgKeyScore: array<f32, 64>;
var<workgroup> wgKeySkill: array<u32, 64>;
var<workgroup> wgKeyId: array<u32, 64>;

struct Deltas { dQi: f32, dStab: f32, dMaxStab: f32, dPerf: f32, dComp: f32 };

fn roundAway(v: f32, p: f32) -> f32 {
  return sign(v) * floor(abs(v) * p + 0.5) / p;
}
fn floorTo(v: f32) -> f32 { return floor(v); }
fn roundScaling(v: f32) -> f32 {
  if (v > 10.0) { return floor(v); }
  if (v < -10.0) { return ceil(v); }
  let p = select(100.0, 10000.0, abs(v) < 0.01);
  return roundAway(v, p);
}
fn applyBoost(v: f32, boost: f32) -> f32 {
  if (v > 0.0 && boost != 0.0) { return floorTo(v * (1.0 + boost / 100.0)); }
  return v;
}
fn boostedBar(critted: f32, boost: f32, floorUnboosted: f32) -> f32 {
  if (boost != 0.0 && critted > 0.0) { return floorTo(critted * (1.0 + boost / 100.0)); }
  if (floorUnboosted > 0.5) { return floorTo(critted); }
  return critted;
}
fn clampQi(p: f32, maxPool: f32) -> f32 { return clamp(p, 0.0, maxPool); }
fn makeCrit(cc: f32, cm: f32) -> f32 {
  let p = clamp(cc / 100.0, 0.0, 1.0);
  return (1.0 - p) + p * cm / 100.0;
}
fn statValue(flag: u32, control: f32, intensity: f32, maxpool: f32,
             maxstab: f32, completion: f32, perfection: f32,
             critchance: f32, critmultiplier: f32, stacks: f32) -> f32 {
  switch (flag) {
    case 0u: { return 1.0; }
    case 1u: { return control; }
    case 2u: { return intensity; }
    case 3u: { return maxpool; }
    case 4u: { return maxstab; }
    case 5u: { return completion; }
    case 6u: { return perfection; }
    case 7u: { return critchance; }
    case 8u: { return critmultiplier; }
    case 9u: { return stacks; }
    default: { return 0.0; }
  }
}
fn fmodValue(sb: u32, fb: u32, rawControl: f32, rawIntensity: f32,
             rawCC: f32, rawCM: f32, maxPool: f32, maxStability: f32,
             completionState: f32, perfectionState: f32) -> vec2<f32> {
  let f = sb + FMOD_OFF + fb * 7u;
  let field = skills[f];
  let stacks = skills[f + 6u];
  let statFlag = u32(skills[f + 2u]);
  let scalingFlag = u32(skills[f + 3u]);
  let scalingMax = skills[f + 4u];
  let maxCap = skills[f + 5u];
  var v = skills[f + 1u];
  if (statFlag != 0u) {
    v = v * statValue(statFlag, rawControl, rawIntensity, maxPool, maxStability,
                      completionState, perfectionState, rawCC, rawCM, stacks);
  }
  if (scalingFlag != 0u) {
    var sv = statValue(scalingFlag, rawControl, rawIntensity, maxPool, maxStability,
                       completionState, perfectionState, rawCC, rawCM, stacks);
    // Signed cap: negative is a lower bound (engine cap < 0 ? max : min).
    if (scalingMax > 0.0) { sv = min(sv, scalingMax); }
    else if (scalingMax < 0.0) { sv = max(sv, scalingMax); }
    v = v * sv;
  }
  v = roundScaling(v);
  if (maxCap > 0.0) { v = min(v, maxCap); }
  else if (maxCap < 0.0) { v = max(v, maxCap); }
  return vec2<f32>(field, v);
}
fn addMerged(pOut: ptr<function, array<f32, 16>>,
              sOut: ptr<function, array<f32, 16>>,
              qOut: ptr<function, array<f32, 16>>,
              nOut: ptr<function, u32>,
              prob: f32, stab: f32, pen: f32) {
  if (prob <= 0.0) { return; }
  let n = *nOut;
  for (var j = 0u; j < n; j = j + 1u) {
    if (abs((*sOut)[j] - stab) <= 0.001 && abs((*qOut)[j] - pen) <= 0.001) {
      (*pOut)[j] = (*pOut)[j] + prob;
      return;
    }
  }
  if (n >= MAX_OUTCOMES) { return; }
  (*pOut)[n] = prob;
  (*sOut)[n] = stab;
  (*qOut)[n] = pen;
  *nOut = n + 1u;
}
fn addOutcome(pOut: ptr<function, array<f32, 16>>,
               sOut: ptr<function, array<f32, 16>>,
               qOut: ptr<function, array<f32, 16>>,
               nOut: ptr<function, u32>,
               stab: f32, pen: f32, chance: f32) {
  if (chance <= 0.0) { return; }
  let n = *nOut;
  if (chance >= 1.0) {
    for (var j = 0u; j < n; j = j + 1u) {
      (*sOut)[j] = (*sOut)[j] + stab;
      (*qOut)[j] = (*qOut)[j] + pen;
    }
    return;
  }
  var tp: array<f32, 16>;
  var ts: array<f32, 16>;
  var tq: array<f32, 16>;
  var tn = 0u;
  for (var j = 0u; j < n; j = j + 1u) {
    addMerged(&tp, &ts, &tq, &tn, (*pOut)[j] * (1.0 - chance), (*sOut)[j], (*qOut)[j]);
    addMerged(&tp, &ts, &tq, &tn, (*pOut)[j] * chance, (*sOut)[j] + stab, (*qOut)[j] + pen);
  }
  for (var j = 0u; j < tn; j = j + 1u) {
    (*pOut)[j] = tp[j];
    (*sOut)[j] = ts[j];
    (*qOut)[j] = tq[j];
  }
  *nOut = tn;
}

fn childDeltas(sb: u32, nb: u32) -> Deltas {
  let pool = inCombined[nb + 0u];
  var curPool = pool;
  let stability = inCombined[nb + 1u];
  let qiFactor = inCombined[nb + 7u];
  let basePoolFlat = inCombined[nb + 36u];
  let buffCrit = inCombined[nb + 37u];
  let stabFactor = inCombined[nb + 8u];
  let poolPct0 = inCombined[nb + 9u];
  let stabPct0 = inCombined[nb + 10u];
  let maxPool = inCombined[nb + 11u];
  let maxStability = inCombined[nb + 12u];
  let completionState = inCombined[nb + 13u];
  let perfectionState = inCombined[nb + 14u];
  let completionHeadroom = inCombined[nb + 17u];
  let perfectionHeadroom = inCombined[nb + 18u];
  let controlCondMult = inCombined[nb + 23u];
  let intensityCondMult = inCombined[nb + 24u];
  let rawControl = inCombined[nb + 28u];
  let rawIntensity = inCombined[nb + 29u];
  let rawCC = inCombined[nb + 30u];
  let rawCM = inCombined[nb + 31u];
  // Skill record head slot 18 (SUCCESS_WEIGHT in records.ts) holds the BASE
  // skill roll only. The craft's own bonus lives in node slot 27
  // (NODE_SUCCESS_BONUS) and buff-created bonuses arrive as fmod field 11;
  // childDeltas folds them into accSuccessBonus below and clamps the sum.

  var qiCost = 0.0;
  var stabilityCost = 0.0;
  let branchCount = u32(skills[sb + 9u]);
  for (var c = 0u; c < branchCount; c = c + 1u) {
    let cb = sb + COST_OFF + c * 6u;
    let cp = skills[cb];
    let qiBase = skills[sb + 0u] + basePoolFlat + skills[cb + 5u];
    let qiChan = qiBase * (qiFactor * skills[cb + 1u]);
    let costQi = max(0.0, floorTo(qiChan * (poolPct0 * skills[cb + 3u])));
    let stabChan = skills[sb + 1u] * (stabFactor * skills[cb + 2u]);
    let costStab = floorTo(stabChan * (stabPct0 * skills[cb + 4u]));
    qiCost = qiCost + cp * costQi;
    stabilityCost = stabilityCost + cp * costStab;
  }

  let cBonusMul = skills[sb + 12u];
  let iBonusMul = skills[sb + 13u];
  let ccBonusAdd = skills[sb + 14u];
  let cmBonusAdd = skills[sb + 15u];
  var accControl = inCombined[nb + 2u];
  var accIntensity = inCombined[nb + 3u];
  var accCC = inCombined[nb + 4u];
  var accCM = inCombined[nb + 5u];
  var accBoostP = inCombined[nb + 19u];
  var accBoostC = inCombined[nb + 20u];
  var accBoostQ = inCombined[nb + 21u];
  var accBoostS = inCombined[nb + 22u];
  // Start from the node's own success bonus; created-buff fmods (field 11)
  // add to it within this technique's frame, mirroring skills.ts:2560.
  var accSuccessBonus = inCombined[nb + 27u];
  var accPoolPct = poolPct0;
  var accStabPct = stabPct0;

  var masteredStab = 0.0;
  var masteredPen = 0.0;
  var masteredPerfection = 0.0;
  var masteredCompletion = 0.0;
  var rawPerfection = 0.0;
  var rawCompletion = 0.0;
  var dPerfectionBuff = 0.0;
  var dCompletionBuff = 0.0;

  let mCount = u32(skills[sb + 16u]);
  var m = 0u;
  loop {
    if (m >= mCount) { break; }
    let mb = MENTRY_OFF + (sb + m * 7u);
    let op = u32(skills[mb]);
    let chance = skills[mb + 3u];
    if (op == 6u) {
      let fcnt = u32(skills[mb + 6u]);
      let fstart = u32(skills[mb + 5u]);
      for (var k = 0u; k < fcnt; k = k + 1u) {
        let mv = fmodValue(sb, fstart + k, rawControl, rawIntensity,
                           rawCC, rawCM, maxPool, maxStability,
                           completionState, perfectionState);
          let field = u32(mv.x);
          let v = mv.y;
          switch (field) {
            case 1u: { accControl = accControl + floorTo(v); }
            case 2u: { accIntensity = accIntensity + floorTo(v); }
            case 3u: { accCC = accCC + v; }
            case 4u: { accCM = accCM + v; }
            case 5u: { accPoolPct = floorTo(accPoolPct * v / 100.0); }
            case 6u: { accStabPct = floorTo(accStabPct * v / 100.0); }
            case 7u: { accBoostC = accBoostC + v; }
            case 8u: { accBoostP = accBoostP + v; }
            case 9u: { accBoostQ = accBoostQ + v; }
            case 10u: { accBoostS = accBoostS + v; }
            case 11u: { accSuccessBonus = accSuccessBonus + v; }
            default: {}
          }
        }
      m = m + 1u;
      continue;
    }
    if (chance > 0.0) {
      let frameControl = accControl * cBonusMul * controlCondMult;
      let frameIntensity = accIntensity * iBonusMul * intensityCondMult;
      let frameCC = accCC + ccBonusAdd;
      let frameCM = accCM + cmBonusAdd;
      let frameCrit = makeCrit(frameCC, frameCM);
      let value = roundScaling(skills[mb + 1u] * statValue(
        u32(skills[mb + 2u]), frameControl, frameIntensity, maxPool, maxStability,
        completionState, perfectionState, frameCC, frameCM, 0.0));
      let weight = value * chance;
      let zeroFlag = skills[mb + 4u];
      let effective = select(weight, 0.0, zeroFlag > 0.5 && weight < 0.0);
      switch (op) {
        case 1u: {
          masteredPerfection = masteredPerfection + boostedBar(effective * frameCrit, accBoostP, 1.0);
        }
        case 2u: {
          masteredCompletion = masteredCompletion + boostedBar(effective * frameCrit, accBoostC, 1.0);
        }
        case 3u: { masteredStab = masteredStab + weight; }
        case 4u: { masteredPen = masteredPen - weight; }
        case 5u: { curPool = clampQi(curPool + applyBoost(weight, accBoostQ), maxPool); }
        default: {}
      }
    }
    m = m + 1u;
  }

  // Engine clamp: clamp(base skill roll + folded success bonus, 0, 1).
  let successWeight = clamp(skills[sb + 18u] + accSuccessBonus, 0.0, 1.0);

  curPool = clampQi(curPool - qiCost, maxPool);

  let frameControl = accControl * cBonusMul * controlCondMult;
  let frameIntensity = accIntensity * iBonusMul * intensityCondMult;
  let frameCC = accCC + ccBonusAdd;
  let frameCM = accCM + cmBonusAdd;
  let frameCrit = makeCrit(frameCC, frameCM);
  accPoolPct = poolPct0;
  accStabPct = stabPct0;

  var pOut: array<f32, 16>;
  var sOut: array<f32, 16>;
  var qOut: array<f32, 16>;
  pOut[0] = 1.0;
  sOut[0] = stability - stabilityCost + masteredStab
            + select(skills[sb + 2u], 0.0, skills[sb + 6u] > 0.5);
  qOut[0] = -skills[sb + 3u] + masteredPen;
  var nOut = 1u;

  for (var e = 0u; e < MAX_EFFECTS; e = e + 1u) {
    let eb = sb + EFF_OFF + e * 7u;
    let op = u32(skills[eb]);
    if (op == 0u) { continue; }
    var chance = skills[eb + 3u];
    if (chance <= 0.0) { continue; }
    let critMode = skills[eb + 4u];
    let zeroFlag = skills[eb + 5u];
    let critRatio = skills[eb + 6u];
    // mirror.mjs phaseApplyEffects weights technique chances by the success
    // roll; phaseBuffEffects (critMode > 0) uses the condition chance alone.
    chance = select(chance * successWeight, chance, critMode > 0.5);
    var value = roundScaling(skills[eb + 1u] * statValue(
      u32(skills[eb + 2u]), frameControl, frameIntensity, maxPool, maxStability,
      completionState, perfectionState, frameCC, frameCM, 0.0));
    // Only bar gains discard a negative result from a positive-valued amount;
    // stability, max stability and pool keep their signed value.
    if (op == 1u || op == 2u) {
      value = select(value, 0.0, zeroFlag > 0.5 && value < 0.0);
    }
    let raw = value * chance;
    switch (op) {
      case 1u: {
        if (critMode > 0.5) {
          let critted = raw * (buffCrit * critRatio);
          dPerfectionBuff = dPerfectionBuff + boostedBar(critted, accBoostP, 0.0);
        } else { rawPerfection = rawPerfection + raw; }
      }
      case 2u: {
        if (critMode > 0.5) {
          let critted = raw * (buffCrit * critRatio);
          dCompletionBuff = dCompletionBuff + boostedBar(critted, accBoostC, 0.0);
        } else { rawCompletion = rawCompletion + raw; }
      }
      case 3u: { addOutcome(&pOut, &sOut, &qOut, &nOut, value, 0.0, chance); }
      case 4u: { addOutcome(&pOut, &sOut, &qOut, &nOut, 0.0, -value, chance); }
      case 5u: { curPool = clampQi(curPool + applyBoost(raw, accBoostQ), maxPool); }
      default: {}
    }
  }

  curPool = clampQi(curPool + skills[sb + 5u], maxPool);

  var dPerfection = masteredPerfection + dPerfectionBuff
                    + boostedBar(rawPerfection * frameCrit, accBoostP, 1.0);
  var dCompletion = masteredCompletion + dCompletionBuff
                    + boostedBar(rawCompletion * frameCrit, accBoostC, 1.0);
  dPerfection = min(dPerfection, perfectionHeadroom);
  dCompletion = min(dCompletion, completionHeadroom);

  let penaltyBase = inCombined[nb + 34u];
  let initialMax = inCombined[nb + 35u];
  let decay = select(1.0, 0.0, skills[sb + 4u] > 0.5);
  var expectedStability = 0.0;
  var expectedPenalty = 0.0;
  for (var j = 0u; j < nOut; j = j + 1u) {
    let penalty = min(initialMax, max(0.0, ceil(penaltyBase + qOut[j] + decay)));
    let cleaned = floor(max(0.0, min(maxStability - (penalty - penaltyBase), sOut[j])));
    expectedStability = expectedStability + pOut[j] * cleaned;
    expectedPenalty = expectedPenalty + pOut[j] * penalty;
  }
  let dStability = expectedStability - stability;
  let dMaxStability = penaltyBase - expectedPenalty;
  let dQi = curPool - pool;

  return Deltas(dQi, dStability, dMaxStability, dPerfection, dCompletion);
}

fn scoreOf(dQi: f32, dStab: f32, dMaxStab: f32, dPerf: f32, dComp: f32, stability: f32) -> f32 {
  var s = dComp * 3.0 + dPerf + dStab * 0.05;
  if (stability + dStab <= 0.0) { s = s - 1000000.0; }
  return s;
}
fn isWorse(sA: f32, kA: u32, sB: f32, kB: u32) -> bool {
  if (sA != sB) { return sA < sB; }
  return kA > kB;
}
fn isBetter(sA: f32, kA: u32, idA: u32, sB: f32, kB: u32, idB: u32) -> bool {
  if (sA != sB) { return sA > sB; }
  if (kA != kB) { return kA < kB; }
  return idA < idB;
}

@compute @workgroup_size(64)
fn scoreKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let children = arrayLength(&deltas) / OUT;
  if (idx >= children) { return; }
  let aCount = arrayLength(&skills) / REC;
  if (aCount == 0u) { return; }
  let node = idx / aCount;
  let a = idx % aCount;
  let d = childDeltas(a * REC, node * NS);
  let ob = idx * OUT;
  deltas[ob] = d.dQi;
  deltas[ob + 1u] = d.dStab;
  deltas[ob + 2u] = d.dMaxStab;
  deltas[ob + 3u] = d.dPerf;
  deltas[ob + 4u] = d.dComp;
}

@compute @workgroup_size(64)
fn selectKernel(@builtin(workgroup_id) wg: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
  let node = wg.x;
  if (node >= BN) { return; }
  let tid = lid.x;
  let nb = node * NS;
  let aCount = arrayLength(&skills) / REC;
  let a = tid;

  let isDead = inCombined[nb + NODE + 1u] < 0.5;
  // Step 0 honours the per-node opening slot (NODE_ROOT_SLOT in records.ts).
  // Every initial node is byte-identical, so without this mask the per-node
  // top-K selects the same topK openings everywhere and the beam never sees
  // the rest of the techniques.
  let openingSlot = u32(inCombined[nb + 26u]);
  let openingBlocked =
    inCombined[nb + NODE + 0u] < 0.5 && openingSlot < aCount && a != openingSlot;
  var myScore = -1.0e30;
  var mySkill = a;
  var myId = node * AN + a;
  var myDelta = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var myComp = 0.0;
  if (!isDead && a < aCount && !openingBlocked) {
    let d = childDeltas(a * REC, nb);
    myScore = scoreOf(d.dQi, d.dStab, d.dMaxStab, d.dPerf, d.dComp, inCombined[nb + 1u]);
    mySkill = a;
    myDelta = vec4<f32>(d.dQi, d.dStab, d.dMaxStab, d.dPerf);
    myComp = d.dComp;
  }

  wgScore[tid] = myScore;
  wgSkill[tid] = mySkill;
  wgId[tid] = myId;
  wgDelta[tid] = myDelta;
  wgComp[tid] = myComp;
  workgroupBarrier();

  for (var k = 2u; k <= 64u; k = k << 1u) {
    for (var j = k >> 1u; j > 0u; j = j >> 1u) {
      let partner = tid ^ j;
      if (partner > tid) {
        let mineWorse = isWorse(wgScore[tid], wgSkill[tid], wgScore[partner], wgSkill[partner]);
        let partnerWorse = isWorse(wgScore[partner], wgSkill[partner], wgScore[tid], wgSkill[tid]);
        let doSwap = select(partnerWorse, mineWorse, (tid & k) == 0u);
        if (doSwap) {
          let s0 = wgScore[tid]; let k0 = wgSkill[tid]; let i0 = wgId[tid];
          let d0 = wgDelta[tid]; let c0 = wgComp[tid];
          let s1 = wgScore[partner]; let k1 = wgSkill[partner]; let i1 = wgId[partner];
          let d1 = wgDelta[partner]; let c1 = wgComp[partner];
          wgScore[tid] = s1; wgSkill[tid] = k1; wgId[tid] = i1;
          wgDelta[tid] = d1; wgComp[tid] = c1;
          wgScore[partner] = s0; wgSkill[partner] = k0; wgId[partner] = i0;
          wgDelta[partner] = d0; wgComp[partner] = c0;
        }
      }
      workgroupBarrier();
    }
  }

  if (tid < KN) {
    let cb = (node * KN + tid) * CAND;
    cands[cb] = wgScore[tid];
    cands[cb + 1u] = f32(wgSkill[tid]);
    cands[cb + 2u] = wgDelta[tid].x;
    cands[cb + 3u] = wgDelta[tid].y;
    cands[cb + 4u] = wgDelta[tid].z;
    cands[cb + 5u] = wgDelta[tid].w;
    cands[cb + 6u] = wgComp[tid];
    cands[cb + 7u] = f32(wgId[tid]);
  }
}

@compute @workgroup_size(64)
fn selectTopKernel(@builtin(workgroup_id) wg: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wg.x * 64u;
  let candCount = arrayLength(&cands) / CAND;
  if (base >= candCount) { return; }
  let tid = lid.x;
  let cand = base + tid;
  var myScore = -1.0e31;
  var mySkill = 0xffffffffu;
  var myId = 0xffffffffu;
  if (cand < candCount) {
    let cb = cand * CAND;
    myScore = cands[cb];
    mySkill = u32(cands[cb + 1u]);
    myId = u32(cands[cb + 7u]);
  }
  var rank = 0u;
  var tile = 0u;
  loop {
    let tileBase = tile * 64u;
    if (tileBase >= candCount) { break; }
    let t = tileBase + tid;
    if (t < candCount) {
      let tb = t * CAND;
      wgKeyScore[tid] = cands[tb];
      wgKeySkill[tid] = u32(cands[tb + 1u]);
      wgKeyId[tid] = u32(cands[tb + 7u]);
    } else {
      wgKeyScore[tid] = -1.0e31;
      wgKeySkill[tid] = 0xffffffffu;
      wgKeyId[tid] = 0xffffffffu;
    }
    workgroupBarrier();
    if (cand < candCount) {
      for (var j = 0u; j < 64u; j = j + 1u) {
        if (tileBase + j < candCount &&
            isBetter(wgKeyScore[j], wgKeySkill[j], wgKeyId[j], myScore, mySkill, myId)) {
          rank = rank + 1u;
        }
      }
    }
    workgroupBarrier();
    tile = tile + 1u;
  }
  if (cand < candCount && rank < BN) {
    let cb = cand * CAND;
    let db = rank * CAND;
    for (var q = 0u; q < CAND; q = q + 1u) { top[db + q] = cands[cb + q]; }
  }
}

@compute @workgroup_size(64)
fn applyKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = gid.x;
  if (node >= BN) { return; }
  let cb = node * CAND;
  let srcId = u32(top[cb + 7u]);
  let srcNode = srcId / AN;
  let inBase = srcNode * NS;
  let outBase = node * NS;

  // Replay trace: entry (level, node) stores the parent node and the applied
  // skill. The level comes from a per-level storage buffer, not a uniform write:
  // all writes precede the single submit, so a shared uniform would end up
  // holding only the last level for every dispatch.
  let level = u32(levelInfo[0]);
  let tb = (level * BN + node) * 2u;
  trace[tb] = f32(srcNode);
  trace[tb + 1u] = f32(srcId % AN);

  var rec: array<f32, 40>;
  for (var k = 0u; k < NS; k = k + 1u) { rec[k] = inCombined[inBase + k]; }
  for (var k = 0u; k < NS; k = k + 1u) { outCombined[outBase + k] = rec[k]; }

  // Slot 25 traces the opening skill of the branch (NODE_ROOT in records.ts).
  // Step 0 stamps the skill just applied; every later level inherits the stamp
  // unchanged, so the final nodes can be grouped by root without extra state.
  outCombined[outBase + 25u] = select(rec[25u], f32(srcId % AN), rec[NODE + 0u] < 0.5);

  let parentAlive = rec[NODE + 1u];
  if (parentAlive < 0.5 || top[cb] <= -1.0e29) {
    outCombined[outBase + 1u] = 0.0;
    outCombined[outBase + NODE + 1u] = 0.0;
    return;
  }

  let maxPool = rec[11u];
  let initialMax = rec[35u];
  let capC = rec[15u];
  let capP = rec[16u];
  let pool = clamp(rec[0u] + top[cb + 2u], 0.0, maxPool);
  var stability = rec[1u] + top[cb + 3u];
  var alive = 1.0;
  if (stability <= 0.0) { stability = 0.0; alive = 0.0; }
  let maxStability = clamp(rec[12u] + top[cb + 4u], 0.0, initialMax);
  let completion = clamp(rec[13u] + top[cb + 6u], 0.0, capC);
  let perfection = clamp(rec[14u] + top[cb + 5u], 0.0, capP);
  let penaltyBase = clamp(rec[34u] - top[cb + 4u], 0.0, initialMax);

  outCombined[outBase + 0u] = pool;
  outCombined[outBase + 1u] = stability;
  outCombined[outBase + 12u] = maxStability;
  outCombined[outBase + 13u] = completion;
  outCombined[outBase + 14u] = perfection;
  outCombined[outBase + 17u] = max(0.0, capC - completion);
  outCombined[outBase + 18u] = max(0.0, capP - perfection);
  outCombined[outBase + 34u] = penaltyBase;
  outCombined[outBase + NODE + 0u] = rec[NODE + 0u] + 1.0;
  outCombined[outBase + NODE + 1u] = alive;
}

@compute @workgroup_size(64)
fn resetKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = gid.x;
  let count = arrayLength(&inCombined) / NS;
  if (node >= count) { return; }
  for (var k = 0u; k < NS; k = k + 1u) {
    inCombined[node * NS + k] = initCombined[node * NS + k];
  }
}
`;

export interface KernelDimensions {
  readonly beamWidth: number;
  readonly skillCount: number;
  readonly topK: number;
}

/** Substitute the three compile-time dimensions; nothing else is templated. */
export function buildKernelSource(dim: KernelDimensions): string {
  const positive = (value: number, fallback: number): number =>
    Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
  return GPU_KERNEL_WGSL.replace('__BN__', String(positive(dim.beamWidth, 1)))
    .replace('__AN__', String(positive(dim.skillCount, 1)))
    .replace('__KN__', String(positive(dim.topK, 1)));
}
