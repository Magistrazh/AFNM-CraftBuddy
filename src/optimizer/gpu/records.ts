/**
 * Numeric bridge for the GPU backend.
 *
 * Port of the CPU preparation in tools/gpu-probe/stage4.html (which mirrors
 * tools/gpu-probe/mirror.mjs 179/179): resolve buffs, scaling payloads,
 * conditions and harmony into flat f32 records; the shader sees opcodes only.
 * Pure: no device, no state mutation, no game access.
 */

import { getConditionEffectsForConfig, type OptimizerConfig, type SkillDefinition } from '../skills';
import type { CraftingState } from '../state';
import type {
  BuffDefinition,
  ConditionEffect,
  HarmonyData,
  Scaling,
} from '../gameTypes';

// Layout constants — must stay in sync with kernel.ts / stage7.html.
export const OPS = {
  perfection: 1,
  completion: 2,
  stability: 3,
  maxStability: 4,
  pool: 5,
  createBuff: 6,
} as const;

export const STATS: Readonly<Record<string, number>> = {
  flat: 0,
  control: 1,
  intensity: 2,
  maxpool: 3,
  maxstability: 4,
  completion: 5,
  perfection: 6,
  critchance: 7,
  critmultiplier: 8,
  stacks: 9,
};

const F = {
  control: 1,
  intensity: 2,
  critchance: 3,
  critmultiplier: 4,
  poolPct: 5,
  stabPct: 6,
  boostCompletion: 7,
  boostPerfection: 8,
  boostQi: 9,
  boostStability: 10,
  successBonus: 11,
} as const;

const FKEY: Readonly<Record<string, number>> = {
  control: F.control,
  intensity: F.intensity,
  critchance: F.critchance,
  critmultiplier: F.critmultiplier,
  poolCostPercentage: F.poolPct,
  stabilityCostPercentage: F.stabPct,
  completionBoost: F.boostCompletion,
  perfectionBoost: F.boostPerfection,
  qiBoost: F.boostQi,
  stabilityBoost: F.boostStability,
  successChanceBonus: F.successBonus,
};

export const MAX_EFFECTS = 12;
export const MAX_MENTRIES = 16;
export const MAX_FMODS = 32;
/** Must match kernel.ts `MAX_OUTCOMES` (the stability-outcome merge buffer). */
export const MAX_OUTCOMES = 16;
export const MAX_COST_BRANCHES = 8;
export const HEAD = 20;
/** Skill-record head slot holding the derived per-skill success roll. */
export const SUCCESS_WEIGHT = 18;
export const EFF_OFF = HEAD;
export const COST_OFF = EFF_OFF + MAX_EFFECTS * 7;
export const MENTRY_OFF = COST_OFF + MAX_COST_BRANCHES * 6;
export const FMOD_OFF = MENTRY_OFF + MAX_MENTRIES * 7;
export const REC = FMOD_OFF + MAX_FMODS * 7;
export const NODE = 38;
/** Node slot 25 traces the opening skill of the branch. */
export const NODE_ROOT = 25;
/**
 * Root seeding slot: node i is allowed to open with skill `i % skillCount`.
 * Without it every initial node is identical, the per-node top-K selects the
 * same skills everywhere, and the beam never considers more than topK openings.
 */
export const NODE_ROOT_SLOT = 26;
/**
 * Node slot 27 carries the craft's own `successChanceBonus`. It is the base of
 * the roll: the shader adds live buff `successChanceBonus` fmods (field 11),
 * the harmony bonus (param 29) and condition `chance` bonuses (param 30) and
 * clamps the sum with the base skill roll (head slot 18) — the same clamp as
 * skills.ts:3171-3181, taken before the mastery phases (skills.ts:4331-4344).
 */
export const NODE_SUCCESS_BONUS = 27;
export const EXTRA = 2;
export const NS = NODE + EXTRA;
export const OUT = 5;
export const CAND = 8;

type Raw = Record<string, unknown>;
type Vars = Record<string, number>;
type UpgradeMap = Record<string, { additive: number; multiplier: number }>;

interface MirrorState {
  pool: number;
  maxPool: number;
  stability: number;
  maxStability: number;
  control: number;
  intensity: number;
  critChance: number;
  critMultiplier: number;
  overcrit: number;
  overcritCapBonus: number;
  successChanceBonus: number;
  qiCostReduction: number;
  qiCostPenalty: number;
  stabilityCostReduction: number;
  stabilityCostPenalty: number;
  poolCostFlat: number;
  poolCostPercentage: number;
  stabilityCostPercentage: number;
  completion: number;
  perfection: number;
  completionBonus: number;
  stabilityPenalty: number;
  initialMaxStability: number;
  /** skills.ts:1924-1940 turn-counter control/intensity state. */
  controlBuffTurns: number;
  controlBuffMultiplier: number;
  intensityBuffTurns: number;
  intensityBuffMultiplier: number;
  harmonyType?: string;
  harmonyData?: HarmonyData;
  nativeVariables?: Record<string, number>;
}

interface MirrorBuff {
  key: string;
  name: string;
  stacks: number;
  definition?: BuffDefinition;
  stats: Record<string, Scaling>;
  effects: unknown[];
  onFusion?: unknown[];
  onRefine?: unknown[];
  onStabilize?: unknown[];
  onSupport?: unknown[];
}

interface MirrorData {
  state: MirrorState;
  buffs: MirrorBuff[];
  conditionEffects: ConditionEffect[];
  caps: { completion: number; perfection: number };
  targetCompletion: number;
  targetPerfection: number;
  skills: SkillDefinition[];
}

interface EvalFrame {
  buffed: {
    control: number;
    intensity: number;
    critchance: number;
    critmultiplier: number;
  };
  controlMult: number;
  intensityMult: number;
  critFactor?: number;
}

interface EvalCtx {
  data: MirrorData;
  upgrades: UpgradeMap;
  frame: EvalFrame;
  note: (kind: string, name: string) => void;
}

interface Frame {
  buffed: {
    control: number;
    intensity: number;
    critchance: number;
    critmultiplier: number;
    poolCostPercentage: number;
    stabilityCostPercentage: number;
  };
  boosts: {
    completion: number;
    perfection: number;
    stability: number;
    qi: number;
  };
  controlMult: number;
  intensityMult: number;
  critFactor: number;
  poolCostFlat: number;
  qiFactor: number;
  stabFactor: number;
  poolPct: number;
  stabPct: number;
}

interface ExtraBuff {
  name?: string;
  stacks: number;
  stats: Record<string, Scaling>;
}

interface PaymentBranch {
  probability: number;
  qiRatio: number;
  stabRatio: number;
  poolRatio: number;
  stabPctRatio: number;
  extraFlat: number;
}

interface CritContext {
  liveStacks?: Record<string, number>;
  masteryExtraBuffs?: ExtraBuff[];
  upgrades?: UpgradeMap;
  frame?: { critFactor?: number };
}

function obj(value: unknown): Raw | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Raw)
    : undefined;
}

function num(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function normalizeBuffName(value: unknown): string {
  return str(value).trim().toLowerCase().replace(/\s+/g, '_');
}

const f = (value: number): number => Math.fround(value);
const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));
const roundAway = (value: number, precision: number): number =>
  f((value < 0 ? -1 : 1) * (Math.floor(Math.abs(value) * precision + 0.5) / precision));
const floorTo = (value: number): number => Math.floor(value);
const floorToDouble = (value: number): number =>
  Math.floor(Math.round(value * 1e10) / 1e10);
const roundScaling = (value: number): number => {
  if (!Number.isFinite(value)) return 0;
  if (value > 10) return Math.floor(value);
  if (value < -10) return Math.ceil(value);
  return roundAway(value, Math.abs(value) < 0.01 ? 1e4 : 1e2);
};

const DEFAULT_HARMONY_MODS = {
  controlMultiplier: 1,
  intensityMultiplier: 1,
  critChanceBonus: 0,
  successChanceBonus: 0,
  qiCostReduction: 0,
  stabilityCostReduction: 0,
};

const DERIVED_HARMONY_BUFF_NAMES: Readonly<Record<string, readonly string[]>> = {
  forge: ['heat'],
  resonance: ['resonance'],
  eccentricDecree: ['decree_focus'],
  alchemical: ['reaction'],
};

const STATE_TRACKED_BUFFS = new Set([
  'completion_bonus',
  'inscription',
  'captivating_cadence',
  'captivating_cadence_chain',
]);

const CRIT_STAT_KEYS = [
  'critchance',
  'critmultiplier',
  'overcrit',
  'overcritCapBonus',
  'refinementCritChance',
];

const ACTION_EFFECTS: Readonly<Record<string, string>> = {
  fusion: 'onFusion',
  refine: 'onRefine',
  stabilize: 'onStabilize',
  support: 'onSupport',
};

function combineCostStat(
  reduction: boolean,
  current: number,
  contribution: number,
): number {
  const value = reduction
    ? 100 - ((100 - Math.min(100, current)) * (100 - Math.min(100, contribution))) / 100
    : ((100 + current) * (100 + contribution)) / 100 - 100;
  return Math.round(value * 1e10) / 1e10;
}

function clampForgeHeat(value: number): number {
  return Math.max(0, Math.min(10, Math.floor(value)));
}

function forgeHeatBand(heat: number): string {
  if (heat >= 4 && heat <= 6) return 'optimal';
  if (heat >= 2 && heat <= 3) return 'controlPenalty';
  if (heat >= 7 && heat <= 9) return 'intensityPenalty';
  if (heat === 0) return 'controlCollapse';
  if (heat === 10) return 'intensityCollapse';
  return 'neutral';
}

function forgeWorksModifiers(heat: number): typeof DEFAULT_HARMONY_MODS {
  const mods = { ...DEFAULT_HARMONY_MODS };
  switch (forgeHeatBand(heat)) {
    case 'optimal':
      mods.controlMultiplier = 1.5;
      mods.intensityMultiplier = 1.5;
      break;
    case 'controlPenalty':
      mods.controlMultiplier = 0.5;
      break;
    case 'intensityPenalty':
      mods.intensityMultiplier = 0.5;
      break;
    case 'controlCollapse':
      mods.controlMultiplier = -9;
      break;
    case 'intensityCollapse':
      mods.intensityMultiplier = -9;
      break;
  }
  return mods;
}

function getHarmonyMods(data: MirrorData): typeof DEFAULT_HARMONY_MODS {
  const hd = data.state.harmonyData;
  const type = data.state.harmonyType;
  if (!hd || !type) return { ...DEFAULT_HARMONY_MODS };
  switch (type) {
    case 'forge': {
      const fw = hd.forgeWorks;
      const heat = clampForgeHeat(num(fw?.heat, 0));
      const effective = heat !== 1 ? heat : clampForgeHeat(num(fw?.lastBuffedHeat, heat));
      return forgeWorksModifiers(effective);
    }
    case 'alchemical': {
      const raw = obj(hd.additionalData?.alchemicalReactionModifiers);
      if (!raw) return { ...DEFAULT_HARMONY_MODS };
      const mods: Record<string, number> = { ...DEFAULT_HARMONY_MODS };
      for (const [key, value] of Object.entries(raw)) {
        if (typeof value === 'number' && Number.isFinite(value)) {
          mods[key] = value;
        } else if (key === 'poolCostPercentage' && typeof value === 'number') {
          mods.qiCostReduction = 100 - value;
        } else if (key === 'stabilityCostPercentage' && typeof value === 'number') {
          mods.stabilityCostReduction = 100 - value;
        }
      }
      if (mods.qiCostReduction === undefined && typeof raw.poolCostPercentage === 'number') {
        mods.qiCostReduction = 100 - num(raw.poolCostPercentage);
      }
      if (
        mods.stabilityCostReduction === undefined &&
        typeof raw.stabilityCostPercentage === 'number'
      ) {
        mods.stabilityCostReduction = 100 - num(raw.stabilityCostPercentage);
      }
      return mods as typeof DEFAULT_HARMONY_MODS;
    }
    case 'inscription': {
      const bonus = num(hd.inscribedPatterns?.stacks, 0) * 0.02;
      return {
        ...DEFAULT_HARMONY_MODS,
        controlMultiplier: 1 + bonus,
        intensityMultiplier: 1 + bonus,
      };
    }
    case 'resonance': {
      const strength = num(hd.resonance?.strength, 0);
      return {
        ...DEFAULT_HARMONY_MODS,
        critChanceBonus: strength * 3,
        successChanceBonus: strength * 0.03,
      };
    }
    case 'eccentricDecree': {
      const focusedBar = hd.eccentricDecree?.focusedBar ?? 'completion';
      return {
        ...DEFAULT_HARMONY_MODS,
        intensityMultiplier: focusedBar === 'completion' ? 1.5 : 1,
        controlMultiplier: focusedBar === 'perfection' ? 1.5 : 1,
      };
    }
    case 'captivatingCadence': {
      const bonus = num(hd.captivatingCadence?.chain, 0) * 0.02;
      return {
        ...DEFAULT_HARMONY_MODS,
        controlMultiplier: 1 + bonus,
        intensityMultiplier: 1 + bonus,
      };
    }
    default:
      return { ...DEFAULT_HARMONY_MODS };
  }
}

/** Name-based classification only: independent of the current state's
 *  harmonyData. The catalog is built once, so the active-subsystem gate has to
 *  be evaluated per node by the shader (see `harmonyDerivedMask` in NS_STATE). */
function isDerivedHarmonyCandidate(data: MirrorData, buff: MirrorBuff): boolean {
  const type = data.state.harmonyType;
  if (!type) return false;
  const names = DERIVED_HARMONY_BUFF_NAMES[type] ?? [];
  return [buff.key, buff.name].some((candidate) => {
    const normalized = normalizeBuffName(candidate);
    return names.includes(normalized) || (type === 'forge' && normalized.startsWith('heat_('));
  });
}

function isDerivedHarmonyBuff(data: MirrorData, buff: MirrorBuff): boolean {
  const hd = data.state.harmonyData;
  if (!hd || !isDerivedHarmonyCandidate(data, buff)) return false;
  const type = data.state.harmonyType;
  return (
    (type === 'forge' && Boolean(hd.forgeWorks)) ||
    (type === 'resonance' && Boolean(hd.resonance)) ||
    (type === 'eccentricDecree' && Boolean(hd.eccentricDecree)) ||
    (type === 'alchemical' && hd.additionalData?.alchemicalReactionModifiers !== undefined)
  );
}

/** Bit of `harmonyDerivedMask` that gates stripping of this subsystem's buffs. */
function derivedHarmonyActiveBit(type: string | undefined): number {
  switch (type) {
    case 'forge': return 1;
    case 'resonance': return 2;
    case 'eccentricDecree': return 4;
    case 'alchemical': return 8;
    default: return 0;
  }
}

/** Dedicated, caller-detectable signal that a fixed-capacity record dropped data. */
function noteOverflow(note: (kind: string, name: string) => void, name: string): void {
  note('overflow', name);
}

/** Count skipped payloads instead of silently treating them as zero-value successes. */
function makeNoteSink(): {
  note: (kind: string, name: string) => void;
  snapshot: () => Record<string, number>;
} {
  const skipped = new Map<string, number>();
  return {
    note(kind, name) {
      const key = `${kind}:${name}`;
      skipped.set(key, (skipped.get(key) ?? 0) + 1);
    },
    snapshot() {
      const out: Record<string, number> = {};
      skipped.forEach((count, key) => {
        out[key] = count;
      });
      return out;
    },
  };
}

function applyUpgrades(value: unknown, upgrades: UpgradeMap): unknown {
  if (Object.keys(upgrades).length === 0) return value;
  const seen = new WeakMap<object, unknown>();
  const walk = (node: unknown): unknown => {
    if (!node || typeof node !== 'object') return node;
    const cached = seen.get(node as object);
    if (cached !== undefined) return cached;
    if (Array.isArray(node)) {
      const result = node.map(walk);
      seen.set(node, result);
      return result;
    }
    const source = node as Raw;
    const clone: Raw = {};
    seen.set(node, clone);
    let changed = false;
    for (const [key, child] of Object.entries(source)) {
      const upgraded = walk(child);
      clone[key] = upgraded;
      if (upgraded !== child) changed = true;
    }
    const upgradeKey = str(source.upgradeKey).trim();
    const rule = upgradeKey ? upgrades[upgradeKey] : undefined;
    if (rule) {
      for (const field of ['amount', 'value', 'cooldown']) {
        const child = clone[field];
        if (typeof child !== 'number' || !Number.isFinite(child)) continue;
        const upgraded = (child + rule.additive) * rule.multiplier;
        if (upgraded !== child) {
          clone[field] = upgraded;
          changed = true;
        }
      }
    }
    return changed ? clone : node;
  };
  return walk(value);
}

function nativeVar(key: string, data: MirrorData): number | undefined {
  const native = data.state.nativeVariables;
  if (!native || !key) return undefined;
  if (key in native) return native[key];
  const lower = key.toLowerCase();
  if (lower in native) return native[lower];
  const normalized = lower.trim().replace(/\s+/g, '_');
  return normalized in native ? native[normalized] : undefined;
}

function scalingVar(key: unknown, ctx: EvalCtx, vars: Vars): number {
  const name = str(key);
  switch (name) {
    case 'flat':
      return 1;
    case 'control':
      return ctx.frame.buffed.control * ctx.frame.controlMult;
    case 'intensity':
      return ctx.frame.buffed.intensity * ctx.frame.intensityMult;
    case 'stacks':
      return vars.stacks ?? 0;
    case 'maxpool':
      return ctx.data.state.maxPool;
    case 'maxstability':
      return ctx.data.state.maxStability;
    case 'completion':
      return ctx.data.state.completion;
    case 'perfection':
      return ctx.data.state.perfection;
    case 'maxcompletion':
      return ctx.data.targetCompletion;
    case 'maxperfection':
      return ctx.data.targetPerfection;
    case 'critchance':
      return ctx.frame.buffed.critchance;
    case 'critmultiplier':
      return ctx.frame.buffed.critmultiplier;
    default: {
      const seeded = nativeVar(name, ctx.data);
      return seeded !== undefined ? seeded : 0;
    }
  }
}

const EQN_FUNCTIONS = ['floor', 'ceil', 'round', 'min', 'max', 'abs'] as const;
const EQN_RESERVED = new Set([
  'true', 'false', 'null', 'undefined', 'if', 'else', 'return', 'const', 'let',
  'var', 'new', 'this', 'typeof', 'instanceof', 'in', 'void', 'delete',
  'function', 'while', 'for', 'do', 'switch', 'case', 'break', 'continue',
  'default', 'try', 'catch', 'finally', 'throw', 'class', 'extends', 'super',
  'import', 'export', 'await', 'yield', 'with', 'Math',
]);
const EQN_ALLOWED_CHARS = /^[\w\s+\-*/%().,<>=!&|:{}?]+$/;
const EQN_BLOCKED_KEYWORDS =
  /\b(?:while|for|do|switch|try|catch|finally|class|function|new|return|throw|import|export|await|yield|with|const|let|var|delete|this|super)\b/i;
const EQN_ASSIGNMENT = /(^|[^=!<>])=($|[^=])/;

/**
 * Full variable frame handed to a game eqn. The engine's `evalExpression`
 * resolves unknown identifiers to 0 through a Proxy, so an expression that
 * mentions `pool`/`maxpool` (absent from the old hand-picked scope) used to
 * throw and emit an `eqn:` note instead of evaluating.
 */
function expressionVariables(ctx: EvalCtx, vars: Vars): Vars {
  const s = ctx.data.state;
  return {
    ...(s.nativeVariables ?? {}),
    ...vars,
    control: ctx.frame.buffed.control * ctx.frame.controlMult,
    intensity: ctx.frame.buffed.intensity * ctx.frame.intensityMult,
    stacks: vars.stacks ?? 0,
    maxpool: s.maxPool,
    maxstability: s.maxStability,
    completion: s.completion,
    perfection: s.perfection,
    maxcompletion: ctx.data.targetCompletion,
    maxperfection: ctx.data.targetPerfection,
    critchance: ctx.frame.buffed.critchance,
    critmultiplier: ctx.frame.buffed.critmultiplier,
    pool: s.pool,
    stability: s.stability,
    overcrit: s.overcrit,
    overcritCapBonus: s.overcritCapBonus,
    poolCostFlat: s.poolCostFlat,
    successChanceBonus: s.successChanceBonus,
  };
}

/**
 * Numeric mirror of gameTypes.ts `evalExpression`. All identifiers are
 * declared as function parameters with a 0 fallback, so a missing symbol can
 * never raise a ReferenceError; blocked/unparseable expressions keep a
 * specific note instead of being silently treated as a zero-value success.
 */
function evalEqn(
  eqn: unknown,
  scope: Vars,
  note: (kind: string, name: string) => void,
): number {
  if (typeof eqn !== 'string' || eqn.trim() === '') return 0;
  const normalized = eqn
    .replace(/\band\b/g, '&&')
    .replace(/\bor\b/g, '||')
    .replace(/\{rng\}/g, '0.5');
  if (
    normalized.length > 1024 ||
    !EQN_ALLOWED_CHARS.test(normalized) ||
    EQN_BLOCKED_KEYWORDS.test(normalized) ||
    EQN_ASSIGNMENT.test(normalized)
  ) {
    note('eqn', `unsupported:${eqn}`);
    return 0;
  }
  const names: string[] = [];
  const seen = new Set<string>();
  for (const id of normalized.match(/[A-Za-z_$][A-Za-z0-9_$]*/g) ?? []) {
    if (EQN_RESERVED.has(id) || (EQN_FUNCTIONS as readonly string[]).includes(id)) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    names.push(id);
  }
  const values = names.map((name) => {
    const value = scope[name];
    return typeof value === 'number' && Number.isFinite(value) ? value : 0;
  });
  try {
    const fn = new Function(
      ...names,
      ...EQN_FUNCTIONS,
      `"use strict"; return (${normalized});`,
    );
    const result: unknown = fn(
      ...values,
      Math.floor,
      Math.ceil,
      Math.round,
      Math.min,
      Math.max,
      Math.abs,
    );
    if (typeof result === 'boolean') return result ? 1 : 0;
    const numeric = Number(result);
    return Number.isFinite(numeric) ? numeric : 0;
  } catch {
    note('eqn', `invalid:${eqn}`);
    return 0;
  }
}

function evalScaling(
  payload: unknown,
  ctx: EvalCtx,
  vars: Vars = {},
  deferRound = false,
): number {
  if (payload === undefined || payload === null) return 0;
  const upgraded = applyUpgrades(payload, ctx.upgrades);
  if (typeof upgraded === 'number') return Number.isFinite(upgraded) ? upgraded : 0;
  const s = obj(upgraded);
  if (!s) return 0;
  let result = num(s.value, 0);
  if (!Number.isFinite(result)) return 0;
  if (s.stat) result *= scalingVar(s.stat, ctx, vars);
  if (s.scaling) {
    let value = scalingVar(s.scaling, ctx, vars);
    if (s.scalingMax) {
      const cap = evalScaling(s.scalingMax, ctx, vars);
      value = cap < 0 ? Math.max(value, cap) : Math.min(value, cap);
    }
    result *= value;
  }
  if (s.eqn) {
    result *= evalEqn(s.eqn, expressionVariables(ctx, vars), ctx.note);
  }
  if (s.max) {
    const cap = evalScaling(s.max, ctx, vars);
    result = cap < 0 ? Math.max(result, cap) : Math.min(result, cap);
  }
  return deferRound ? result : roundScaling(result);
}

function isStackOnlyCritScaling(payload: unknown): boolean {
  const s = obj(payload);
  if (!s) return true;
  return Object.keys(s).every((key) => {
    if (key === 'value' || key === 'upgradeKey') return true;
    if (key === 'scaling') return s.scaling === 'stacks';
    if (key === 'max' || key === 'scalingMax') {
      const inner = obj(s[key]);
      return Boolean(inner) && Object.keys(inner ?? {}).every(
        (innerKey) => innerKey === 'value' || innerKey === 'upgradeKey',
      );
    }
    return false;
  });
}

function expectedCritMultiplier(
  chance: number,
  multiplier: number,
  overcrit = 0,
  overcritCapBonus = 0,
  chainGrowth = 0,
): number {
  if (overcrit === 0) {
    const c = Math.max(0, Math.min(chance, 100)) / 100;
    const m = (multiplier + Math.max(0, chance - 100) * 3) / 100;
    return 1 - c + c * m;
  }
  const effective =
    multiplier + Math.max(0, chance - 100) * 3 + Math.max(0, overcrit - 90) * 5;
  const c = Math.max(0, Math.min(chance, 100)) / 100;
  let chain = 1;
  const cap = Math.min(overcrit, Math.min(99, 90 + Math.max(0, overcritCapBonus)));
  let threshold = Math.max(0, Math.min(chance, 100));
  let reach = 1;
  for (let step = 0; step < 9; step++) {
    threshold = Math.floor((threshold * cap) / 100);
    if (threshold <= 0) break;
    reach *= Math.min(threshold, 100) / 100;
    chain += reach * (1 + chainGrowth / 100) ** step;
  }
  return 1 - c + c * (effective / 100) * chain;
}

function buildBuffCritFactor(
  data: MirrorData,
  state: MirrorState,
  ctx: CritContext,
  note: (kind: string, name: string) => void,
): number {
  let chance = state.critChance;
  let multiplier = state.critMultiplier;
  let overcrit = state.overcrit;
  let overcritCapBonus = state.overcritCapBonus;
  let growth = 0;
  let localOnly = true;
  const base = data.state;
  const evalCtx: EvalCtx = {
    data,
    upgrades: ctx.upgrades ?? {},
    frame: {
      buffed: {
        control: base.control,
        intensity: base.intensity,
        critchance: chance,
        critmultiplier: multiplier,
      },
      controlMult: 1,
      intensityMult: 1,
    },
    note,
  };
  const fold = (stats: unknown, stacks: number): void => {
    const table = obj(stats);
    if (!table) return;
    for (const key of CRIT_STAT_KEYS) {
      const scaling = table[key];
      if (!scaling || typeof scaling !== 'object') continue;
      if (!isStackOnlyCritScaling(scaling)) {
        localOnly = false;
        continue;
      }
      // evalCtx already carries ctx.upgrades; evalScaling applies them once.
      // A second explicit applyUpgrades here double-folds the mastery rule.
      const value = evalScaling(scaling, evalCtx, { stacks });
      if (key === 'critchance') chance += value;
      else if (key === 'critmultiplier') multiplier += value;
      else if (key === 'overcrit') overcrit += value;
      else if (key === 'overcritCapBonus') overcritCapBonus += value;
    }
  };
  for (const buff of data.buffs) {
    if (isDerivedHarmonyBuff(data, buff)) continue;
    const name = normalizeBuffName(buff.name || buff.key);
    const stacks = ctx.liveStacks?.[name] ?? buff.stacks ?? 1;
    growth += stacks > 0 ? num(obj(buff.definition)?.overcritChainGrowth, 0) : 0;
    fold(buff.stats, stacks);
  }
  for (const buff of ctx.masteryExtraBuffs ?? []) {
    const name = normalizeBuffName(buff.name ?? '');
    const stacks = ctx.liveStacks?.[name] ?? buff.stacks ?? 1;
    fold(buff.stats, stacks);
  }
  chance += getHarmonyMods(data).critChanceBonus;
  return localOnly
    ? expectedCritMultiplier(chance, multiplier, overcrit, overcritCapBonus, growth)
    : (ctx.frame?.critFactor ?? 1);
}

function conditionProbability(
  condition: unknown,
  effect: unknown,
  buff: MirrorBuff | undefined,
  data: MirrorData,
  note: (kind: string, name: string) => void,
): number {
  const c = obj(condition);
  if (!c) return 1;
  if (c.kind === 'chance') return clamp01(num(c.percentage, 0) / 100);
  if (c.kind === 'buff') {
    let count: number;
    if (c.buff === 'self') {
      count = buff ? (buff.stacks ?? 1) : 0;
    } else {
      const holder = obj(c.buff);
      const definition = obj(holder?.definition);
      const effectHolder = obj(effect);
      const effectBuff = obj(effectHolder?.buff);
      const effectDefinition = obj(effectBuff?.definition);
      const name = normalizeBuffName(
        definition?.name ?? holder?.name ?? effectDefinition?.name ?? effectBuff?.name,
      );
      const held = data.buffs.find(
        (candidate) =>
          normalizeBuffName(candidate.name) === name ||
          normalizeBuffName(candidate.key) === name,
      );
      count = held ? (held.stacks ?? 0) : 0;
    }
    const target = num(c.count, 0);
    if (c.mode === 'more') return count >= target ? 1 : 0;
    if (c.mode === 'less') return count < target ? 1 : 0;
    return count === target ? 1 : 0;
  }
  // Engine: evaluateEffectCondition -> evaluateScaling({ value: 1, eqn },
  // { ...variables, stacks: selfStacks }) > 0. Deterministic in simulation.
  if (c.kind === 'condition') {
    const evalCtx: EvalCtx = {
      data,
      upgrades: {},
      frame: {
        buffed: {
          control: data.state.control,
          intensity: data.state.intensity,
          critchance: data.state.critChance,
          critmultiplier: data.state.critMultiplier,
        },
        controlMult: 1,
        intensityMult: 1,
      },
      note,
    };
    const selfStacks = buff ? (buff.stacks ?? 1) : 0;
    const value = evalEqn(
      str(c.condition),
      expressionVariables(evalCtx, { stacks: selfStacks }),
      note,
    );
    return value > 0 ? 1 : 0;
  }
  note('condition', str(c.kind || '(none)'));
  return 0;
}

function buildFrame(
  data: MirrorData,
  state: MirrorState,
  extraBuffs: ExtraBuff[] = [],
  heldStacks: Record<string, number> = {},
  upgrades: UpgradeMap = {},
  note: (kind: string, name: string) => void = () => undefined,
): Frame {
  const boosts = {
    completion: 0,
    perfection: state.completionBonus * 10,
    stability: 0,
    qi: 0,
  };
  // skills.ts:1924-1940 — a turn-counter control/intensity buff multiplies the
  // base stat before any buff stat is folded, unless a live buff carries the
  // stat explicitly. The shader applies the same rule (kernelState.ts
  // liveHasStatBuff); without it here the carry stand compares against a frame
  // the engine never uses.
  const liveBuffList = data.buffs.filter(
    (buff) =>
      !STATE_TRACKED_BUFFS.has(normalizeBuffName(buff.name || buff.key)) &&
      !isDerivedHarmonyBuff(data, buff),
  );
  const hasExplicitControlBuff = liveBuffList.some(
    (buff) => obj(buff.stats)?.control !== undefined,
  );
  const hasExplicitIntensityBuff = liveBuffList.some(
    (buff) => obj(buff.stats)?.intensity !== undefined,
  );
  const controlTurnsMult =
    state.controlBuffTurns > 0 && !hasExplicitControlBuff
      ? state.controlBuffMultiplier
      : 1;
  const intensityTurnsMult =
    state.intensityBuffTurns > 0 && !hasExplicitIntensityBuff
      ? state.intensityBuffMultiplier
      : 1;
  const buffed = {
    control: state.control * controlTurnsMult,
    intensity: state.intensity * intensityTurnsMult,
    critchance: state.critChance,
    critmultiplier: state.critMultiplier,
    poolCostPercentage: 100,
    stabilityCostPercentage: 100,
  };
  const cost = {
    qiReduction: state.qiCostReduction,
    qiPenalty: state.qiCostPenalty,
    stabilityReduction: state.stabilityCostReduction,
    stabilityPenalty: state.stabilityCostPenalty,
    poolCostFlat: state.poolCostFlat,
  };
  let hasPoolPctBuff = false;
  let hasStabPctBuff = false;
  // Buff stats must be re-evaluated against the state this frame is built for,
  // including when a caller rebuilds it for a perturbed node (buildNodes).
  const evalCtx: EvalCtx = {
    data,
    upgrades,
    frame: {
      buffed: {
        control: state.control,
        intensity: state.intensity,
        critchance: state.critChance,
        critmultiplier: state.critMultiplier,
      },
      controlMult: 1,
      intensityMult: 1,
    },
    note,
  };
  const allBuffs: ExtraBuff[] = [
    ...liveBuffList.map((buff) => ({
      name: buff.name,
      stacks: heldStacks[normalizeBuffName(buff.name || buff.key)] ?? buff.stacks,
      stats: buff.stats,
    })),
    ...extraBuffs,
  ];
  for (const buff of allBuffs) {
    const stacks = buff.stacks ?? 1;
    for (const [key, rawScaling] of Object.entries(buff.stats ?? {})) {
      if (!rawScaling || typeof rawScaling !== 'object') continue;
      // evalScaling applies upgrades through evalCtx; applying them here too
      // would fold every mastery rule twice. mirror.mjs builds this once.
      const raw = evalScaling(rawScaling, evalCtx, { stacks });
      if (!Number.isFinite(raw)) continue;
      if (key === 'completionBoost') {
        boosts.completion += raw;
        continue;
      }
      if (key === 'perfectionBoost') {
        boosts.perfection += raw;
        continue;
      }
      if (key === 'stabilityBoost') {
        boosts.stability += raw;
        continue;
      }
      if (key === 'qiBoost') {
        boosts.qi += raw;
        continue;
      }
      if (key === 'successChanceBonus') {
        // The shader folds this stat: node slot 27 (own bonus) plus created-buff
        // fmods with field 11. Consume it here so it is not reported as skipped.
        continue;
      }
      if (key === 'poolCostPercentage' || key === 'stabilityCostPercentage') {
        buffed[key] = Math.floor((buffed[key] / 100) * (raw / 100) * 100);
        if (key === 'poolCostPercentage') hasPoolPctBuff = true;
        else hasStabPctBuff = true;
        continue;
      }
      if (key === 'qiCostReduction') {
        cost.qiReduction = combineCostStat(true, cost.qiReduction, raw);
        continue;
      }
      if (key === 'qiCostPenalty') {
        cost.qiPenalty = combineCostStat(false, cost.qiPenalty, raw);
        continue;
      }
      if (key === 'stabilityCostReduction') {
        cost.stabilityReduction = combineCostStat(true, cost.stabilityReduction, raw);
        continue;
      }
      if (key === 'stabilityCostPenalty') {
        cost.stabilityPenalty = combineCostStat(false, cost.stabilityPenalty, raw);
        continue;
      }
      if (key === 'poolCostFlat') {
        cost.poolCostFlat += raw;
        continue;
      }
      if (key === 'control' || key === 'intensity') {
        buffed[key] += floorToDouble(raw);
        continue;
      }
      if (key === 'critchance' || key === 'critmultiplier') {
        buffed[key] += raw;
        continue;
      }
      if (!key.endsWith('CostReduction') && !key.endsWith('CostPenalty')) {
        note('buffStat', key);
      }
    }
  }

  const harmony = getHarmonyMods(data);
  if (harmony.controlMultiplier !== 1) {
    buffed.control += (harmony.controlMultiplier - 1) * state.control;
  }
  if (harmony.intensityMultiplier !== 1) {
    buffed.intensity += (harmony.intensityMultiplier - 1) * state.intensity;
  }
  if (harmony.critChanceBonus) buffed.critchance += harmony.critChanceBonus;
  if (harmony.qiCostReduction) {
    cost.qiReduction = combineCostStat(true, cost.qiReduction, harmony.qiCostReduction);
  }
  if (harmony.stabilityCostReduction) {
    cost.stabilityReduction = combineCostStat(
      true,
      cost.stabilityReduction,
      harmony.stabilityCostReduction,
    );
  }

  const conditionStat = (stat: string): number => {
    let multiplier = 1;
    for (const effect of data.conditionEffects) {
      if (effect?.kind === stat && typeof effect.multiplier === 'number') {
        multiplier += effect.multiplier;
      }
    }
    return multiplier;
  };
  const p = clamp01(buffed.critchance / 100);
  const legacyPct = (value: number, raw: number, present: boolean): number =>
    present ? Math.floor(((value || 100) / 100) * 100) / Math.max(1e-10, raw) : 1;
  return {
    buffed,
    boosts,
    controlMult: conditionStat('control'),
    intensityMult: conditionStat('intensity'),
    critFactor: 1 - p + p * (buffed.critmultiplier / 100),
    poolCostFlat: Math.max(0, Math.floor(cost.poolCostFlat)),
    qiFactor: (Math.max(0, 100 - cost.qiReduction) * Math.max(0, 100 + cost.qiPenalty)) / 1e4,
    stabFactor:
      (Math.max(0, 100 - cost.stabilityReduction) * Math.max(0, 100 + cost.stabilityPenalty)) /
      1e4,
    poolPct: legacyPct(
      buffed.poolCostPercentage,
      state.poolCostPercentage ?? 100,
      hasPoolPctBuff,
    ),
    stabPct: legacyPct(
      buffed.stabilityCostPercentage,
      state.stabilityCostPercentage ?? 100,
      hasStabPctBuff,
    ),
  };
}

function branchRecord(frame: Frame, probability: number, baseFrame: Frame): PaymentBranch {
  const ratio = (a: number, b: number): number => (Math.abs(b) > 1e-12 ? a / b : 1);
  return {
    probability,
    qiRatio: ratio(frame.qiFactor, baseFrame.qiFactor),
    stabRatio: ratio(frame.stabFactor, baseFrame.stabFactor),
    poolRatio: ratio(frame.poolPct, baseFrame.poolPct),
    stabPctRatio: ratio(frame.stabPct, baseFrame.stabPct),
    extraFlat: frame.poolCostFlat - baseFrame.poolCostFlat,
  };
}

function expectedPaymentBranches(
  data: MirrorData,
  skill: Raw,
  baseFrame: Frame,
  note: (kind: string, name: string) => void,
): PaymentBranch[] {
  const chanceOf = (condition: unknown): number => {
    const c = obj(condition);
    return c?.kind === 'chance' ? clamp01(num(c.percentage, 0) / 100) : 1;
  };
  const entries = list(skill.masteryEntries).filter((entry) => {
    const e = obj(entry);
    return e?.kind === 'effect' && Array.isArray(e.effects);
  });
  const heldNames = new Set(data.buffs.map((buff) => normalizeBuffName(buff.name || buff.key)));
  const creations: Array<{ stats: Record<string, Scaling>; stacks: number; probability: number }> = [];
  let varies = false;
  for (const entry of entries) {
    const e = obj(entry);
    if (!e) continue;
    const entryProb = chanceOf(e.condition);
    if (entryProb > 0 && entryProb < 1) varies = true;
    for (const effect of list(e.effects)) {
      const effectRec = obj(effect);
      if (effectRec?.kind !== 'createBuff') continue;
      const buffRecord = obj(effectRec.buff) ?? {};
      const definition = obj(buffRecord.definition) ?? buffRecord;
      const stats = (definition.stats ?? {}) as Record<string, Scaling>;
      if (
        !Object.keys(stats).some(
          (key) => key === 'poolCostPercentage' || key === 'stabilityCostPercentage',
        )
      ) {
        continue;
      }
      const effectProb = chanceOf(effectRec.condition);
      if (effectProb > 0 && effectProb < 1) varies = true;
      const name = normalizeBuffName(definition.name);
      if (heldNames.has(name)) continue;
      heldNames.add(name);
      creations.push({
        stats,
        stacks: Math.floor(num(obj(effectRec.stacks)?.value, 1)),
        probability: entryProb * effectProb,
      });
    }
  }
  if (!varies) {
    return [
      branchRecord(
        buildFrame(
          data,
          data.state,
          creations.map((creation) => ({ stacks: creation.stacks, stats: creation.stats })),
          {},
          {},
          note,
        ),
        1,
        baseFrame,
      ),
    ];
  }

  let branches: Array<{ probability: number; buffs: ExtraBuff[] }> = [
    { probability: 1, buffs: [] },
  ];
  for (const creation of creations) {
    const next: Array<{ probability: number; buffs: ExtraBuff[] }> = [];
    for (const branch of branches) {
      if (creation.probability < 1) {
        next.push({
          probability: branch.probability * (1 - creation.probability),
          buffs: branch.buffs,
        });
      }
      if (creation.probability > 0) {
        next.push({
          probability: branch.probability * creation.probability,
          buffs: [
            ...branch.buffs,
            { stacks: creation.stacks, stats: creation.stats },
          ],
        });
      }
    }
    branches = next;
  }
  const frames = branches.map((branch) =>
    buildFrame(data, data.state, branch.buffs, {}, {}, note),
  );
  const costs = frames.map((frame) => ({
    qi: Math.max(
      0,
      floorTo(f(f((num(skill.qiCost, 0) + frame.poolCostFlat)) * frame.qiFactor) * frame.poolPct),
    ),
    stab: floorTo(f(f(num(skill.stabilityCost, 0) * frame.stabFactor) * frame.stabPct)),
  }));
  if (
    costs.length > 0 &&
    costs.every((cost) => cost.qi === costs[0].qi && cost.stab === costs[0].stab)
  ) {
    return [branchRecord(frames[0], 1, baseFrame)];
  }
  return branches.map((branch, index) =>
    branchRecord(frames[index], branch.probability, baseFrame),
  );
}

function opOf(kind: unknown): number {
  switch (kind) {
    case 'perfection':
      return OPS.perfection;
    case 'completion':
      return OPS.completion;
    case 'stability':
      return OPS.stability;
    case 'maxStability':
      return OPS.maxStability;
    case 'pool':
      return OPS.pool;
    case 'createBuff':
      return OPS.createBuff;
    default:
      return 0;
  }
}

function buildSkills(data: MirrorData, count: number, note: (kind: string, name: string) => void): Float32Array {
  const records = new Float32Array(count * REC);
  const baseFrame = buildFrame(data, data.state, [], {}, {}, note);
  const ctxForValues: EvalCtx = {
    data,
    upgrades: {},
    frame: {
      buffed: {
        control: 1,
        intensity: 1,
        critchance: data.state.critChance,
        critmultiplier: data.state.critMultiplier,
      },
      controlMult: 1,
      intensityMult: 1,
    },
    note,
  };
  const ctxForMods: EvalCtx = {
    data,
    upgrades: {},
    frame: {
      buffed: {
        control: data.state.control,
        intensity: data.state.intensity,
        critchance: data.state.critChance,
        critmultiplier: data.state.critMultiplier,
      },
      controlMult: 1,
      intensityMult: 1,
    },
    note,
  };

  for (let a = 0; a < count; a++) {
    const skill = obj(data.skills[a % data.skills.length]) ?? {};
    const b = a * REC;
    const effects = list(skill.effects);

    const upgrades: UpgradeMap = {};
    let controlBonus = 0;
    let intensityBonus = 0;
    let critChanceBonus = 0;
    let critMultiplierBonus = 0;
    const effectEntries: Array<{ factor: number; resolved: Array<{ effect: Raw; chance: number }> }> = [];
    for (const entry of list(skill.masteryEntries)) {
      const e = obj(entry);
      const kind = str(e?.kind || '(none)');
      if (!e || kind === '(none)') {
        note('mastery', kind);
        continue;
      }
      const factor = conditionProbability(e.condition, e, undefined, data, note);
      switch (kind) {
        case 'effect': {
          if (factor <= 0) break;
          const resolved: Array<{ effect: Raw; chance: number }> = [];
          for (const effect of list(e.effects)) {
            const effectRec = obj(effect);
            if (!effectRec) continue;
            const chance = conditionProbability(effectRec.condition, effectRec, undefined, data, note);
            resolved.push({ effect: effectRec, chance: chance * factor });
          }
          effectEntries.push({ factor, resolved });
          break;
        }
        case 'upgrade': {
          const key = str(e.upgradeKey).trim();
          const change = num(e.change, 0);
          if (!key || change === 0) break;
          const rule = upgrades[key] ?? { additive: 0, multiplier: 1 };
          if (e.shouldMultiply) rule.multiplier *= 1 + change;
          else rule.additive += change;
          upgrades[key] = rule;
          break;
        }
        case 'control':
          controlBonus += (num(e.percentage, 0) / 100) * factor;
          break;
        case 'intensity':
          intensityBonus += (num(e.percentage, 0) / 100) * factor;
          break;
        case 'critchance':
          critChanceBonus += num(e.percentage, 0) * factor;
          break;
        case 'critmultiplier':
          critMultiplierBonus += num(e.percentage, 0) * factor;
          break;
        default:
          note('mastery', kind);
      }
    }
    ctxForValues.upgrades = upgrades;
    ctxForMods.upgrades = upgrades;

    const heldBuffs = new Map(
      data.buffs.map((buff) => [normalizeBuffName(buff.name || buff.key), buff] as const),
    );
    const heldStacksForCrit: Record<string, number> = {};
    const masteryExtraBuffs: ExtraBuff[] = [];
    for (const { resolved } of effectEntries) {
      for (const { effect, chance } of resolved) {
        if (effect.kind !== 'createBuff') continue;
        const stacksScaling = applyUpgrades(effect.stacks ?? {}, upgrades);
        // `chance` already carries the entry factor; multiplying by `factor`
        // again would square it (mirror.mjs uses value * factor * effectChance).
        const stacks = Math.floor(num(obj(stacksScaling)?.value, 1) * chance);
        if (!Number.isFinite(stacks) || stacks <= 0) continue;
        const buffRecord = obj(effect.buff) ?? {};
        const definition = obj(buffRecord.definition) ?? buffRecord;
        const name = normalizeBuffName(definition.name);
        if (heldBuffs.has(name)) {
          const held = heldBuffs.get(name);
          if (!held) continue;
          const current = heldStacksForCrit[name] ?? held.stacks ?? 1;
          const maxStacks = num(definition.maxStacks ?? obj(held.definition)?.maxStacks, 0);
          heldStacksForCrit[name] = maxStacks > 0 ? Math.min(current + stacks, maxStacks) : current + stacks;
          continue;
        }
        const existing = masteryExtraBuffs.find(
          (buff) => normalizeBuffName(buff.name ?? '') === name,
        );
        if (existing) {
          existing.stacks += stacks;
          continue;
        }
        masteryExtraBuffs.push({
          name: str(definition.name),
          stacks,
          stats: (definition.stats ?? {}) as Record<string, Scaling>,
        });
      }
    }

    let effectCount = 0;
    const add = (
      op: number,
      value: number,
      stat: unknown,
      chance: number,
      critMode = 0,
      zeroFlag = 0,
      critRatio = 1,
    ): void => {
      if (effectCount >= MAX_EFFECTS || !op) {
        note('effectsFull', str(skill.name));
        if (effectCount >= MAX_EFFECTS) noteOverflow(note, str(skill.name));
        return;
      }
      const e = b + EFF_OFF + effectCount * 7;
      records[e] = op;
      records[e + 1] = Number.isFinite(value) ? value : 0;
      records[e + 2] = STATS[str(stat)] ?? 0;
      records[e + 3] = chance;
      records[e + 4] = critMode;
      records[e + 5] = zeroFlag;
      records[e + 6] = critRatio;
      effectCount++;
    };
    const zeroFlagOf = (effect: Raw): number => (num(obj(effect.amount)?.value, 0) > 0 ? 1 : 0);
    const statOf = (amount: unknown): unknown => obj(amount)?.stat;
    const pure = (amount: unknown): unknown => {
      if (amount === undefined) return { value: 0 };
      if (typeof amount === 'number') return amount;
      const source = obj(amount);
      if (!source) return { value: 0 };
      const { stat: _stat, ...rest } = source;
      void _stat;
      return rest;
    };
    const amountOf = (effect: Raw): unknown =>
      effect.amount !== undefined ? effect.amount : { value: num(effect.value, 0) };

    records[b + 0] = num(skill.qiCost, 0);
    records[b + 1] = num(skill.stabilityCost, 0);
    records[b + 2] = num(skill.stabilityGain, 0);
    records[b + 3] = num(skill.maxStabilityChange, 0);
    records[b + 4] = skill.preventsMaxStabilityDecay ? 1 : 0;
    const hasPoolEffect = effects.some((effect) => obj(effect)?.kind === 'pool');
    records[b + 5] = !hasPoolEffect && skill.restoresQi ? num(skill.qiRestore, 0) : 0;
    records[b + 6] = effects.length > 0 ? 1 : 0;
    records[b + 12] = 1 + controlBonus;
    records[b + 13] = 1 + intensityBonus;
    records[b + 14] = critChanceBonus;
    records[b + 15] = critMultiplierBonus;
    // Base skill roll only. The craft's own bonus is written to node slot 27
    // (NODE_SUCCESS_BONUS) and buff-created bonuses arrive as fmod field 11;
    // the shader clamps(base + both) when rolling a technique (skills.ts:4497).
    records[b + SUCCESS_WEIGHT] = clamp01(num(skill.successChance, 1));

    const branches = expectedPaymentBranches(data, skill, baseFrame, note);
    records[b + 9] = Math.min(branches.length, MAX_COST_BRANCHES);
    if (branches.length > MAX_COST_BRANCHES) {
      note('costBranchesFull', str(skill.name));
      noteOverflow(note, str(skill.name));
    }
    for (let c = 0; c < records[b + 9]; c++) {
      const cb = b + COST_OFF + c * 6;
      records[cb] = branches[c].probability;
      records[cb + 1] = branches[c].qiRatio;
      records[cb + 2] = branches[c].stabRatio;
      records[cb + 3] = branches[c].poolRatio;
      records[cb + 4] = branches[c].stabPctRatio;
      records[cb + 5] = branches[c].extraFlat;
    }

    if (skill.isDisciplinedTouch) {
      add(OPS.completion, num(skill.baseCompletionGain, 0), 'intensity', 1);
      add(OPS.perfection, num(skill.basePerfectionGain, 0), 'intensity', 1);
    } else {
      for (const effect of effects) {
        const effectRec = obj(effect);
        if (!effectRec) continue;
        const op = opOf(effectRec.kind);
        if (!op) {
          note('effect', str(effectRec.kind || '(none)'));
          continue;
        }
        const chance = conditionProbability(effectRec.condition, effectRec, undefined, data, note);
        if (chance <= 0) continue;
        const amount = applyUpgrades(amountOf(effectRec), upgrades);
        add(
          op,
          evalScaling(pure(amount), ctxForValues, {}, true),
          statOf(amount),
          chance,
          0,
          zeroFlagOf(effectRec),
        );
      }
    }

    const actionKey = ACTION_EFFECTS[str(skill.type)];
    const liveStacks: Record<string, number> = {};
    for (const buff of data.buffs) {
      const name = normalizeBuffName(buff.name || buff.key);
      liveStacks[name] = heldStacksForCrit[name] ?? buff.stacks ?? 1;
    }
    for (const buff of masteryExtraBuffs) {
      const name = normalizeBuffName(buff.name ?? '');
      if (!(name in liveStacks)) liveStacks[name] = buff.stacks ?? 1;
    }
    // mirror.mjs falls back to ctx.frame.critFactor, which by buff-effect time
    // is the POST-mastery frame (phaseMasteryFrame's last makeFrame), not the
    // pre-action baseFrame. Rebuild that frame from the mastery upserts and
    // add the same crit bonuses the kernel folds in via sb+14/sb+15.
    const postMasteryFrame = buildFrame(
      data,
      data.state,
      masteryExtraBuffs,
      heldStacksForCrit,
      upgrades,
      note,
    );
    const postCrit = clamp01(
      (postMasteryFrame.buffed.critchance + critChanceBonus) / 100,
    );
    const postMasteryCrit =
      1 - postCrit + postCrit * ((postMasteryFrame.buffed.critmultiplier + critMultiplierBonus) / 100);
    const critCtx: CritContext = {
      liveStacks,
      masteryExtraBuffs,
      upgrades,
      frame: { critFactor: postMasteryCrit },
    };
    // The denominator is the pre-mastery reference: the kernel's buffCrit is
    // built from the node's base state, so the ratio converts it to post-mastery.
    const initialBuffCrit = buildBuffCritFactor(data, data.state, {
      frame: { critFactor: baseFrame.critFactor },
    }, note);
    for (const buff of data.buffs) {
      if (isDerivedHarmonyBuff(data, buff)) continue;
      const name = normalizeBuffName(buff.name || buff.key);
      const buffEffects = [
        ...(actionKey ? list(obj(buff.definition)?.[actionKey] ?? buff[actionKey as keyof MirrorBuff]) : []),
        ...buff.effects,
      ];
      for (const effect of buffEffects) {
        const effectRec = obj(effect);
        if (!effectRec) continue;
        const chance = conditionProbability(effectRec.condition, effectRec, buff, data, note);
        if (chance <= 0) continue;
        const live = liveStacks[name] ?? buff.stacks ?? 1;
        // A self addStack mutates the live stack count before the rest of the
        // buff phase. Later effects in this same phase read it through the
        // shared `liveStacks` object (crit factor and their own stack overlay).
        // It has no numeric record of its own, so it must precede the op check.
        if (effectRec.kind === 'addStack' && effectRec.buff === undefined) {
          const delta = evalScaling(effectRec.stacks ?? {}, ctxForValues, { stacks: live });
          if (Number.isFinite(delta)) {
            const maxStacks = num(obj(buff.definition)?.maxStacks, 0);
            const next = live + Math.floor(delta * chance);
            liveStacks[name] = Math.max(0, maxStacks > 0 ? Math.min(next, maxStacks) : next);
          }
          continue;
        }
        const op = opOf(effectRec.kind);
        if (!op) {
          note('effect', str(effectRec.kind || '(none)'));
          continue;
        }
        const factor = buildBuffCritFactor(data, data.state, critCtx, note);
        const critRatio = initialBuffCrit !== 0 ? factor / initialBuffCrit : 1;
        const amount = applyUpgrades(amountOf(effectRec), upgrades);
        add(
          op,
          evalScaling(pure(amount), ctxForValues, {}, true),
          statOf(amount),
          chance,
          1,
          zeroFlagOf(effectRec),
          Number.isFinite(critRatio) ? critRatio : 1,
        );
      }
    }

    let mCount = 0;
    let fCount = 0;
    const pushFMod = (field: number, scaling: unknown, stacks: number): void => {
      if (fCount >= MAX_FMODS) {
        note('fmodsFull', str(skill.name));
        noteOverflow(note, str(skill.name));
        return;
      }
      const r = b + FMOD_OFF + fCount * 7;
      const upgraded = applyUpgrades(scaling, upgrades);
      if (typeof upgraded === 'number') {
        records[r] = field;
        records[r + 1] = upgraded;
        records[r + 6] = stacks;
        fCount++;
        return;
      }
      const s = obj(upgraded);
      if (!s) return;
      const fieldStat = STATS[str(s.stat)] ?? 0;
      let scalingStat = 0;
      if (s.scaling) {
        scalingStat = STATS[str(s.scaling)] ?? 0;
        if (!scalingStat) note('scaling', str(s.scaling));
      }
      // The shader sees opcodes only: fold the eqn into the numeric value at
      // record-build time (multiplication commutes with stat/scaling and with
      // the later roundScaling in fmodValue). Expressions that cannot be
      // represented numerically keep a specific eqn note.
      let value = num(s.value, 0);
      if (s.eqn) {
        value *= evalEqn(s.eqn, expressionVariables(ctxForMods, { stacks }), note);
      }
      let scalingMax = s.scalingMax
        ? evalScaling(s.scalingMax, ctxForMods, { stacks })
        : 0;
      if (s.scaling && s.scalingMax && scalingMax === 0) {
        // A zero cap collapses the scaling variable to zero. The shader reads
        // 0 as "no cap", so fold the collapse into the value instead.
        value = 0;
        scalingMax = 0;
      }
      const maxCap = s.max ? evalScaling(s.max, ctxForMods, { stacks }) : 0;
      records[r] = field;
      records[r + 1] = Number.isFinite(value) ? value : 0;
      records[r + 2] = fieldStat;
      records[r + 3] = scalingStat;
      // Signed: negative caps are lower bounds (engine: cap < 0 ? max : min).
      records[r + 4] = Number.isFinite(scalingMax) ? scalingMax : 0;
      records[r + 5] = Number.isFinite(maxCap) ? maxCap : 0;
      records[r + 6] = stacks;
      fCount++;
    };

    for (const { resolved } of effectEntries) {
      const numeric: Array<{
        op: number;
        value: number;
        stat: number;
        chance: number;
        zeroFlag: number;
      }> = [];
      const created: Array<{ stacks: number; mods: Array<{ key: string; scaling: unknown }> }> = [];
      for (const { effect, chance } of resolved) {
        if (effect.kind === 'createBuff') {
          const buffRecord = obj(effect.buff) ?? {};
          const definition = obj(buffRecord.definition) ?? buffRecord;
          const stacksScaling = applyUpgrades(effect.stacks ?? {}, upgrades);
          // `chance` already carries the entry factor; do not square it.
          const stacks = Math.floor(num(obj(stacksScaling)?.value, 1) * chance);
          if (!Number.isFinite(stacks) || stacks <= 0) continue;
          const stats = (definition.stats ?? {}) as Record<string, Scaling>;
          const mods: Array<{ key: string; scaling: unknown }> = [];
          for (const [key, scaling] of Object.entries(stats)) {
            const field = FKEY[key];
            if (!field) {
              if (!key.endsWith('CostReduction') && !key.endsWith('CostPenalty')) {
                note('buffStat', key);
              }
              continue;
            }
            mods.push({ key, scaling });
          }
          created.push({ stacks, mods });
          continue;
        }
        const op = opOf(effect.kind);
        if (!op) {
          note('masteryEffect', str(effect.kind || '(none)'));
          continue;
        }
        if (chance <= 0) continue;
        const amount = applyUpgrades(amountOf(effect), upgrades);
        numeric.push({
          op,
          value: evalScaling(pure(amount), ctxForValues, {}, true),
          stat: STATS[str(statOf(amount))] ?? 0,
          chance,
          zeroFlag: zeroFlagOf(effect),
        });
      }
      for (const n of numeric) {
        if (mCount >= MAX_MENTRIES) {
          note('masteryFull', str(skill.name));
          noteOverflow(note, str(skill.name));
          break;
        }
        const rb = b + MENTRY_OFF + mCount * 7;
        records[rb] = n.op;
        records[rb + 1] = n.value;
        records[rb + 2] = n.stat;
        records[rb + 3] = n.chance;
        records[rb + 4] = n.zeroFlag;
        mCount++;
      }
      if (!created.length) continue;
      if (mCount >= MAX_MENTRIES) {
        note('masteryFull', str(skill.name));
        noteOverflow(note, str(skill.name));
        break;
      }
      const fb = b + MENTRY_OFF + mCount * 7;
      const fmodStart = fCount;
      for (const c of created) {
        for (const m of c.mods) pushFMod(FKEY[m.key], m.scaling, c.stacks);
      }
      records[fb] = 6; // phase barrier opcode
      records[fb + 5] = fmodStart;
      records[fb + 6] = fCount - fmodStart;
      mCount++;
    }
    // The kernel merges stability / maxStability outcomes into a fixed
    // MAX_OUTCOMES buffer. Each chance-gated bar can double the branch count;
    // merging can only reduce it, so this is a conservative upper bound and a
    // potential silent data loss must be signalled to the caller.
    let outcomeBranches = 1;
    for (let e = 0; e < effectCount; e++) {
      const eb = b + EFF_OFF + e * 7;
      const op = records[eb];
      const chance = records[eb + 3];
      if ((op === OPS.stability || op === OPS.maxStability) && chance > 0 && chance < 1) {
        outcomeBranches *= 2;
      }
    }
    if (outcomeBranches > MAX_OUTCOMES) noteOverflow(note, str(skill.name));

    records[b + 16] = mCount;
    records[b + 17] = fCount;
  }
  return records;
}

function buildNodes(
  data: MirrorData,
  nodeCount: number,
  perturb: boolean,
  skillCount: number,
  note: (kind: string, name: string) => void,
): Float32Array {
  const arr = new Float32Array(nodeCount * NODE);
  const base = data.state;
  const capC = data.caps.completion;
  const capP = data.caps.perfection;
  const roots = Math.max(1, Math.floor(skillCount));
  for (let i = 0; i < nodeCount; i++) {
    const state: MirrorState = { ...base };
    if (perturb) {
      // Harness-only variation from stage7.html; not used by the real search.
      state.pool = Math.min(base.maxPool, Math.max(0, base.pool + (i % 97) * 2 - 50));
      state.stability = Math.min(base.maxStability, Math.max(0, base.stability + (i % 13) - 3));
      state.control = base.control + (i % 5) * 37;
      state.intensity = base.intensity + (i % 7) * 23;
      state.critChance = base.critChance + (i % 3) * 2;
      state.critMultiplier = base.critMultiplier + (i % 4) * 3;
    }
    const frame = buildFrame(data, state, [], {}, {}, note);
    const b = i * NODE;
    // Opening-skill seeding: the shader only honours this at step 0, so deeper
    // levels still choose freely among all techniques.
    arr[b + NODE_ROOT_SLOT] = i % roots;
    arr[b + NODE_SUCCESS_BONUS] = state.successChanceBonus;
    arr[b + 0] = state.pool;
    arr[b + 1] = state.stability;
    arr[b + 2] = frame.buffed.control;
    arr[b + 3] = frame.buffed.intensity;
    arr[b + 4] = frame.buffed.critchance;
    arr[b + 5] = frame.buffed.critmultiplier;
    arr[b + 6] = frame.critFactor;
    arr[b + 7] = frame.qiFactor;
    arr[b + 8] = frame.stabFactor;
    arr[b + 9] = frame.poolPct;
    arr[b + 10] = frame.stabPct;
    arr[b + 11] = state.maxPool;
    arr[b + 12] = state.maxStability;
    arr[b + 13] = state.completion;
    arr[b + 14] = state.perfection;
    arr[b + 15] = capC;
    arr[b + 16] = capP;
    arr[b + 17] = Math.max(0, capC - state.completion);
    arr[b + 18] = Math.max(0, capP - state.perfection);
    arr[b + 19] = frame.boosts.perfection;
    arr[b + 20] = frame.boosts.completion;
    arr[b + 21] = frame.boosts.qi;
    arr[b + 22] = frame.boosts.stability;
    arr[b + 23] = frame.controlMult;
    arr[b + 24] = frame.intensityMult;
    arr[b + 28] = state.control;
    arr[b + 29] = state.intensity;
    arr[b + 30] = state.critChance;
    arr[b + 31] = state.critMultiplier;
    arr[b + 32] = data.targetCompletion;
    arr[b + 33] = data.targetPerfection;
    arr[b + 34] = state.stabilityPenalty;
    arr[b + 35] = state.initialMaxStability;
    arr[b + 36] = frame.poolCostFlat;
    arr[b + 37] = buildBuffCritFactor(
      data,
      state,
      { frame: { critFactor: frame.critFactor } },
      note,
    );
  }
  return arr;
}

function makeCombined(nodes: Float32Array, nodeCount: number): Float32Array {
  const combined = new Float32Array(nodeCount * NS);
  for (let i = 0; i < nodeCount; i++) {
    for (let k = 0; k < NODE; k++) combined[i * NS + k] = nodes[i * NODE + k];
    combined[i * NS + NODE] = 0; // step
    combined[i * NS + NODE + 1] = 1; // alive
  }
  return combined;
}

function toMirrorData(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  condition: string | undefined,
): MirrorData {
  const buffs: MirrorBuff[] = [];
  state.buffs.forEach((tracked, key) => {
    const definition = tracked.definition;
    // Older replay snapshots carry stats/effects on the buff entry itself with
    // no definition (mirror.json does); live states carry them on definition.
    const source = obj(tracked) ?? {};
    buffs.push({
      key,
      name: tracked.name || key,
      stacks: tracked.stacks,
      definition,
      stats: (source.stats ?? definition?.stats ?? {}) as Record<string, Scaling>,
      effects: list(source.effects ?? definition?.effects),
      onFusion: list(source.onFusion ?? definition?.onFusion),
      onRefine: list(source.onRefine ?? definition?.onRefine),
      onStabilize: list(source.onStabilize ?? definition?.onStabilize),
      onSupport: list(source.onSupport ?? definition?.onSupport),
    });
  });
  const base: MirrorState = {
    pool: state.qi,
    maxPool: config.maxQi,
    stability: state.stability,
    maxStability: state.maxStability,
    control: config.baseControl,
    intensity: config.baseIntensity,
    critChance: state.critChance,
    critMultiplier: state.critMultiplier,
    overcrit: state.overcrit,
    overcritCapBonus: state.overcritCapBonus,
    successChanceBonus: state.successChanceBonus,
    qiCostReduction: state.qiCostReduction,
    qiCostPenalty: state.qiCostPenalty,
    stabilityCostReduction: state.stabilityCostReduction,
    stabilityCostPenalty: state.stabilityCostPenalty,
    poolCostFlat: state.poolCostFlat,
    poolCostPercentage: state.poolCostPercentage,
    stabilityCostPercentage: state.stabilityCostPercentage,
    completion: state.completion,
    perfection: state.perfection,
    completionBonus: state.completionBonus,
    stabilityPenalty: state.stabilityPenalty,
    initialMaxStability: state.initialMaxStability,
    controlBuffTurns: num(state.controlBuffTurns, 0),
    controlBuffMultiplier: num(state.controlBuffMultiplier, 1),
    intensityBuffTurns: num(state.intensityBuffTurns, 0),
    intensityBuffMultiplier: num(state.intensityBuffMultiplier, 1),
    harmonyType: config.craftingType,
    harmonyData: state.harmonyData,
    nativeVariables: state.nativeVariables,
  };
  return {
    state: base,
    buffs,
    conditionEffects: getConditionEffectsForConfig(config, condition ?? 'neutral'),
    caps: {
      completion: config.maxCompletion ?? 1e12,
      perfection: config.maxPerfection ?? 1e12,
    },
    targetCompletion: config.targetCompletion ?? 0,
    targetPerfection: config.targetPerfection ?? 0,
    skills,
  };
}

export interface GpuRecordOptions {
  /** Condition used to resolve recipe condition effects (default: neutral). */
  readonly currentCondition?: string;
  /** Number of replicated beam nodes (default: 1). */
  readonly nodeCount?: number;
  /** Harness-only synthetic node variation; never used by the real search. */
  readonly perturbNodes?: boolean;
  readonly beamWidth?: number;
  readonly topK?: number;
}

export interface GpuRecordSet {
  readonly skills: Float32Array;
  readonly nodes: Float32Array;
  readonly skillNames: string[];
  readonly skillCount: number;
  readonly nodeCount: number;
  /** Skipped effect/condition kinds, for diagnostics. */
  readonly notes: Record<string, number>;
  readonly sizes: {
    readonly skillsBytes: number;
    readonly nodesBytes: number;
    readonly deltasBytes: number;
    readonly candidatesBytes: number;
    readonly topBytes: number;
  };
}

/** Build the flat f32 records consumed by GPU_KERNEL_WGSL. */
export function buildGpuRecords(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  options: GpuRecordOptions = {},
): GpuRecordSet {
  const sink = makeNoteSink();
  const skillCount = Math.min(Math.max(1, Math.floor(skills.length)), 64);
  const nodeCount = Math.max(1, Math.floor(options.nodeCount ?? 1));
  const beamWidth = Math.max(1, Math.floor(options.beamWidth ?? nodeCount));
  const topK = Math.max(1, Math.floor(options.topK ?? 1));
  const data = toMirrorData(
    state,
    config,
    skills.slice(0, skillCount),
    options.currentCondition,
  );
  const skillRecords = buildSkills(data, skillCount, sink.note);
  const nodeRecords = buildNodes(
    data,
    nodeCount,
    options.perturbNodes === true,
    skillCount,
    sink.note,
  );
  return {
    skills: skillRecords,
    nodes: makeCombined(nodeRecords, nodeCount),
    skillNames: data.skills.map((skill) => skill.name),
    skillCount,
    nodeCount,
    notes: sink.snapshot(),
    sizes: {
      skillsBytes: skillRecords.byteLength,
      nodesBytes: nodeCount * NS * 4,
      deltasBytes: nodeCount * skillCount * OUT * 4,
      candidatesBytes: beamWidth * topK * CAND * 4,
      topBytes: beamWidth * CAND * 4,
    },
  };
}

// ===========================================================================
// Stateful catalog / full-state nodes.
//
// Separate from the compact kernel above: `NODE`/`NS` and the existing skill
// records are untouched. These buffers are the input of the next-generation
// stateful WGSL transition, which walks active buffs instead of folding their
// stats into a single frame at record-build time.
//
// Capacities come from measuring the 17 replay fixtures, their replay traces
// and `hardest-54` (see tools/gpu-probe + scripts in S:/Temp/opencode):
//   23 active buffs (trace), 99 distinct definitions, 3 stats/buff,
//   4 effects in one trigger list, 6 effects total/buff, 4 initial-state keys,
//   1 event-trigger block with 1 effect, 0 additionalVariables, 54 skills.
// Every capacity below is >= 1.5x the observed maximum, except the two small
// floors (4 variables, 64 skills) which keep the layout aligned with the
// existing skill-record cap while still exceeding any observed value.
// ===========================================================================

/** Active-buff slots in a node (23 observed in the overcraft replay trace). */
export const BUFF_SLOT_CAPACITY = 36;
/** Distinct buff definitions in the catalog (99 observed). */
export const BUFF_DEFS_CAPACITY = 160;
export const BUFF_STATS_CAPACITY = 6;
export const BUFF_EFFECTS_PER_DEF = 12;
export const BUFF_TRIGGER_SLOTS = 7;
/** 5 action triggers + 2 runtime event-trigger blocks (1 observed). */
export const BUFF_ACTION_TRIGGERS = 5;
export const BUFF_VAR_CAPACITY = 4;
export const BUFF_INITIAL_STATE_CAPACITY = 8;
/** Cooldown capacity; matches the compact kernel's 64-skill ceiling. */
export const BUFF_SKILL_CAPACITY = 64;

/** Scaling payload: value, two variable opcodes, eqn factor, two caps, upgrade rule, two dynamic key hashes. */
export const BUFF_SCALING_WORDS = 12;
/** Stat entry = stat-key opcode + scaling payload. */
export const BUFF_STAT_WORDS = 1 + BUFF_SCALING_WORDS;
/** Effect entry = kind opcode + scaling payload + condition/chance/target/mode. */
export const BUFF_EFFECT_WORDS = 1 + BUFF_SCALING_WORDS + 9;
/** Trigger entry = kind opcode, event-name hash, effect offset and count. */
export const BUFF_TRIGGER_WORDS = 4;
export const BUFF_VAR_WORDS = 2;
export const BUFF_INITIAL_WORDS = 2;

/** Definition header layout (all values are f32). */
export const BUFF_DEF = {
  id: 0,
  maxStacks: 1,
  canStack: 2,
  lifetimeKind: 3,
  baseTurns: 4,
  sealedMaxStability: 5,
  overcritChainGrowth: 6,
  statCount: 7,
  statOffset: 8,
  effectCount: 9,
  effectOffset: 10,
  triggerCount: 11,
  triggerOffset: 12,
  varCount: 13,
  varOffset: 14,
  initialCount: 15,
  initialOffset: 16,
  critCycleInterval: 17,
  critCycleChance: 18,
  critCycleOvercrit: 19,
  bonusMaximumQuality: 20,
  flags: 21,
  WORDS: 24,
} as const;

/** `BUFF_DEF.flags` bits. Derived harmony buffs are stripped from the live map
 *  (mirror.mjs:2750-2753) once the replacing subsystem model is available. */
export const BUFF_DEF_FLAGS = {
  derivedHarmony: 1,
  /** skills.ts:1924-1940 — an explicit `control` stat suppresses the
   *  controlBuffTurns multiplier; same for intensity. */
  hasControlStat: 2,
  hasIntensityStat: 4,
} as const;

export const BUFF_DEF_OFF = 0;
export const BUFF_STAT_OFF = BUFF_DEF_OFF + BUFF_DEFS_CAPACITY * BUFF_DEF.WORDS;
export const BUFF_EFFECT_OFF =
  BUFF_STAT_OFF + BUFF_DEFS_CAPACITY * BUFF_STATS_CAPACITY * BUFF_STAT_WORDS;
export const BUFF_TRIGGER_OFF =
  BUFF_EFFECT_OFF + BUFF_DEFS_CAPACITY * BUFF_EFFECTS_PER_DEF * BUFF_EFFECT_WORDS;
export const BUFF_VAR_OFF =
  BUFF_TRIGGER_OFF + BUFF_DEFS_CAPACITY * BUFF_TRIGGER_SLOTS * BUFF_TRIGGER_WORDS;
export const BUFF_INITIAL_OFF =
  BUFF_VAR_OFF + BUFF_DEFS_CAPACITY * BUFF_VAR_CAPACITY * BUFF_VAR_WORDS;
export const BUFF_CATALOG_WORDS =
  BUFF_INITIAL_OFF + BUFF_DEFS_CAPACITY * BUFF_INITIAL_STATE_CAPACITY * BUFF_INITIAL_WORDS;
export const BUFF_CATALOG_BYTES = BUFF_CATALOG_WORDS * 4;

/** Variable/stat opcodes. Dynamic names (`blaze`, `Banked Heat`, ...) carry a 24-bit hash instead. */
export const BUFF_VAR_OPS: Readonly<Record<string, number>> = {
  none: 0,
  flat: 1,
  control: 2,
  intensity: 3,
  stacks: 4,
  maxpool: 5,
  maxstability: 6,
  completion: 7,
  perfection: 8,
  maxcompletion: 9,
  maxperfection: 10,
  critchance: 11,
  critmultiplier: 12,
  overcrit: 13,
  overcritcapbonus: 14,
  pool: 15,
  stability: 16,
  poolcostflat: 17,
  successchancebonus: 18,
  poolcostpercentage: 19,
  stabilitycostpercentage: 20,
  qicostreduction: 21,
  stabilitycostreduction: 22,
  completionboost: 23,
  refinementcritchance: 24,
  // mirror.mjs:1593-1633 reads every one of these from the frame. Without an
  // opcode the catalog dropped them with a `buffStat:` note, so a held buff
  // carrying one of these stats never reached the shader's frame fold.
  perfectionboost: 25,
  qiboost: 26,
  stabilityboost: 27,
  qicostpenalty: 28,
  stabilitycostpenalty: 29,
};
export const BUFF_VAR_DYNAMIC = 100;

export const BUFF_EFFECT_OPS: Readonly<Record<string, number>> = {
  completion: 1,
  perfection: 2,
  stability: 3,
  maxStability: 4,
  pool: 5,
  negate: 6,
  createBuff: 7,
  addStack: 8,
  changeToxicity: 9,
  modifyBuffGroup: 10,
  setState: 11,
  consumeBuff: 12,
};

export const BUFF_CONDITION_OPS: Readonly<Record<string, number>> = {
  none: 0,
  chance: 1,
  buff: 2,
  condition: 3,
};
export const BUFF_CONDITION_MODE_OPS: Readonly<Record<string, number>> = {
  equal: 1,
  more: 2,
  less: 3,
};
export const BUFF_EFFECT_MODE_OPS: Readonly<Record<string, number>> = {
  set: 1,
  add: 2,
  all: 3,
  highest: 4,
  lowest: 5,
  random: 6,
};
export const BUFF_TRIGGER_OPS: Readonly<Record<string, number>> = {
  onAction: 1,
  onFusion: 2,
  onRefine: 3,
  onStabilize: 4,
  onSupport: 5,
  event: 6,
};
/** Lifetime semantics derived from the buff-granting skill's `buffType`. */
export const BUFF_LIFETIME: Readonly<Record<string, number>> = {
  none: 0,
  control: 1,
  intensity: 2,
};

/**
 * Harmony opcode stored in a state node and in frame params. Mirrors
 * `HARMONY_OPS` in scripts/gpu-harness.ts; only the first three advance in
 * mirror.mjs `computeHarmonyAdvance` (2954-3054).
 */
export const HARMONY_OPS: Readonly<Record<string, number>> = {
  forge: 1,
  alchemical: 2,
  resonance: 3,
  eccentricDecree: 4,
  inscription: 5,
  captivatingCadence: 6,
};

/** Technique type ids: the `SKILL_ACTION_TYPE` opcodes, already sorted by name. */
export const HARMONY_TECHNIQUE_IDS: Readonly<Record<string, number>> = {
  fusion: 1,
  refine: 2,
  stabilize: 3,
  support: 4,
};

/** Observed maxima over 17 replay fixtures + hardest-54: charges 2, combo 3,
 *  recommended types 4, reaction-modifier keys 6. All capacities have headroom. */
export const HARMONY_CHARGE_CAPACITY = 4;
export const HARMONY_COMBO_CAPACITY = 4;
export const HARMONY_RECOMMEND_CAPACITY = 4;
export const HARMONY_REACTION_CAPACITY = 8;

export const HARMONY_REACTION_KEYS = [
  'controlMultiplier',
  'intensityMultiplier',
  'critChanceBonus',
  'qiCostReduction',
  'stabilityCostReduction',
  'successChanceBonus',
] as const;

/** Dynamic state-key hashes used by buff trigger effects (mirror.mjs:2836). */
export const HASH_BLAZE = hash24('blaze');
export const HASH_COMPLETION_GAINED = hash24('completiongained');

/** Future-condition slots carried by every state node (engine always shows 3). */
export const CONDITION_QUEUE_SLOTS = 3;

/**
 * Numeric condition codes used in the queue slots. They mirror the engine's
 * `normalizeConditionType` aliases so the WGSL transition kernel can classify
 * an id without string handling. `other` is any unrecognised lowercase name.
 */
export const CONDITION_KIND = {
  neutral: 0,
  positive: 1,
  negative: 2,
  veryPositive: 3,
  veryNegative: 4,
  other: 5,
} as const;

/** Engine `normalizeConditionType` (conditions.ts:17-42) as a numeric code. */
export function conditionKindOf(condition: string | undefined): number {
  const c = String(condition ?? '').toLowerCase();
  switch (c) {
    case 'neutral':
    case 'balanced':
      return CONDITION_KIND.neutral;
    case 'positive':
    case 'harmonious':
      return CONDITION_KIND.positive;
    case 'negative':
    case 'resistant':
      return CONDITION_KIND.negative;
    case 'verypositive':
    case 'excellent':
    case 'brilliant':
      return CONDITION_KIND.veryPositive;
    case 'verynegative':
    case 'corrupted':
      return CONDITION_KIND.veryNegative;
    default:
      return CONDITION_KIND.other;
  }
}

/** Full-state node layout; independent of `NODE`/`NS` used by the compact kernel. */
/**
 * Tail offset shared by the S4 stability-projection outputs. Everything before
 * it keeps its established offsets; only WORDS grows.
 */
const NS_S4_TAIL =
  49 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
  HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY;

export const NS_STATE = {
  completion: 0,
  perfection: 1,
  pool: 2,
  stability: 3,
  maxStability: 4,
  stabilityPenalty: 5,
  step: 6,
  alive: 7,
  control: 8,
  intensity: 9,
  critChance: 10,
  critMultiplier: 11,
  overcrit: 12,
  overcritCapBonus: 13,
  controlMult: 14,
  intensityMult: 15,
  completionBoost: 16,
  perfectionBoost: 17,
  qiBoost: 18,
  stabilityBoost: 19,
  poolCostFlat: 20,
  poolCostPercentage: 21,
  stabilityCostPercentage: 22,
  qiCostReduction: 23,
  qiCostPenalty: 24,
  stabilityCostReduction: 25,
  stabilityCostPenalty: 26,
  successChanceBonus: 27,
  completionBonus: 28,
  harmony: 29,
  /** Current condition id (hash, neutral = 0) as before. */
  conditionId: 30,
  /** Three forecast slots, CONDITION_KIND codes at [base + 0..2]. */
  conditionQueue: 31,
  flags: 31 + CONDITION_QUEUE_SLOTS,
  /** Global control/intensity duration counters (skills.ts:4610-4628). They
   *  cannot be recovered from buff slots: a counter survives its last buff. */
  controlBuffTurns: 32 + CONDITION_QUEUE_SLOTS,
  intensityBuffTurns: 33 + CONDITION_QUEUE_SLOTS,
  /** Live derived-subsystem gate (records.ts derivedHarmonyActiveBit). A
   *  catalog built once cannot know which subsystem is active at a later
   *  node, so the node carries the per-step mask instead. */
  harmonyDerivedMask: 34 + CONDITION_QUEUE_SLOTS,
  /** Live harmony subsystem payload (mirror.mjs:2954-3054). */
  harmonyType: 35 + CONDITION_QUEUE_SLOTS,
  harmonyForgeHeat: 36 + CONDITION_QUEUE_SLOTS,
  harmonyForgeLastBuffedHeat: 37 + CONDITION_QUEUE_SLOTS,
  harmonyAlchCharges: 38 + CONDITION_QUEUE_SLOTS,
  harmonyAlchLastCombo: 38 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY,
  harmonyResonanceType:
    38 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY,
  harmonyResonanceStrength:
    39 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY,
  harmonyResonancePendingType:
    40 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY,
  harmonyResonancePendingCount:
    41 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY,
  harmonyReactionMods:
    42 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY,
  harmonyRecommendMask:
    42 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY,
  buffs:
    43 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY,
  cooldowns:
    43 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5,
  /** Unclamped condition-generation harmony (skills.ts:5916-5926). The clamped
   *  `harmony` above is unsuitable for conditions.ts:98-99 (±110 in corpus). */
  conditionGenerationHarmony:
    43 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY,
  /** Score-input values that used to be substituted from the host engine
   *  state by the depth-2 stand (depth2-harness.ts scoreInputFromNode). */
  toxicity:
    44 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY,
  maxToxicity:
    45 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY,
  controlBuffMultiplier:
    46 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY,
  intensityBuffMultiplier:
    47 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY,
  /** Config-level default used when a buff-granting technique has no explicit
   *  multiplier (skills.ts:4630-4643). */
  defaultBuffMultiplier:
    48 + CONDITION_QUEUE_SLOTS + HARMONY_CHARGE_CAPACITY + HARMONY_COMBO_CAPACITY +
      HARMONY_REACTION_CAPACITY + BUFF_SLOT_CAPACITY * 5 + BUFF_SKILL_CAPACITY,

  // --- S4: StabilityProjection outputs (stability.ts:192-267) -----------------
  /** Minimum cleaned stability over the surviving projection. */
  floorStability: NS_S4_TAIL,
  /** Minimum cleaned maximum over the surviving projection. */
  floorMaxStability: NS_S4_TAIL + 1,
  /** P(s > 0) over the surviving projection. */
  survivalProbability: NS_S4_TAIL + 2,
  /** Sum p*s over outcomes with s>0; the host divides by survivalProbability. */
  survivingStability: NS_S4_TAIL + 3,
  /** Sum p*penalty over outcomes with s>0 (same division boundary). */
  survivingPenalty: NS_S4_TAIL + 4,
  /** E[s|s>0], written by the host in f64 with snapRoundoff (stability.ts:5-8). */
  condStability: NS_S4_TAIL + 5,
  /** E[penalty|s>0], written by the host in f64 with snapRoundoff. */
  condPenalty: NS_S4_TAIL + 6,
  /** Set to 1 when a projection branch is refused at MAX_OUTCOMES. */
  outcomeOverflow: NS_S4_TAIL + 7,
  /** Debug dump: MAX_OUTCOMES x [stability, penalty, probability]. */
  projectionDebug: NS_S4_TAIL + 8,
  projectionCount: NS_S4_TAIL + 8 + MAX_OUTCOMES * 3,
  survivalCount: NS_S4_TAIL + 9 + MAX_OUTCOMES * 3,
  survivalTwin: NS_S4_TAIL + 10 + MAX_OUTCOMES * 3,
  WORDS: NS_S4_TAIL + 11 + MAX_OUTCOMES * 3,
} as const;
export const BUFF_SLOT_WORDS = 5;
export const NS_STATE_BYTES = NS_STATE.WORDS * 4;
export const STATE_FLAGS = {
  finished: 1,
  depletedPool: 2,
  brokenStability: 4,
} as const;

export interface BuffCatalog {
  /** Flat f32 buffer; slot offsets are `BUFF_*_OFF` + per-definition capacity * index. */
  readonly records: Float32Array;
  /** Normalized buff names; the array index is the definition id stored in nodes. */
  readonly names: string[];
  readonly ids: ReadonlyMap<string, number>;
  readonly count: number;
  readonly notes: Record<string, number>;
  readonly sizes: {
    readonly bytes: number;
    readonly defsBytes: number;
    readonly statsBytes: number;
    readonly effectsBytes: number;
    readonly triggersBytes: number;
    readonly varsBytes: number;
    readonly initialBytes: number;
  };
}

export interface StateNodeSet {
  readonly nodes: Float32Array;
  readonly nodeCount: number;
  readonly buffIds: string[];
  readonly notes: Record<string, number>;
  readonly sizes: { readonly bytes: number; readonly nodeBytes: number };
}

type Note = (kind: string, name: string) => void;
type CatalogCore = Omit<BuffCatalog, 'notes'>;

/** FNV-1a truncated to 24 bits so every key hash is exact in f32. */
function hash24(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash & 0xffffff;
}

function varOp(value: unknown): { readonly op: number; readonly hash: number } {
  const raw = str(value).trim();
  if (!raw) return { op: 0, hash: 0 };
  const known = BUFF_VAR_OPS[normalizeBuffName(raw)];
  return known !== undefined
    ? { op: known, hash: 0 }
    : { op: BUFF_VAR_DYNAMIC, hash: hash24(normalizeBuffName(raw)) };
}

/**
 * The shader carries only the first two `initialState` keys of a definition
 * (node v0/v1, see `dynamicVar` in kernelState.ts). A scaling/eqn that names
 * any other dynamic key cannot be reconstructed in f32. Report it instead of
 * silently evaluating it as zero. `blaze`, the only dynamic key in the replay
 * corpus, is declared in initialState slot 0 and is therefore not reported.
 */
function noteDynamicLoss(
  payload: unknown,
  declared: ReadonlySet<string>,
  name: string,
  note: Note,
): void {
  const walk = (node: unknown): void => {
    const s = obj(node);
    if (!s) return;
    for (const field of ['stat', 'scaling']) {
      const raw = str(s[field]).trim();
      if (!raw) continue;
      const key = normalizeBuffName(raw);
      if (BUFF_VAR_OPS[key] === undefined && !declared.has(key)) {
        note('dynamicVarLost', `${name}:${raw}`);
      }
    }
    for (const field of ['max', 'scalingMax']) walk(s[field]);
  };
  walk(payload);
}

function catalogCtx(data: MirrorData, note: Note): EvalCtx {
  const frame = buildFrame(data, data.state, [], {}, {}, note);
  return {
    data,
    upgrades: {},
    frame: {
      buffed: {
        control: frame.buffed.control,
        intensity: frame.buffed.intensity,
        critchance: frame.buffed.critchance,
        critmultiplier: frame.buffed.critmultiplier,
      },
      controlMult: frame.controlMult,
      intensityMult: frame.intensityMult,
    },
    note,
  };
}

/**
 * Encode a game `Scaling` without reimplementing it: numbers and caps go
 * through the same `evalEqn`/`evalScaling` bridge as the compact kernel, while
 * the raw opcode pair lets the shader re-evaluate against a later node state.
 */
function encodeScaling(payload: unknown, ctx: EvalCtx, note: Note): number[] {
  const upgraded = applyUpgrades(payload, ctx.upgrades);
  const node = obj(upgraded);
  const value = typeof upgraded === 'number' ? upgraded : num(node?.value, 0);
  const stat = varOp(node?.stat);
  const scaling = varOp(node?.scaling);
  const eqn = node?.eqn ? evalEqn(str(node.eqn), expressionVariables(ctx, {}), note) : 1;
  const cap = (raw: unknown): [number, number] =>
    raw === undefined ? [0, 0] : [evalScaling(raw, ctx, {}, true), 1];
  const [maxCap, maxOn] = cap(node?.max);
  const [scalingCap, scalingOn] = cap(node?.scalingMax);
  const upgradeKey = str(node?.upgradeKey).trim();
  const rule = upgradeKey ? ctx.upgrades[upgradeKey] : undefined;
  // The catalog is built once with no action upgrades, so a buff payload keeps
  // its upgradeKey hash here. skills.ts:5434-5439 evaluates held buff effects
  // with the CURRENT action's mastery upgrades; the shader resolves the hash
  // against that action's `upgrade` mastery records at eval time. Skill-side
  // payloads already carry the composed rule in `value`, so they keep word 8
  // as the additive term.
  const catalogUpgradeHash =
    !rule && upgradeKey && Object.keys(ctx.upgrades).length === 0
      ? hash24(normalizeBuffName(upgradeKey))
      : 0;
  return [
    value,
    stat.op,
    scaling.op,
    eqn,
    maxCap,
    maxOn,
    scalingCap,
    scalingOn,
    catalogUpgradeHash || (rule?.additive ?? 0),
    rule?.multiplier ?? 1,
    stat.hash,
    scaling.hash,
  ];
}

function encodeEffect(
  raw: unknown,
  ctx: EvalCtx,
  note: Note,
  ids: ReadonlyMap<string, number>,
): number[] {
  const effect = obj(raw) ?? {};
  const kind = str(effect.kind);
  const kindOp = BUFF_EFFECT_OPS[kind] ?? 0;
  if (!kindOp) note('buffEffect', kind || '(none)');
  const scaling = encodeScaling(
    effect.amount ?? effect.stacks ?? effect.value ?? 0,
    ctx,
    note,
  );
  const condition = obj(effect.condition);
  const condKind = condition ? str(condition.kind) : 'none';
  const condOp = BUFF_CONDITION_OPS[condKind] ?? 0;
  if (condition && !condOp) note('buffCondition', condKind);
  const buffHolder = obj(condition?.buff);
  const condName = condition?.buff === 'self'
    ? 'self'
    : normalizeBuffName(buffHolder?.name ?? obj(buffHolder?.definition)?.name);
  const condTarget = condName === 'self' ? -2 : (ids.get(condName) ?? -1);
  const chance = conditionProbability(effect.condition, effect, undefined, ctx.data, note);
  const effectBuff = obj(effect.buff);
  const buffName = normalizeBuffName(
    obj(effectBuff?.definition)?.name ?? effectBuff?.name,
  );
  const buffTarget = ids.get(buffName) ?? -1;
  const stateKey = str(effect.key).trim();
  return [
    kindOp,
    ...scaling,
    condOp,
    num(condition?.percentage ?? condition?.count, 0),
    BUFF_CONDITION_MODE_OPS[str(condition?.mode)] ?? 0,
    condTarget,
    chance,
    buffTarget,
    stateKey ? hash24(normalizeBuffName(stateKey)) : 0,
    (effect.value !== undefined ? 1 : 0) | (kind === 'negate' ? 2 : 0),
    BUFF_EFFECT_MODE_OPS[str(effect.mode)] ?? 0,
  ];
}

interface CollectedBuffs {
  readonly defs: Map<string, Raw>;
  readonly lifetimes: Map<string, { readonly kind: number; readonly turns: number }>;
}

/** Walk every buff reachable from a state or from any technique payload. */
function collectBuffs(
  state: CraftingState,
  skills: SkillDefinition[],
): CollectedBuffs {
  const defs = new Map<string, Raw>();
  const lifetimes = new Map<string, { kind: number; turns: number }>();
  const walked = new Set<object>();
  const score = (def: Raw): number => {
    const lists: unknown[] = [
      def.effects,
      def.onFusion,
      def.onRefine,
      def.onStabilize,
      def.onSupport,
    ];
    return (
      Object.keys(obj(def.stats) ?? {}).length * 4 +
      lists.reduce<number>((sum, entryList) => sum + list(entryList).length, 0) +
      Object.keys(obj(def.additionalVariables) ?? {}).length +
      Object.keys(obj(def.initialState) ?? {}).length
    );
  };
  const add = (raw: unknown): void => {
    const holder = obj(raw);
    if (!holder || walked.has(holder)) return;
    walked.add(holder);
    const def = obj(holder.definition) ?? holder;
    const name = normalizeBuffName(def.name);
    if (!name) return;
    const previous = defs.get(name);
    if (!previous || score(def) > score(previous)) defs.set(name, def);
    const lists = [
      def.effects,
      def.onFusion,
      def.onRefine,
      def.onStabilize,
      def.onSupport,
      ...list(def.triggeredEffects).map((block) => obj(block)?.effects),
    ];
    for (const effects of lists) {
      for (const effect of list(effects)) {
        const entry = obj(effect);
        if (!entry) continue;
        if (entry.buff) add(entry.buff);
        const condition = obj(entry.condition);
        if (condition && typeof condition.buff === 'object') add(condition.buff);
      }
    }
  };

  state.buffs.forEach((tracked, key) => {
    add(tracked.definition ?? { ...tracked, name: tracked.name || key });
  });
  for (const skill of skills) {
    const rawSkill = obj(skill) ?? {};
    if (rawSkill.grantedBuff) add(rawSkill.grantedBuff);
    const buffType = rawSkill.buffType;
    const lifetimeKind =
      buffType === 1 || buffType === 'control' || buffType === 'CONTROL'
        ? BUFF_LIFETIME.control
        : buffType === 2 || buffType === 'intensity' || buffType === 'INTENSITY'
          ? BUFF_LIFETIME.intensity
          : BUFF_LIFETIME.none;
    if (lifetimeKind !== BUFF_LIFETIME.none) {
      const grant = normalizeBuffName(obj(rawSkill.grantedBuff)?.name);
      if (grant) {
        lifetimes.set(grant, { kind: lifetimeKind, turns: num(rawSkill.buffDuration, 0) });
      }
    }
    for (const effect of list(rawSkill.effects)) {
      const entry = obj(effect);
      if (entry?.buff) add(entry.buff);
    }
    for (const mastery of list(rawSkill.masteryEntries)) {
      for (const effect of list(obj(mastery)?.effects)) {
        const entry = obj(effect);
        if (entry?.buff) add(entry.buff);
      }
    }
  }
  return { defs, lifetimes };
}

function catalogInternal(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  note: Note,
): CatalogCore {
  const data = toMirrorData(state, config, skills.slice(0, BUFF_SKILL_CAPACITY), undefined);
  const ctx = catalogCtx(data, note);
  const collected = collectBuffs(state, skills.slice(0, BUFF_SKILL_CAPACITY));
  const names = Array.from(collected.defs.keys()).sort((a, b) => a.localeCompare(b));
  const ids = new Map(names.map((name, index) => [name, index] as const));
  const records = new Float32Array(BUFF_CATALOG_WORDS);
  const count = Math.min(names.length, BUFF_DEFS_CAPACITY);
  if (names.length > BUFF_DEFS_CAPACITY) {
    for (const name of names.slice(BUFF_DEFS_CAPACITY)) note('buffOverflow', name);
  }

  for (let id = 0; id < count; id++) {
    const name = names[id];
    const def = collected.defs.get(name) ?? {};
    const declaredVars = new Set(
      Object.keys(obj(def.initialState) ?? {}).map((key) => normalizeBuffName(key)),
    );
    const base = id * BUFF_DEF.WORDS;
    const lifetime = collected.lifetimes.get(name);
    records[base + BUFF_DEF.id] = id;
    records[base + BUFF_DEF.maxStacks] = num(def.maxStacks, 0);
    records[base + BUFF_DEF.canStack] =
      def.canStack === true || num(def.maxStacks, 0) > 1 ? 1 : 0;
    records[base + BUFF_DEF.lifetimeKind] = lifetime?.kind ?? BUFF_LIFETIME.none;
    records[base + BUFF_DEF.baseTurns] = lifetime?.turns ?? 0;
    records[base + BUFF_DEF.sealedMaxStability] = def.sealedMaxStability === true ? 1 : 0;
    records[base + BUFF_DEF.overcritChainGrowth] = num(def.overcritChainGrowth, 0);
    const cycle = obj(def.critCycle);
    records[base + BUFF_DEF.critCycleInterval] = num(cycle?.interval, 0);
    records[base + BUFF_DEF.critCycleChance] = num(cycle?.chance ?? cycle?.critchance, 0);
    records[base + BUFF_DEF.critCycleOvercrit] = num(cycle?.overcrit, 0);
    records[base + BUFF_DEF.flags] =
      (isDerivedHarmonyCandidate(
        data,
        {
          key: name,
          name: str(def.name) || name,
          stacks: 1,
          stats: (obj(def.stats) ?? {}) as Record<string, Scaling>,
          effects: list(def.effects),
        },
      )
        ? BUFF_DEF_FLAGS.derivedHarmony
        : 0) |
      (obj(def.stats)?.control !== undefined ? BUFF_DEF_FLAGS.hasControlStat : 0) |
      (obj(def.stats)?.intensity !== undefined ? BUFF_DEF_FLAGS.hasIntensityStat : 0);
    if (def.bonusMaximumQuality !== undefined) {
      noteDynamicLoss(def.bonusMaximumQuality, declaredVars, name, note);
      records[base + BUFF_DEF.bonusMaximumQuality] = evalScaling(
        def.bonusMaximumQuality,
        ctx,
        {},
        true,
      );
    }

    const stats = obj(def.stats) ?? {};
    const statBase = BUFF_STAT_OFF + id * BUFF_STATS_CAPACITY * BUFF_STAT_WORDS;
    let statCount = 0;
    for (const [key, payload] of Object.entries(stats)) {
      if (statCount >= BUFF_STATS_CAPACITY) {
        note('buffOverflow', name);
        break;
      }
      const op = BUFF_VAR_OPS[normalizeBuffName(key)];
      if (!op) {
        note('buffStat', key);
        continue;
      }
      noteDynamicLoss(payload, declaredVars, name, note);
      const encoded = encodeScaling(payload, ctx, note);
      const entry = statBase + statCount * BUFF_STAT_WORDS;
      records[entry] = op;
      for (let k = 0; k < BUFF_SCALING_WORDS; k++) records[entry + 1 + k] = encoded[k];
      statCount++;
    }
    records[base + BUFF_DEF.statCount] = statCount;
    records[base + BUFF_DEF.statOffset] = statBase;

    const effectBase = BUFF_EFFECT_OFF + id * BUFF_EFFECTS_PER_DEF * BUFF_EFFECT_WORDS;
    const triggerBase = BUFF_TRIGGER_OFF + id * BUFF_TRIGGER_SLOTS * BUFF_TRIGGER_WORDS;
    let effectCount = 0;
    let triggerCount = 0;
    const addTrigger = (kindOp: number, effects: unknown, eventHash = 0): void => {
      const start = effectCount;
      for (const raw of list(effects)) {
        if (effectCount >= BUFF_EFFECTS_PER_DEF) {
          note('buffOverflow', name);
          break;
        }
        const effectRec = obj(raw) ?? {};
        noteDynamicLoss(
          effectRec.amount ?? effectRec.stacks ?? effectRec.value,
          declaredVars,
          name,
          note,
        );
        const encoded = encodeEffect(raw, ctx, note, ids);
        const entry = effectBase + effectCount * BUFF_EFFECT_WORDS;
        for (let k = 0; k < BUFF_EFFECT_WORDS; k++) records[entry + k] = encoded[k] ?? 0;
        effectCount++;
      }
      const entry = triggerBase + triggerCount * BUFF_TRIGGER_WORDS;
      records[entry] = kindOp;
      records[entry + 1] = eventHash;
      records[entry + 2] = start;
      records[entry + 3] = effectCount - start;
      triggerCount++;
    };
    addTrigger(BUFF_TRIGGER_OPS.onAction, def.effects);
    addTrigger(BUFF_TRIGGER_OPS.onFusion, def.onFusion);
    addTrigger(BUFF_TRIGGER_OPS.onRefine, def.onRefine);
    addTrigger(BUFF_TRIGGER_OPS.onStabilize, def.onStabilize);
    addTrigger(BUFF_TRIGGER_OPS.onSupport, def.onSupport);
    const blocks = list(def.triggeredEffects);
    for (const block of blocks.slice(0, BUFF_TRIGGER_SLOTS - BUFF_ACTION_TRIGGERS)) {
      const payload = obj(block);
      addTrigger(
        BUFF_TRIGGER_OPS.event,
        payload?.effects,
        hash24(normalizeBuffName(str(payload?.trigger))),
      );
    }
    if (blocks.length > BUFF_TRIGGER_SLOTS - BUFF_ACTION_TRIGGERS) {
      note('buffOverflow', name);
    }
    records[base + BUFF_DEF.effectCount] = effectCount;
    records[base + BUFF_DEF.effectOffset] = effectBase;
    records[base + BUFF_DEF.triggerCount] = triggerCount;
    records[base + BUFF_DEF.triggerOffset] = triggerBase;

    const vars = obj(def.additionalVariables) ?? {};
    const varBase = BUFF_VAR_OFF + id * BUFF_VAR_CAPACITY * BUFF_VAR_WORDS;
    let varCount = 0;
    for (const [key, value] of Object.entries(vars)) {
      if (varCount >= BUFF_VAR_CAPACITY) {
        note('buffOverflow', name);
        break;
      }
      records[varBase + varCount * BUFF_VAR_WORDS] = hash24(normalizeBuffName(key));
      records[varBase + varCount * BUFF_VAR_WORDS + 1] = num(value, 0);
      varCount++;
    }
    records[base + BUFF_DEF.varCount] = varCount;
    records[base + BUFF_DEF.varOffset] = varBase;

    const initial = obj(def.initialState) ?? {};
    const initialBase = BUFF_INITIAL_OFF + id * BUFF_INITIAL_STATE_CAPACITY * BUFF_INITIAL_WORDS;
    let initialCount = 0;
    for (const [key, value] of Object.entries(initial)) {
      if (initialCount >= BUFF_INITIAL_STATE_CAPACITY) {
        note('buffOverflow', name);
        break;
      }
      // `initialState` values are game expressions; resolve them through the
      // same eqn bridge instead of hard-coding a parser here.
      const seed = evalEqn(str(value), expressionVariables(ctx, {}), note);
      records[initialBase + initialCount * BUFF_INITIAL_WORDS] = hash24(normalizeBuffName(key));
      records[initialBase + initialCount * BUFF_INITIAL_WORDS + 1] = seed;
      initialCount++;
    }
    records[base + BUFF_DEF.initialCount] = initialCount;
    records[base + BUFF_DEF.initialOffset] = initialBase;
  }

  return {
    records,
    names,
    ids,
    count,
    sizes: {
      bytes: BUFF_CATALOG_BYTES,
      defsBytes: BUFF_DEFS_CAPACITY * BUFF_DEF.WORDS * 4,
      statsBytes: BUFF_DEFS_CAPACITY * BUFF_STATS_CAPACITY * BUFF_STAT_WORDS * 4,
      effectsBytes: BUFF_DEFS_CAPACITY * BUFF_EFFECTS_PER_DEF * BUFF_EFFECT_WORDS * 4,
      triggersBytes: BUFF_DEFS_CAPACITY * BUFF_TRIGGER_SLOTS * BUFF_TRIGGER_WORDS * 4,
      varsBytes: BUFF_DEFS_CAPACITY * BUFF_VAR_CAPACITY * BUFF_VAR_WORDS * 4,
      initialBytes:
        BUFF_DEFS_CAPACITY * BUFF_INITIAL_STATE_CAPACITY * BUFF_INITIAL_WORDS * 4,
    },
  };
}

/**
 * Pure buff catalog for the stateful shader. Notes are keyed
 * `buffOverflow:<name>` whenever a fixed-capacity pool would drop data.
 */
export function buildBuffCatalog(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  note?: Note,
): BuffCatalog {
  const sink = makeNoteSink();
  const forward: Note = (kind, name) => {
    sink.note(kind, name);
    note?.(kind, name);
  };
  const catalog = catalogInternal(state, config, skills, forward);
  return { ...catalog, notes: sink.snapshot() };
}

/** Stable 24-bit id for the condition string in force (neutral = 0). */
export function conditionIdOf(condition: string | undefined): number {
  const normalized = normalizeBuffName(condition ?? 'neutral');
  return normalized === 'neutral' || normalized === 'balanced'
    ? 0
    : hash24(normalized);
}

/** Encode one subsystem payload into a node. Fields absent from the snapshot
 *  use -1 sentinels so the shader can reconstruct the model's `undefined`. */
function encodeHarmonyNode(
  nodes: Float32Array,
  base: number,
  state: CraftingState,
  config: OptimizerConfig,
  note: Note,
): void {
  const hd = obj(state.harmonyData);
  const type = str(obj(config)?.craftingType);
  nodes[base + NS_STATE.harmonyType] = HARMONY_OPS[type] ?? 0;
  const active =
    (type === 'forge' && Boolean(hd?.forgeWorks)) ||
    (type === 'resonance' && Boolean(hd?.resonance)) ||
    (type === 'eccentricDecree' && Boolean(hd?.eccentricDecree)) ||
    (type === 'alchemical' &&
      obj(hd?.additionalData)?.alchemicalReactionModifiers !== undefined);
  nodes[base + NS_STATE.harmonyDerivedMask] = active ? derivedHarmonyActiveBit(type) : 0;

  const clampHeat = (value: unknown): number =>
    Math.max(0, Math.min(10, Math.floor(num(value, 0))));
  const fw = obj(hd?.forgeWorks);
  nodes[base + NS_STATE.harmonyForgeHeat] = fw ? clampHeat(fw.heat) : 0;
  nodes[base + NS_STATE.harmonyForgeLastBuffedHeat] =
    fw && fw.lastBuffedHeat !== undefined && fw.lastBuffedHeat !== null
      ? clampHeat(fw.lastBuffedHeat)
      : -1;

  const aa = obj(hd?.alchemicalArts);
  const charges = list(aa?.charges).map((v) => HARMONY_TECHNIQUE_IDS[str(v)] ?? 0);
  const combo = list(aa?.lastCombo).map((v) => HARMONY_TECHNIQUE_IDS[str(v)] ?? 0);
  if (charges.length > HARMONY_CHARGE_CAPACITY || combo.length > HARMONY_COMBO_CAPACITY) {
    note('harmonyDataLost', 'alchemicalArts overflow');
  }
  for (let i = 0; i < HARMONY_CHARGE_CAPACITY; i++) {
    nodes[base + NS_STATE.harmonyAlchCharges + i] = charges[i] ?? -1;
  }
  for (let i = 0; i < HARMONY_COMBO_CAPACITY; i++) {
    nodes[base + NS_STATE.harmonyAlchLastCombo + i] = combo[i] ?? -1;
  }
  const mods = obj(obj(hd?.additionalData)?.alchemicalReactionModifiers);
  const reaction = HARMONY_REACTION_KEYS.map((key) =>
    mods ? num(mods[key], key.endsWith('Multiplier') ? 1 : 0) : key.endsWith('Multiplier') ? 1 : 0,
  );
  for (let i = 0; i < HARMONY_REACTION_CAPACITY; i++) {
    nodes[base + NS_STATE.harmonyReactionMods + i] = reaction[i] ?? 0;
  }
  if (hd?.additionalData && typeof hd.additionalData === 'object') {
    for (const key of Object.keys(hd.additionalData as Raw)) {
      if (key !== 'alchemicalReactionModifiers') {
        note('harmonyDataLost', `additionalData.${key}`);
      }
    }
  }
  if (mods) {
    for (const key of Object.keys(mods)) {
      if (!(HARMONY_REACTION_KEYS as readonly string[]).some(
        (known) => normalizeBuffName(known) === normalizeBuffName(key),
      )) {
        note('harmonyDataLost', `alchemicalReactionModifiers.${key}`);
      }
    }
  }

  const res = obj(hd?.resonance);
  nodes[base + NS_STATE.harmonyResonanceType] = res?.resonance
    ? HARMONY_TECHNIQUE_IDS[str(res.resonance)] ?? -1
    : -1;
  nodes[base + NS_STATE.harmonyResonanceStrength] = res ? num(res.strength, 0) : 0;
  nodes[base + NS_STATE.harmonyResonancePendingType] = res?.pendingResonance
    ? HARMONY_TECHNIQUE_IDS[str(res.pendingResonance)] ?? -1
    : -1;
  nodes[base + NS_STATE.harmonyResonancePendingCount] = res ? num(res.pendingCount, 0) : 0;

  let mask = 0;
  const recommended = list(hd?.recommendedTechniqueTypes);
  if (recommended.length > HARMONY_RECOMMEND_CAPACITY) {
    note('harmonyDataLost', 'recommendedTechniqueTypes overflow');
  }
  for (const name of recommended.slice(0, HARMONY_RECOMMEND_CAPACITY)) {
    const id = HARMONY_TECHNIQUE_IDS[str(name)];
    if (id) mask |= 1 << (id - 1);
  }
  nodes[base + NS_STATE.harmonyRecommendMask] = mask;
}

/**
 * Replicated full-state input nodes. Every node carries the same state; the
 * shader mutates its own copy during the beam search.
 */
export function buildStateNodes(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  nodeCount = 1,
  sharedCatalog?: BuffCatalog,
): StateNodeSet {
  const sink = makeNoteSink();
  const count = Math.max(1, Math.floor(nodeCount));
  const skillCount = Math.min(skills.length, BUFF_SKILL_CAPACITY);
  // A verification stand may pin the catalog so two states use the same ids;
  // otherwise ids are assigned from the state's own reachable buffs.
  const catalog = sharedCatalog ??
    catalogInternal(state, config, skills.slice(0, skillCount), sink.note);
  const data = toMirrorData(state, config, skills.slice(0, skillCount), undefined);
  const frame = buildFrame(data, data.state, [], {}, {}, sink.note);
  const nodes = new Float32Array(count * NS_STATE.WORDS);
  for (let node = 0; node < count; node++) {
    for (let slot = 0; slot < BUFF_SLOT_CAPACITY; slot++) {
      nodes[node * NS_STATE.WORDS + NS_STATE.buffs + slot * BUFF_SLOT_WORDS] = -1;
    }
  }
  const cooldowns = new Map<string, number>();
  state.cooldowns.forEach((turns, key) => {
    cooldowns.set(normalizeBuffName(key), num(turns, 0));
  });

  for (let node = 0; node < count; node++) {
    const b = node * NS_STATE.WORDS;
    nodes[b + NS_STATE.completion] = state.completion;
    nodes[b + NS_STATE.perfection] = state.perfection;
    nodes[b + NS_STATE.pool] = state.qi;
    nodes[b + NS_STATE.stability] = state.stability;
    nodes[b + NS_STATE.maxStability] = state.maxStability;
    nodes[b + NS_STATE.stabilityPenalty] = state.stabilityPenalty;
    nodes[b + NS_STATE.step] = state.step;
    nodes[b + NS_STATE.alive] = state.stability > 0 ? 1 : 0;
    nodes[b + NS_STATE.controlBuffTurns] = state.controlBuffTurns;
    nodes[b + NS_STATE.intensityBuffTurns] = state.intensityBuffTurns;
    nodes[b + NS_STATE.control] = config.baseControl;
    nodes[b + NS_STATE.intensity] = config.baseIntensity;
    nodes[b + NS_STATE.critChance] = state.critChance;
    nodes[b + NS_STATE.critMultiplier] = state.critMultiplier;
    nodes[b + NS_STATE.overcrit] = state.overcrit;
    nodes[b + NS_STATE.overcritCapBonus] = state.overcritCapBonus;
    nodes[b + NS_STATE.controlMult] = frame.controlMult;
    nodes[b + NS_STATE.intensityMult] = frame.intensityMult;
    nodes[b + NS_STATE.completionBoost] = frame.boosts.completion;
    nodes[b + NS_STATE.perfectionBoost] = frame.boosts.perfection;
    nodes[b + NS_STATE.qiBoost] = frame.boosts.qi;
    nodes[b + NS_STATE.stabilityBoost] = frame.boosts.stability;
    nodes[b + NS_STATE.poolCostFlat] = frame.poolCostFlat;
    nodes[b + NS_STATE.poolCostPercentage] = state.poolCostPercentage;
    nodes[b + NS_STATE.stabilityCostPercentage] = state.stabilityCostPercentage;
    nodes[b + NS_STATE.qiCostReduction] = state.qiCostReduction;
    nodes[b + NS_STATE.qiCostPenalty] = state.qiCostPenalty;
    nodes[b + NS_STATE.stabilityCostReduction] = state.stabilityCostReduction;
    nodes[b + NS_STATE.stabilityCostPenalty] = state.stabilityCostPenalty;
    nodes[b + NS_STATE.successChanceBonus] = state.successChanceBonus;
    nodes[b + NS_STATE.completionBonus] = state.completionBonus;
    nodes[b + NS_STATE.harmony] = state.harmony;
    // conditions.ts:98-99 reads the unclamped generation harmony, not the
    // ±100-clamped display bar (skills.ts:5916-5926).
    nodes[b + NS_STATE.conditionGenerationHarmony] =
      state.conditionGenerationHarmony ?? state.harmony;
    // Score-input values carried in the node so scoring never reads host data.
    nodes[b + NS_STATE.toxicity] = num(state.toxicity, 0);
    nodes[b + NS_STATE.maxToxicity] = num(state.maxToxicity, 0);
    nodes[b + NS_STATE.controlBuffMultiplier] = num(state.controlBuffMultiplier, 1);
    nodes[b + NS_STATE.intensityBuffMultiplier] = num(state.intensityBuffMultiplier, 1);
    nodes[b + NS_STATE.defaultBuffMultiplier] = num(config.defaultBuffMultiplier, 0);
    nodes[b + NS_STATE.conditionId] = conditionIdOf(str(obj(config)?.currentCondition));
    encodeHarmonyNode(nodes, b, state, config, sink.note);
    nodes[b + NS_STATE.flags] =
      (state.completion >= (config.targetCompletion ?? Infinity) &&
      state.perfection >= (config.targetPerfection ?? Infinity)
        ? STATE_FLAGS.finished
        : 0) |
      (state.qi <= 0 ? STATE_FLAGS.depletedPool : 0) |
      (state.stability <= 0 ? STATE_FLAGS.brokenStability : 0);

    let slot = 0;
    state.buffs.forEach((tracked, key) => {
      const name = normalizeBuffName(tracked.name || key);
      const id = catalog.ids.get(name);
      if (id === undefined) {
        sink.note('buffMissing', name);
        return;
      }
      if (slot >= BUFF_SLOT_CAPACITY) {
        sink.note('buffOverflow', name);
        return;
      }
      const entry = b + NS_STATE.buffs + slot * BUFF_SLOT_WORDS;
      nodes[entry] = id;
      nodes[entry + 1] = num(tracked.stacks, 0);
      const lifetimeKind = catalog.records[id * BUFF_DEF.WORDS + BUFF_DEF.lifetimeKind];
      nodes[entry + 2] =
        lifetimeKind === BUFF_LIFETIME.control
          ? state.controlBuffTurns
          : lifetimeKind === BUFF_LIFETIME.intensity
            ? state.intensityBuffTurns
            : 0;
      const internal = obj(tracked.internalState) ?? {};
      const initialBase =
        BUFF_INITIAL_OFF + id * BUFF_INITIAL_STATE_CAPACITY * BUFF_INITIAL_WORDS;
      const initialCount = catalog.records[id * BUFF_DEF.WORDS + BUFF_DEF.initialCount];
      let positional = 0;
      for (const [varKey, value] of Object.entries(internal)) {
        const hash = hash24(normalizeBuffName(varKey));
        let target = -1;
        for (let k = 0; k < initialCount; k++) {
          if (catalog.records[initialBase + k * BUFF_INITIAL_WORDS] === hash) {
            target = k;
            break;
          }
        }
        if (target < 0 || target >= 2) target = positional++;
        if (target >= 2) {
          sink.note('buffOverflow', `${name}:internalState`);
          continue;
        }
        nodes[entry + 3 + target] = num(value, 0);
      }
      slot++;
    });

    for (let i = 0; i < skillCount; i++) {
      const skill = skills[i];
      const key = normalizeBuffName(skill.key || skill.name);
      const name = normalizeBuffName(skill.name);
      nodes[b + NS_STATE.cooldowns + i] =
        cooldowns.get(key) ?? cooldowns.get(name) ?? 0;
    }
  }

  return {
    nodes,
    nodeCount: count,
    buffIds: catalog.names,
    notes: sink.snapshot(),
    sizes: { bytes: nodes.byteLength, nodeBytes: NS_STATE_BYTES },
  };
}

// ===========================================================================
// Stateful skill records.
//
// The compact `REC` records above pre-evaluate every scaling against one
// snapshot; these records keep the raw 12-word payload so a stateful WGSL
// transition can re-evaluate each skill against every node's own frame. The
// phase order mirrors tools/gpu-probe/mirror.mjs:
//   phasePaymentCosts -> phaseMasteryFrame -> phaseSuccessRoll ->
//   phaseApplyEffects -> phaseBuffEffects -> phaseCleanup
// Buff definitions live in `BuffCatalog`; nodes live in `NS_STATE`.
//
// Capacities measured over the 17 replay fixtures + hardest-54 (329 skills):
//   9 technique effects (Stellar Infusion), 4 mastery entries (Brilliant
//   Respite), 4 mastery records, 5 created buffs (Focus), 3 cost-relevant
//   creations (Skillful Manipulation), 2 payment branches (Delayed Fusion),
//   4 keys in a scaling payload, 2 keys in a stacks payload, cooldown 99999999.
// All pool capacities are >= 1.5x the observed per-skill maximum.
// ===========================================================================

/** 9 observed -> 16 (1.78x). */
export const SKILL_EFFECT_CAPACITY = 16;
/** 2 observed branches -> 8 (4x); also covers 3 chance-gated creations. */
export const SKILL_COST_BRANCH_CAPACITY = 8;
/** 4 observed mastery entries / records -> 8 (2x). */
export const SKILL_MASTERY_CAPACITY = 8;
/** 5 observed createBuff effects -> 8 (1.6x). */
export const SKILL_CREATED_BUFF_CAPACITY = 8;

/** Reuse the catalog's encoding: value, 2 var opcodes, eqn, 2 caps, upgrade, 2 hashes. */
export const SKILL_SCALING_WORDS = BUFF_SCALING_WORDS;

/** kind, scaling, cond kind/mode/target, chance, crit mode, target bar, flags. */
export const SKILL_EFFECT_WORDS = 1 + SKILL_SCALING_WORDS + 7;
/**
 * probability, qi payload, stability payload, requiredPostStability, mastery
 * flag and a bitmask over created-buff entries whose stats alter the cost.
 */
export const SKILL_COST_BRANCH_WORDS = 1 + SKILL_SCALING_WORDS * 2 + 3;
/** kind, scaling, condition (3), chance, count, effect kind/ref, upgrade (2), entry, entry chance, flags, upgrade key hash. */
export const SKILL_MASTERY_WORDS = 1 + SKILL_SCALING_WORDS + 13;
/** catalog buff id, stacks payload, chance, flags, condition (kind/mode/target/param/factor). */
export const SKILL_CREATED_BUFF_WORDS = 1 + SKILL_SCALING_WORDS + 7;

/** Per-skill header (all f32). */
export const SKILL_STATE = {
  successChance: 0,
  consumesTurn: 1,
  actionKind: 2,
  stabilityGain: 3,
  maxStabilityChange: 4,
  restoresMaxStabilityToFull: 5,
  conditionRequirement: 6,
  cooldown: 7,
  buffCostId: 8,
  buffCostAmount: 9,
  effectCount: 10,
  effectOffset: 11,
  costBranchCount: 12,
  costBranchOffset: 13,
  masteryCount: 14,
  masteryOffset: 15,
  createdBuffCount: 16,
  createdBuffOffset: 17,
  restoresQi: 18,
  flags: 19,
  /** Scalar `qiRestore`, zeroed when the technique itself has a pool effect. */
  qiRestore: 20,
  /** Action type opcode (fusion/refine/stabilize/support) for buff triggers. */
  actionType: 21,
  reserved2: 22,
  reserved3: 23,
  /** skills.ts:4630-4643 — per-technique buff multiplier override. */
  buffMultiplier: 24,
  WORDS: 25,
} as const;

/** `actionKind` opcode: items can be used without consuming a turn. */
export const SKILL_ACTION_KIND: Readonly<Record<string, number>> = {
  skill: 1,
  item: 2,
};
/** `buffType` opcode (state.ts:111-115 BuffType.CONTROL / INTENSITY). */
export const SKILL_BUFF_TYPE: Readonly<Record<string, number>> = {
  none: 0,
  control: 1,
  intensity: 2,
};
/** `actionType` opcode: selects a buff's per-turn action trigger (mirror.mjs ACTION_EFFECTS). */
export const SKILL_ACTION_TYPE: Readonly<Record<string, number>> = {
  fusion: 1,
  refine: 2,
  stabilize: 3,
  support: 4,
};
/** `mastery.kind` opcode; `effect` entries use `SKILL_MASTERY.effectKind` for the payload. */
export const SKILL_MASTERY_OPS: Readonly<Record<string, number>> = {
  effect: 1,
  upgrade: 2,
  control: 3,
  intensity: 4,
  critchance: 5,
  critmultiplier: 6,
};
/** `effect.targetBar` opcode: the numeric bar/shortfall an effect writes to. */
export const SKILL_EFFECT_BARS: Readonly<Record<string, number>> = {
  none: 0,
  completion: 1,
  perfection: 2,
  stability: 3,
  maxStability: 4,
  pool: 5,
};
/** Header `flags` bits. `orderedNumeric` is catalog/node-derived: any active
 * buff with effect interceptors switches the engine to the ordered path. */
export const SKILL_STATE_FLAGS = {
  hasEffects: 1,
  isDisciplinedTouch: 2,
  restoresQi: 4,
  preventsMaxStabilityDecay: 8,
  sealedMaxStability: 16,
  noQiCost: 32,
  hasBuffCost: 64,
  hasConditionRequirement: 128,
  orderedNumeric: 256,
} as const;
/** Effect `flags` bits. */
export const SKILL_EFFECT_FLAGS = {
  zero: 1,
  negate: 2,
  directValue: 4,
} as const;
/** Created-buff `flags` bits. */
export const SKILL_CREATED_FLAGS = {
  fromMastery: 1,
  costRelevant: 2,
  /** Held in the state used to build the records, so the cost branch omitted
   *  this creation; the shader re-adds it when a later node no longer holds it. */
  costSuppressedAtBuild: 4,
} as const;

/** Entry layouts, in f32 words from the start of each entry. */
export const SKILL_EFFECT = {
  kind: 0,
  scaling: 1,
  condKind: 1 + SKILL_SCALING_WORDS,
  condMode: 2 + SKILL_SCALING_WORDS,
  condTarget: 3 + SKILL_SCALING_WORDS,
  chance: 4 + SKILL_SCALING_WORDS,
  critMode: 5 + SKILL_SCALING_WORDS,
  targetBar: 6 + SKILL_SCALING_WORDS,
  flags: 7 + SKILL_SCALING_WORDS,
  WORDS: SKILL_EFFECT_WORDS,
} as const;

export const SKILL_COST_BRANCH = {
  probability: 0,
  qiScaling: 1,
  stabilityScaling: 1 + SKILL_SCALING_WORDS,
  requiredPostStability: 1 + SKILL_SCALING_WORDS * 2,
  masteryFlag: 2 + SKILL_SCALING_WORDS * 2,
  creationMask: 3 + SKILL_SCALING_WORDS * 2,
  WORDS: SKILL_COST_BRANCH_WORDS,
} as const;

export const SKILL_MASTERY = {
  kind: 0,
  scaling: 1,
  condKind: 1 + SKILL_SCALING_WORDS,
  condMode: 2 + SKILL_SCALING_WORDS,
  condTarget: 3 + SKILL_SCALING_WORDS,
  chance: 4 + SKILL_SCALING_WORDS,
  count: 5 + SKILL_SCALING_WORDS,
  effectKind: 6 + SKILL_SCALING_WORDS,
  effectRef: 7 + SKILL_SCALING_WORDS,
  upgradeAdditive: 8 + SKILL_SCALING_WORDS,
  upgradeMultiplier: 9 + SKILL_SCALING_WORDS,
  /**
   * skills.ts:4300-4341 — every entry is one engine transition with its own
   * evaluation frame and one floor per bar; records of the same entry share
   * this ordinal so the shader can rebuild that boundary.
   */
  entry: 10 + SKILL_SCALING_WORDS,
  /**
   * The entry's own condition probability. The engine passes it as the
   * transition's `successChance`, so it lands AFTER the critted gain is
   * floored (skills.ts:3105-3116), unlike the per-effect factor in `chance`.
   */
  entryChance: 11 + SKILL_SCALING_WORDS,
  flags: 12 + SKILL_SCALING_WORDS,
  /** skills.ts:5434-5439 — held buff effects inherit the action's mastery
   *  upgrades. The hash lets the shader match a catalog payload to this rule. */
  upgradeKeyHash: 13 + SKILL_SCALING_WORDS,
  WORDS: SKILL_MASTERY_WORDS,
} as const;

export const SKILL_CREATED_BUFF = {
  buffId: 0,
  scaling: 1,
  chance: 1 + SKILL_SCALING_WORDS,
  flags: 2 + SKILL_SCALING_WORDS,
  condKind: 3 + SKILL_SCALING_WORDS,
  condMode: 4 + SKILL_SCALING_WORDS,
  condTarget: 5 + SKILL_SCALING_WORDS,
  condParam: 6 + SKILL_SCALING_WORDS,
  condFactor: 7 + SKILL_SCALING_WORDS,
  WORDS: SKILL_CREATED_BUFF_WORDS,
} as const;

/** Headers live in their own region: a stride of SKILL_STATE.WORDS (24) would
 *  overlap the first effect block (64*16*20 words) from skill 15 onward, so a
 *  later skill's `buffCostId = -1` could overwrite an earlier skill's effect
 *  kind. That is how Fairy's Blessing's 18% stability effect was lost. */
export const SKILL_STATE_EFFECT_OFF = BUFF_SKILL_CAPACITY * SKILL_STATE.WORDS;
export const SKILL_STATE_COST_OFF =
  SKILL_STATE_EFFECT_OFF + BUFF_SKILL_CAPACITY * SKILL_EFFECT_CAPACITY * SKILL_EFFECT_WORDS;
export const SKILL_STATE_MASTERY_OFF =
  SKILL_STATE_COST_OFF + BUFF_SKILL_CAPACITY * SKILL_COST_BRANCH_CAPACITY * SKILL_COST_BRANCH_WORDS;
export const SKILL_STATE_CREATED_OFF =
  SKILL_STATE_MASTERY_OFF + BUFF_SKILL_CAPACITY * SKILL_MASTERY_CAPACITY * SKILL_MASTERY_WORDS;
export const SKILL_STATE_WORDS =
  SKILL_STATE_CREATED_OFF + BUFF_SKILL_CAPACITY * SKILL_CREATED_BUFF_CAPACITY * SKILL_CREATED_BUFF_WORDS;
export const SKILL_STATE_BYTES = SKILL_STATE_WORDS * 4;

export interface StateSkillRecords {
  /** Flat f32 buffer; header at `index * SKILL_STATE.WORDS`, pools at the `SKILL_STATE_*_OFF` bases. */
  readonly records: Float32Array;
  readonly names: string[];
  readonly count: number;
  /** `skillOverflow:<name>` when a fixed-capacity pool would drop data. */
  readonly notes: Record<string, number>;
  readonly sizes: {
    readonly bytes: number;
    readonly headersBytes: number;
    readonly effectsBytes: number;
    readonly costsBytes: number;
    readonly masteryBytes: number;
    readonly createdBytes: number;
  };
}

function noteSkillOverflow(note: Note, name: string): void {
  note('skillOverflow', name);
}

/** Effect interceptors put the engine on the ordered-numeric path (skills.ts:882). */
function catalogHasInterceptors(catalog: BuffCatalog): boolean {
  for (let id = 0; id < catalog.count; id++) {
    const count = catalog.records[id * BUFF_DEF.WORDS + BUFF_DEF.effectCount];
    const offset = catalog.records[id * BUFF_DEF.WORDS + BUFF_DEF.effectOffset];
    for (let e = 0; e < count; e++) {
      const op = catalog.records[offset + e * BUFF_EFFECT_WORDS];
      if (op === BUFF_EFFECT_OPS.addStack || op === BUFF_EFFECT_OPS.negate ||
          op === BUFF_EFFECT_OPS.consumeBuff || op === BUFF_EFFECT_OPS.createBuff ||
          op === BUFF_EFFECT_OPS.setState || op === BUFF_EFFECT_OPS.modifyBuffGroup) {
        return true;
      }
    }
  }
  return false;
}

/** Effect condition as the shader sees it: opcode, mode, target and chance. */
function skillCondition(
  condition: unknown,
  effect: Raw,
  data: MirrorData,
  note: Note,
  ids: ReadonlyMap<string, number>,
): { readonly kind: number; readonly mode: number; readonly target: number; readonly chance: number } {
  const c = obj(condition);
  const kind = c ? BUFF_CONDITION_OPS[str(c.kind)] ?? 0 : BUFF_CONDITION_OPS.none;
  if (c && !kind) note('buffCondition', str(c.kind || '(none)'));
  const holder = obj(c?.buff);
  const name = c?.buff === 'self'
    ? 'self'
    : normalizeBuffName(holder?.name ?? obj(holder?.definition)?.name);
  const target = name === 'self' ? -2 : (ids.get(name) ?? -1);
  return {
    kind,
    mode: BUFF_CONDITION_MODE_OPS[str(c?.mode)] ?? 0,
    target,
    chance: conditionProbability(condition, effect, undefined, data, note),
  };
}

function skillEffectAmount(effect: Raw): unknown {
  if (effect.amount !== undefined) return effect.amount;
  if (effect.stacks !== undefined) return effect.stacks;
  return effect.value !== undefined ? { value: num(effect.value, 0) } : { value: 0 };
}

interface CreatedEntry {
  readonly id: number;
  readonly stacks: unknown;
  readonly chance: number;
  readonly fromMastery: boolean;
  readonly costRelevant: boolean;
  readonly name: string;
}

/**
 * Build the stateful skill family. Pure: neither the compact records nor the
 * catalog are mutated. `catalog.ids` is the single source of buff ids.
 */
export function buildStateSkillRecords(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  catalog: BuffCatalog,
  note?: Note,
): StateSkillRecords {
  const sink = makeNoteSink();
  const forward: Note = (kind, name) => {
    sink.note(kind, name);
    note?.(kind, name);
  };
  const count = Math.min(Math.max(0, Math.floor(skills.length)), BUFF_SKILL_CAPACITY);
  const records = new Float32Array(SKILL_STATE_WORDS);
  const data = toMirrorData(state, config, skills.slice(0, count), undefined);
  const ctx = catalogCtx(data, forward);
  const orderedNumeric = catalogHasInterceptors(catalog);
  const requiredPostStability = Math.max(0, num(obj(config)?.minStability, 0));

  for (let a = 0; a < count; a++) {
    const skill = obj(skills[a]) ?? {};
    const name = str(skill.name) || `skill_${a}`;
    const b = a * SKILL_STATE.WORDS;
    const effects = list(skill.effects);
    const masteries = list(skill.masteryEntries);
    const ids = catalog.ids;
    // mirror.mjs:2350-2376 folds every `upgrade` mastery into the scaling
    // payloads of this skill before any frame is built. The encoded scaling
    // has no upgradeKey slot, so bake the rule into `value` here instead of
    // leaving it for the shader. `applyUpgrades` is the same helper used by
    // the compact bridge, so additive/multiplier composition stays identical.
    ctx.upgrades = {};
    for (const raw of masteries) {
      const entry = obj(raw);
      if (entry?.kind !== 'upgrade') continue;
      const key = str(entry.upgradeKey).trim();
      const change = num(entry.change, 0);
      if (!key || change === 0) continue;
      const rule = ctx.upgrades[key] ?? { additive: 0, multiplier: 1 };
      if (entry.shouldMultiply) rule.multiplier *= 1 + change;
      else rule.additive += change;
      ctx.upgrades[key] = rule;
    }

    records[b + SKILL_STATE.successChance] = clamp01(num(skill.successChance, 1));
    // skills.ts:4400-4401 — an item without an explicit override does not
    // consume a turn, so it must not tick cooldowns, step or buff durations.
    const consumesTurn =
      skill.consumesTurn === undefined
        ? skill.actionKind !== 'item'
        : skill.consumesTurn !== false;
    records[b + SKILL_STATE.consumesTurn] = consumesTurn ? 1 : 0;
    records[b + SKILL_STATE.actionKind] = SKILL_ACTION_KIND[str(skill.actionKind)] ?? SKILL_ACTION_KIND.skill;
    records[b + SKILL_STATE.actionType] = SKILL_ACTION_TYPE[str(skill.type)] ?? 0;
    const rawBuffType = skill.buffType;
    records[b + SKILL_STATE.reserved2] =
      rawBuffType === 1 || rawBuffType === 'control' || rawBuffType === 'CONTROL'
        ? SKILL_BUFF_TYPE.control
        : rawBuffType === 2 || rawBuffType === 'intensity' || rawBuffType === 'INTENSITY'
          ? SKILL_BUFF_TYPE.intensity
          : SKILL_BUFF_TYPE.none;
    records[b + SKILL_STATE.reserved3] = num(skill.buffDuration, 0);
    records[b + SKILL_STATE.buffMultiplier] = num(skill.buffMultiplier, 0);
    records[b + SKILL_STATE.stabilityGain] = num(skill.stabilityGain, 0);
    records[b + SKILL_STATE.maxStabilityChange] = num(skill.maxStabilityChange, 0);
    records[b + SKILL_STATE.restoresMaxStabilityToFull] = skill.restoresMaxStabilityToFull === true ? 1 : 0;
    records[b + SKILL_STATE.conditionRequirement] = conditionIdOf(str(skill.conditionRequirement));
    records[b + SKILL_STATE.cooldown] = num(skill.cooldown, 0);
    const buffCost = obj(skill.buffCost);
    if (buffCost) {
      const costName = normalizeBuffName(buffCost.buffName);
      const id = ids.get(costName);
      if (id === undefined) forward('buffMissing', costName);
      records[b + SKILL_STATE.buffCostId] = id ?? -1;
      // -1 means `consumeAll`; the shader reads the live stack count instead.
      records[b + SKILL_STATE.buffCostAmount] = buffCost.consumeAll === true ? -1 : num(buffCost.amount, 0);
    } else {
      records[b + SKILL_STATE.buffCostId] = -1;
    }
    records[b + SKILL_STATE.restoresQi] = skill.restoresQi === true ? 1 : 0;
    // mirror.mjs:2499-2504 checks `skill.effects`, not mastery effects.
    const hasPoolEffect = effects.some((raw) => obj(raw)?.kind === 'pool');
    records[b + SKILL_STATE.qiRestore] =
      !hasPoolEffect && skill.restoresQi === true ? num(skill.qiRestore, 0) : 0;

    let flags = 0;
    if (effects.length > 0) flags |= SKILL_STATE_FLAGS.hasEffects;
    if (skill.isDisciplinedTouch === true) flags |= SKILL_STATE_FLAGS.isDisciplinedTouch;
    if (skill.restoresQi === true) flags |= SKILL_STATE_FLAGS.restoresQi;
    if (skill.preventsMaxStabilityDecay === true) flags |= SKILL_STATE_FLAGS.preventsMaxStabilityDecay;
    if (obj(skill.grantedBuff)?.sealedMaxStability === true) flags |= SKILL_STATE_FLAGS.sealedMaxStability;
    if (skill.noQiCost === true) flags |= SKILL_STATE_FLAGS.noQiCost;
    if (buffCost) flags |= SKILL_STATE_FLAGS.hasBuffCost;
    if (skill.conditionRequirement) flags |= SKILL_STATE_FLAGS.hasConditionRequirement;
    if (orderedNumeric) flags |= SKILL_STATE_FLAGS.orderedNumeric;
    records[b + SKILL_STATE.flags] = flags;

    // --- created buffs (technique first, then mastery), in encounter order ---
    const created: CreatedEntry[] = [];
    const addCreated = (effect: Raw, factor: number, fromMastery: boolean): void => {
      if (effect.kind !== 'createBuff') return;
      const def = obj(obj(effect.buff)?.definition) ?? obj(effect.buff) ?? {};
      const buffName = normalizeBuffName(def.name);
      const id = ids.get(buffName);
      if (id === undefined) {
        forward('buffMissing', buffName);
        return;
      }
      if (created.length >= SKILL_CREATED_BUFF_CAPACITY) {
        forward('createdBuffsFull', name);
        noteSkillOverflow(forward, name);
        return;
      }
      const condition = skillCondition(effect.condition, effect, data, forward, ids);
      const stats = obj(def.stats) ?? {};
      const costRelevant = Object.keys(stats).some((key) => key === 'poolCostFlat' ||
        key === 'qiCostReduction' || key === 'qiCostPenalty' ||
        key === 'stabilityCostReduction' || key === 'stabilityCostPenalty' ||
        key === 'poolCostPercentage' || key === 'stabilityCostPercentage');
      const entry = SKILL_STATE_CREATED_OFF +
        (a * SKILL_CREATED_BUFF_CAPACITY + created.length) * SKILL_CREATED_BUFF.WORDS;
      records[entry + SKILL_CREATED_BUFF.buffId] = id;
      const payload = encodeScaling(skillEffectAmount(effect), ctx, forward);
      for (let k = 0; k < SKILL_SCALING_WORDS; k++) {
        records[entry + SKILL_CREATED_BUFF.scaling + k] = payload[k] ?? 0;
      }
      records[entry + SKILL_CREATED_BUFF.chance] = clamp01(condition.chance * factor);
      const condRaw = obj(effect.condition);
      records[entry + SKILL_CREATED_BUFF.condKind] = condition.kind;
      records[entry + SKILL_CREATED_BUFF.condMode] = condition.mode;
      records[entry + SKILL_CREATED_BUFF.condTarget] = condition.target;
      records[entry + SKILL_CREATED_BUFF.condParam] = num(
        condRaw?.count ?? condRaw?.percentage,
        0,
      );
      records[entry + SKILL_CREATED_BUFF.condFactor] = factor;
      records[entry + SKILL_CREATED_BUFF.flags] =
        (fromMastery ? SKILL_CREATED_FLAGS.fromMastery : 0) |
        (costRelevant ? SKILL_CREATED_FLAGS.costRelevant : 0);
      created.push({
        id,
        stacks: skillEffectAmount(effect),
        chance: clamp01(condition.chance * factor),
        fromMastery,
        costRelevant,
        name: buffName,
      });
    };
    for (const raw of effects) addCreated(obj(raw) ?? {}, 1, false);

    // --- technique effects (everything phaseApplyEffects reads) ---
    let effectCount = 0;
    const addEffect = (effect: Raw): void => {
      const op = BUFF_EFFECT_OPS[str(effect.kind)] ?? 0;
      if (!op) {
        forward('effect', str(effect.kind || '(none)'));
        return;
      }
      if (effectCount >= SKILL_EFFECT_CAPACITY) {
        forward('effectsFull', name);
        noteSkillOverflow(forward, name);
        return;
      }
      const condition = skillCondition(effect.condition, effect, data, forward, ids);
      if (condition.chance <= 0) return;
      const entry = SKILL_STATE_EFFECT_OFF +
        (a * SKILL_EFFECT_CAPACITY + effectCount) * SKILL_EFFECT.WORDS;
      records[entry + SKILL_EFFECT.kind] = op;
      const payload = encodeScaling(skillEffectAmount(effect), ctx, forward);
      for (let k = 0; k < SKILL_SCALING_WORDS; k++) {
        records[entry + SKILL_EFFECT.scaling + k] = payload[k] ?? 0;
      }
      records[entry + SKILL_EFFECT.condKind] = condition.kind;
      records[entry + SKILL_EFFECT.condMode] = condition.mode;
      records[entry + SKILL_EFFECT.condTarget] = condition.target;
      records[entry + SKILL_EFFECT.chance] = condition.chance;
      records[entry + SKILL_EFFECT.critMode] = 0; // technique path
      records[entry + SKILL_EFFECT.targetBar] = SKILL_EFFECT_BARS[str(effect.kind)] ?? 0;
      records[entry + SKILL_EFFECT.flags] =
        (num(obj(effect.amount)?.value, 0) > 0 ? SKILL_EFFECT_FLAGS.zero : 0) |
        (str(effect.kind) === 'negate' ? SKILL_EFFECT_FLAGS.negate : 0) |
        (effect.value !== undefined ? SKILL_EFFECT_FLAGS.directValue : 0);
      effectCount++;
    };
    if (skill.isDisciplinedTouch === true) {
      // mirror.mjs:2482-2490 replaces only the technique's bar effects; its
      // mastery phases and createBuff bookkeeping stay untouched.
      addEffect({
        kind: 'completion',
        amount: { value: num(skill.baseCompletionGain, 0), stat: 'intensity' },
      });
      addEffect({
        kind: 'perfection',
        amount: { value: num(skill.basePerfectionGain, 0), stat: 'intensity' },
      });
    } else {
      for (const raw of effects) {
        const effect = obj(raw) ?? {};
        if (effect.kind !== 'createBuff') addEffect(effect);
      }
      // skills.ts:3331-3369 — a technique with no effects at all resolves its
      // bars through the legacy scalar path instead of `effects`:
      //   completion = baseCompletionGain, perfection = basePerfectionGain,
      //   control scaling overwrites both (3353-3362) and the fusion branch
      //   overwrites completion with baseCompletionGain * intensity (3363-3367).
      // Nothing was encoded for that path, so such a technique gained no bar.
      // `base*stat` products above 10 go through evalScaling's floor, which is
      // the engine's `safeFloor` (gameTypes.ts:820-911).
      //
      // The gate is the raw effect list, not `effectCount`: the engine takes
      // the preferred path whenever `skill.effects.length > 0` (skills.ts:3198),
      // and a createBuff-only technique (delayed_fusion) therefore never falls
      // back to `baseCompletionGain`.
      if (effects.length === 0) {
        const baseCompletion = num(skill.baseCompletionGain, 0);
        const basePerfection = num(skill.basePerfectionGain, 0);
        // skills.ts:3363-3367 wins over the control overwrite for completion.
        const fusionIntensity =
          skill.scalesWithIntensity === true && skill.type === 'fusion';
        if (skill.scalesWithControl === true) {
          if (basePerfection !== 0) {
            addEffect({
              kind: 'perfection',
              amount: { value: basePerfection, stat: 'control' },
            });
          }
          if (baseCompletion > 0 && !fusionIntensity) {
            addEffect({
              kind: 'completion',
              amount: { value: baseCompletion, stat: 'control' },
            });
          }
        }
        if (fusionIntensity && baseCompletion !== 0) {
          addEffect({
            kind: 'completion',
            amount: { value: baseCompletion, stat: 'intensity' },
          });
        }
        // The legacy buffCost stack multiplier (skills.ts:3335-3349) needs the
        // held stacks at transition time and has no encoding here; a skill that
        // combines buffCost with neither scaling flag stays understated.
      }
    }
    records[b + SKILL_STATE.effectCount] = effectCount;
    records[b + SKILL_STATE.effectOffset] =
      SKILL_STATE_EFFECT_OFF + a * SKILL_EFFECT_CAPACITY * SKILL_EFFECT_WORDS;

    // --- mastery entries (every branch phaseMasteryFrame reads) ---
    const chanceOfEntry = (condition: unknown): number => {
      const c = obj(condition);
      return c?.kind === 'chance' ? clamp01(num(c.percentage, 0) / 100) : 1;
    };
    let masteryCount = 0;
    // `chance` is the FINAL weight: entry factor x the effect's own condition.
    // Passing the entry factor separately and multiplying by `condition.chance`
    // here squared chance conditions and dropped effect-level gates entirely.
    const addMastery = (kindOp: number, payload: unknown, condValue: unknown, chance: number, extra: {
      readonly count?: number;
      readonly effectKind?: number;
      readonly effectRef?: number;
      readonly additive?: number;
      readonly multiplier?: number;
      readonly flags?: number;
      readonly entry?: number;
      readonly entryChance?: number;
      readonly upgradeKeyHash?: number;
    } = {}): void => {
      if (masteryCount >= SKILL_MASTERY_CAPACITY) {
        forward('masteryFull', name);
        noteSkillOverflow(forward, name);
        return;
      }
      const condition = skillCondition(condValue, {}, data, forward, ids);
      const entry = SKILL_STATE_MASTERY_OFF +
        (a * SKILL_MASTERY_CAPACITY + masteryCount) * SKILL_MASTERY.WORDS;
      records[entry + SKILL_MASTERY.kind] = kindOp;
      const encoded = encodeScaling(payload, ctx, forward);
      for (let k = 0; k < SKILL_SCALING_WORDS; k++) {
        records[entry + SKILL_MASTERY.scaling + k] = encoded[k] ?? 0;
      }
      records[entry + SKILL_MASTERY.condKind] = condition.kind;
      records[entry + SKILL_MASTERY.condMode] = condition.mode;
      records[entry + SKILL_MASTERY.condTarget] = condition.target;
      records[entry + SKILL_MASTERY.chance] = clamp01(chance);
      records[entry + SKILL_MASTERY.count] = extra.count ?? 1;
      records[entry + SKILL_MASTERY.effectKind] = extra.effectKind ?? 0;
      records[entry + SKILL_MASTERY.effectRef] = extra.effectRef ?? -1;
      records[entry + SKILL_MASTERY.upgradeAdditive] = extra.additive ?? 0;
      records[entry + SKILL_MASTERY.upgradeMultiplier] = extra.multiplier ?? 1;
      records[entry + SKILL_MASTERY.entry] = extra.entry ?? 0;
      records[entry + SKILL_MASTERY.entryChance] = extra.entryChance ?? 1;
      records[entry + SKILL_MASTERY.flags] = extra.flags ?? 0;
      records[entry + SKILL_MASTERY.upgradeKeyHash] = extra.upgradeKeyHash ?? 0;
      masteryCount++;
    };
    // skills.ts:4300-4341 — one engine transition per masteryEntries element,
    // so every record of that element shares the ordinal.
    for (let masteryEntryIndex = 0; masteryEntryIndex < masteries.length; masteryEntryIndex++) {
      const raw = masteries[masteryEntryIndex];
      const entry = obj(raw) ?? {};
      const kind = str(entry.kind || '(none)');
      const kindOp = SKILL_MASTERY_OPS[kind];
      if (!kindOp || kind === '(none)') {
        forward('mastery', kind);
        continue;
      }
      const factor = chanceOfEntry(entry.condition);
      if (factor <= 0) continue;
      if (kind === 'effect') {
        const listOfEffects = list(entry.effects);
        for (const rawEffect of listOfEffects) {
          const effect = obj(rawEffect) ?? {};
          if (effect.kind === 'createBuff') {
            addCreated(effect, factor, true);
            const def = obj(obj(effect.buff)?.definition) ?? obj(effect.buff) ?? {};
            const buffName = normalizeBuffName(def.name);
            const ref = created.findIndex((candidate) => candidate.name === buffName);
            const effectChance = clamp01(
              conditionProbability(effect.condition, effect, undefined, data, forward),
            );
            addMastery(kindOp, { value: 0 }, effect.condition, factor * effectChance, {
              count: listOfEffects.length,
              effectRef: ref,
              flags: SKILL_CREATED_FLAGS.fromMastery,
              entry: masteryEntryIndex,
              entryChance: factor,
            });
            continue;
          }
          const effectOp = BUFF_EFFECT_OPS[str(effect.kind)] ?? 0;
          if (!effectOp) {
            forward('masteryEffect', str(effect.kind || '(none)'));
            continue;
          }
          const effectChance = clamp01(
            conditionProbability(effect.condition, effect, undefined, data, forward),
          );
          addMastery(kindOp, skillEffectAmount(effect), effect.condition, factor * effectChance, {
            effectKind: effectOp,
            flags: num(obj(effect.amount)?.value, 0) > 0 ? SKILL_EFFECT_FLAGS.zero : 0,
            entry: masteryEntryIndex,
            entryChance: factor,
          });
        }
        if (listOfEffects.length === 0) {
          addMastery(kindOp, { value: 0 }, entry.condition, factor, {
            count: 0,
            entry: masteryEntryIndex,
            entryChance: factor,
          });
        }
        continue;
      }
      if (kind === 'upgrade') {
        const key = str(entry.upgradeKey).trim();
        if (!key) continue;
        const change = num(entry.change, 0);
        if (change === 0) continue;
        const rule = { additive: entry.shouldMultiply ? 0 : change, multiplier: entry.shouldMultiply ? 1 + change : 1 };
        addMastery(kindOp, { value: change, upgradeKey: key }, entry.condition, 1, {
          additive: rule.additive,
          multiplier: rule.multiplier,
          flags: entry.shouldMultiply ? 1 : 0,
          upgradeKeyHash: hash24(normalizeBuffName(key)),
          entry: masteryEntryIndex,
          entryChance: 1,
        });
        continue;
      }
      const percentage = num(entry.percentage, 0);
      const value = kind === 'control' || kind === 'intensity' ? percentage / 100 : percentage;
      addMastery(kindOp, { value }, entry.condition, factor, {
        entry: masteryEntryIndex,
        entryChance: factor,
      });
    }
    records[b + SKILL_STATE.masteryCount] = masteryCount;
    records[b + SKILL_STATE.masteryOffset] =
      SKILL_STATE_MASTERY_OFF + a * SKILL_MASTERY_CAPACITY * SKILL_MASTERY_WORDS;
    records[b + SKILL_STATE.createdBuffCount] = created.length;
    records[b + SKILL_STATE.createdBuffOffset] =
      SKILL_STATE_CREATED_OFF + a * SKILL_CREATED_BUFF_CAPACITY * SKILL_CREATED_BUFF_WORDS;

    // --- payment branches (expectedPaymentCosts + phasePaymentCosts) ---
    // A percentage cost buff is counted once per distinct name; already-held
    // buffs do not change the cost (mirror.mjs:2201-2233).
    const heldNames = new Set(
      [...data.buffs, ...Array.from(state.buffs.values())].map((buff) =>
        normalizeBuffName(str(obj(buff)?.name) || obj(buff)?.key),
      ),
    );
    let varies = false;
    const costCreations: Array<{ readonly ref: number; readonly probability: number }> = [];
    for (const raw of masteries) {
      const entry = obj(raw) ?? {};
      if (entry.kind !== 'effect' || !Array.isArray(entry.effects)) continue;
      const entryProb = chanceOfEntry(entry.condition);
      if (entryProb > 0 && entryProb < 1) varies = true;
      for (const rawEffect of list(entry.effects)) {
        const effect = obj(rawEffect);
        if (effect?.kind !== 'createBuff') continue;
        const def = obj(obj(effect.buff)?.definition) ?? obj(effect.buff) ?? {};
        const stats = obj(def.stats) ?? {};
        // skills.ts:780-798 someCostAffectingBuff: a mastery-created buff whose
        // only cost stat is qiCostReduction (e.g. Draw the Temper's 15%
        // "Energising Enhancement", value 10) still changes the expected
        // payment. Checking just the two legacy percentage stats dropped the
        // branch and kept the full 18 qi, 0.3 above the engine's 15.7.
        if (!Object.keys(stats).some((key) => key === 'poolCostFlat' ||
          key === 'qiCostReduction' || key === 'qiCostPenalty' ||
          key === 'stabilityCostReduction' || key === 'stabilityCostPenalty' ||
          key === 'poolCostPercentage' || key === 'stabilityCostPercentage')) continue;
        const effectProb = chanceOfEntry(effect.condition);
        if (effectProb > 0 && effectProb < 1) varies = true;
        const buffName = normalizeBuffName(def.name);
        if (heldNames.has(buffName)) {
          // Remember the suppressed creation so a shader node that no longer
          // holds the buff can re-add its cost stats (S3 carry).
          const suppressed = created.findIndex((candidate) => candidate.name === buffName);
          if (suppressed >= 0) {
            const flagWord =
              SKILL_STATE_CREATED_OFF +
              (a * SKILL_CREATED_BUFF_CAPACITY + suppressed) * SKILL_CREATED_BUFF.WORDS +
              SKILL_CREATED_BUFF.flags;
            records[flagWord] =
              num(records[flagWord], 0) | SKILL_CREATED_FLAGS.costSuppressedAtBuild;
          }
          continue;
        }
        heldNames.add(buffName);
        const ref = created.findIndex((candidate) => candidate.name === buffName);
        if (ref >= 0) costCreations.push({ ref, probability: clamp01(entryProb * effectProb) });
      }
    }
    // Same probability product as mirror.mjs:2246-2265, keyed by the set of
    // active cost-relevant creations so identical costs merge into one branch.
    let branches: Array<{ probability: number; mask: number }> = [{ probability: 1, mask: 0 }];
    for (const creation of costCreations) {
      const bit = 1 << creation.ref;
      const next: Array<{ probability: number; mask: number }> = [];
      for (const branch of branches) {
        if (creation.probability < 1) {
          next.push({ probability: branch.probability * (1 - creation.probability), mask: branch.mask });
        }
        if (creation.probability > 0) {
          next.push({ probability: branch.probability * creation.probability, mask: branch.mask | bit });
        }
      }
      branches = next;
    }
    const merged = new Map<number, number>();
    for (const branch of branches) {
      merged.set(branch.mask, (merged.get(branch.mask) ?? 0) + branch.probability);
    }
    const branchList = Array.from(merged.entries()).map(([mask, probability]) => ({ mask, probability }));
    if (branchList.length > SKILL_COST_BRANCH_CAPACITY) {
      forward('costBranchesFull', name);
      noteSkillOverflow(forward, name);
    }
    const branchCount = Math.min(branchList.length, SKILL_COST_BRANCH_CAPACITY);
    const qiPayload = encodeScaling({ value: skill.noQiCost === true ? 0 : num(skill.qiCost, 0) }, ctx, forward);
    const stabPayload = encodeScaling({ value: num(skill.stabilityCost, 0) }, ctx, forward);
    for (let c = 0; c < branchCount; c++) {
      const entry = SKILL_STATE_COST_OFF +
        (a * SKILL_COST_BRANCH_CAPACITY + c) * SKILL_COST_BRANCH.WORDS;
      const branch = branchList[c];
      records[entry + SKILL_COST_BRANCH.probability] = branch.probability;
      for (let k = 0; k < SKILL_SCALING_WORDS; k++) {
        records[entry + SKILL_COST_BRANCH.qiScaling + k] = qiPayload[k] ?? 0;
        records[entry + SKILL_COST_BRANCH.stabilityScaling + k] = stabPayload[k] ?? 0;
      }
      records[entry + SKILL_COST_BRANCH.requiredPostStability] = requiredPostStability;
      records[entry + SKILL_COST_BRANCH.masteryFlag] = varies ? 1 : 0;
      records[entry + SKILL_COST_BRANCH.creationMask] = branch.mask;
    }
    records[b + SKILL_STATE.costBranchCount] = branchCount;
    records[b + SKILL_STATE.costBranchOffset] =
      SKILL_STATE_COST_OFF + a * SKILL_COST_BRANCH_CAPACITY * SKILL_COST_BRANCH_WORDS;
  }

  return {
    records,
    names: skills.slice(0, count).map((skill) => str(skill.name)),
    count,
    notes: sink.snapshot(),
    sizes: {
      bytes: records.byteLength,
      headersBytes: BUFF_SKILL_CAPACITY * SKILL_STATE.WORDS * 4,
      effectsBytes: BUFF_SKILL_CAPACITY * SKILL_EFFECT_CAPACITY * SKILL_EFFECT_WORDS * 4,
      costsBytes: BUFF_SKILL_CAPACITY * SKILL_COST_BRANCH_CAPACITY * SKILL_COST_BRANCH_WORDS * 4,
      masteryBytes: BUFF_SKILL_CAPACITY * SKILL_MASTERY_CAPACITY * SKILL_MASTERY_WORDS * 4,
      createdBytes: BUFF_SKILL_CAPACITY * SKILL_CREATED_BUFF_CAPACITY * SKILL_CREATED_BUFF_WORDS * 4,
    },
  };
}
