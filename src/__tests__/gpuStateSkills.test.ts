import { existsSync, readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import {
  BUFF_EFFECT_OPS,
  BUFF_SCALING_WORDS,
  BUFF_SKILL_CAPACITY,
  CAND,
  NODE,
  NS,
  OPS,
  OUT,
  REC,
  SKILL_COST_BRANCH,
  SKILL_COST_BRANCH_CAPACITY,
  SKILL_CREATED_BUFF,
  SKILL_CREATED_BUFF_CAPACITY,
  SKILL_CREATED_BUFF_WORDS,
  SKILL_EFFECT,
  SKILL_EFFECT_CAPACITY,
  SKILL_EFFECT_WORDS,
  SKILL_MASTERY,
  SKILL_MASTERY_CAPACITY,
  SKILL_MASTERY_WORDS,
  SKILL_SCALING_WORDS,
  SKILL_STATE,
  SKILL_STATE_BYTES,
  SKILL_STATE_COST_OFF,
  SKILL_STATE_EFFECT_OFF,
  SKILL_STATE_FLAGS,
  SKILL_STATE_WORDS,
  STATS,
  buildBuffCatalog,
  buildGpuRecords,
  buildStateSkillRecords,
} from '../optimizer/gpu/records';
import { STATE_KERNEL_WGSL } from '../optimizer/gpu/kernelState';
import type { BuffDefinition } from '../optimizer/gameTypes';
import type { OptimizerConfig, SkillDefinition } from '../optimizer/skills';
import { reviveConfigSnapshot, reviveStateSnapshot } from '../modContent/replaySnapshot';
import { BuffType, createStateFromGame, type CraftingState, type TrackedBuff } from '../optimizer/state';

const FIXTURE_DIR = join(__dirname, '__fixtures__/replay-snapshots');
const HARDEST = 'S:/Temp/opencode/scenario/hardest-54.snapshot.json';
const FIXTURES = readdirSync(FIXTURE_DIR)
  .filter((file) => file.endsWith('.snapshot.json'))
  .sort();
const RUN_HARDEST = existsSync(HARDEST);
const FILES = [
  ...FIXTURES.map((file) => join(FIXTURE_DIR, file)),
  ...(RUN_HARDEST ? [HARDEST] : []),
];

type Raw = Record<string, unknown>;
const obj = (value: unknown): Raw | undefined =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Raw)
    : undefined;
const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);
const norm = (value: unknown): string =>
  String(value ?? '').trim().toLowerCase().replace(/\s+/g, '_');

interface Loaded {
  readonly state: CraftingState;
  readonly config: OptimizerConfig;
  readonly skills: SkillDefinition[];
}
function load(file: string): Loaded {
  const snapshot = obj(JSON.parse(readFileSync(file, 'utf8'))) ?? {};
  const input = obj(snapshot.input) ?? {};
  const config = reviveConfigSnapshot(obj(input.config) ?? {});
  return {
    state: reviveStateSnapshot(obj(input.state) ?? {}),
    config,
    skills: config.skills,
  };
}

/** FNV-1a over every byte of a flat buffer, used as a compact kernel guard. */
function digest(buffer: Float32Array): string {
  const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let hash = 2166136261;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i] ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function skillOverflowNotes(notes: Record<string, number>): string[] {
  return Object.keys(notes).filter((key) => key.startsWith('skillOverflow:'));
}

function makeSkill(overrides: Partial<SkillDefinition> = {}): SkillDefinition {
  return {
    name: 'Probe',
    key: 'probe',
    qiCost: 0,
    stabilityCost: 0,
    baseCompletionGain: 0,
    basePerfectionGain: 0,
    stabilityGain: 0,
    maxStabilityChange: 0,
    buffType: BuffType.NONE,
    buffDuration: 1,
    buffMultiplier: 1,
    type: 'support',
    ...overrides,
  };
}

function makeConfig(skills: SkillDefinition[]): OptimizerConfig {
  return {
    maxQi: 100,
    maxStability: 100,
    baseIntensity: 10,
    baseControl: 10,
    minStability: 0,
    skills,
    defaultBuffMultiplier: 1,
  };
}

describe('GPU stateful skills: fixture coverage', () => {
  it('builds a record for every skill in all 17 fixtures + hardest-54 without overflow', () => {
    expect(FIXTURES).toHaveLength(17);
    let total = 0;
    for (const file of FILES) {
      const { state, config, skills } = load(file);
      const catalog = buildBuffCatalog(state, config, skills);
      const records = buildStateSkillRecords(state, config, skills, catalog);
      expect({ file, notes: skillOverflowNotes(records.notes) }).toEqual({ file, notes: [] });
      expect(records.count).toBe(skills.length);
      expect(records.names).toEqual(skills.map((skill) => skill.name));
      expect(records.records.byteLength).toBe(SKILL_STATE_BYTES);
      for (let a = 0; a < records.count; a++) {
        const b = a * SKILL_STATE.WORDS;
        expect(records.records[b + SKILL_STATE.effectCount]).toBeLessThanOrEqual(SKILL_EFFECT_CAPACITY);
        expect(records.records[b + SKILL_STATE.costBranchCount]).toBeLessThanOrEqual(SKILL_COST_BRANCH_CAPACITY);
        expect(records.records[b + SKILL_STATE.masteryCount]).toBeLessThanOrEqual(SKILL_MASTERY_CAPACITY);
        expect(records.records[b + SKILL_STATE.createdBuffCount]).toBeLessThanOrEqual(SKILL_CREATED_BUFF_CAPACITY);
        expect(records.records[b + SKILL_STATE.costBranchCount]).toBeGreaterThanOrEqual(1);
        // Headers live in their own region; the effect pool starts after them.
        // Asserting the pool base catches a header stride that overlaps effects.
        expect(records.records[b + SKILL_STATE.effectOffset]).toBe(
          SKILL_STATE_EFFECT_OFF +
            a * SKILL_EFFECT_CAPACITY * SKILL_EFFECT_WORDS,
        );
        expect(records.records[b + SKILL_STATE.createdBuffOffset]).toBeGreaterThanOrEqual(0);
      }
      total += skills.length;
    }
    expect(total).toBe(RUN_HARDEST ? 329 : 275);
  });

  it('never lets a later header overwrite an earlier effect kind', () => {
    // Regression for the Fairy's Blessing case: a header stride of
    // SKILL_STATE.WORDS used to overlap the first effect block from skill 15
    // onward, so `buffCostId = -1` of a later skill overwrote the kind of an
    // earlier stability effect (41 instead of 41.18 in the stand).
    let checked = 0;
    for (const file of FILES) {
      const { state, config, skills } = load(file);
      const catalog = buildBuffCatalog(state, config, skills);
      const records = buildStateSkillRecords(state, config, skills, catalog);
      for (let a = 0; a < records.count; a++) {
        const b = a * SKILL_STATE.WORDS;
        const offset = records.records[b + SKILL_STATE.effectOffset];
        const count = records.records[b + SKILL_STATE.effectCount];
        expect(offset).toBeGreaterThanOrEqual(SKILL_STATE_EFFECT_OFF);
        expect(offset + count * SKILL_EFFECT_WORDS).toBeLessThanOrEqual(
          SKILL_STATE_COST_OFF,
        );
        for (let e = 0; e < count; e++) {
          const kind = records.records[offset + e * SKILL_EFFECT_WORDS + SKILL_EFFECT.kind];
          expect({
            file,
            skill: records.names[a],
            effect: e,
            kindIsOpcode: kind >= 0,
          }).toEqual({
            file,
            skill: records.names[a],
            effect: e,
            kindIsOpcode: true,
          });
          checked += 1;
        }
      }
    }
    expect(checked).toBeGreaterThan(200);
  });

  it('resolves catalog ids for every createBuff and buffCost', () => {
    for (const file of FILES) {
      const { state, config, skills } = load(file);
      const catalog = buildBuffCatalog(state, config, skills);
      const records = buildStateSkillRecords(state, config, skills, catalog);
      for (let a = 0; a < skills.length; a++) {
        const raw = obj(skills[a]) ?? {};
        if (raw.buffCost) {
          expect(catalog.ids.has(norm(obj(raw.buffCost)?.buffName))).toBe(true);
          expect(records.records[a * SKILL_STATE.WORDS + SKILL_STATE.buffCostId]).toBeGreaterThanOrEqual(0);
        }
        const creations: unknown[] = [
          ...list(raw.effects),
          ...list(raw.masteryEntries).flatMap((entry) => list(obj(entry)?.effects)),
        ].filter((effect) => obj(effect)?.kind === 'createBuff');
        for (const rawEffect of creations) {
          const effect = obj(rawEffect) ?? {};
          const def = obj(obj(effect.buff)?.definition) ?? obj(effect.buff) ?? {};
          const name = norm(def.name);
          expect(catalog.ids.has(name)).toBe(true);
        }
        const count = records.records[a * SKILL_STATE.WORDS + SKILL_STATE.createdBuffCount];
        const offset = records.records[a * SKILL_STATE.WORDS + SKILL_STATE.createdBuffOffset];
        for (let c = 0; c < count; c++) {
          const id = records.records[offset + c * SKILL_CREATED_BUFF.WORDS + SKILL_CREATED_BUFF.buffId];
          expect(id).toBeGreaterThanOrEqual(0);
          expect(catalog.names[id]).toBeDefined();
        }
      }
    }
  });
});

describe('GPU state kernel: skill buff cost', () => {
  // skills.ts:4297-4322 spends `buffCost` in a mastery phase before the
  // technique and the per-turn buff tick; skills.ts:4741-4749 deletes the
  // owner at zero. The kernel must remove the slot or L2 availability still
  // sees a cost that the engine has already consumed.
  it('removes the paid buff from the live slots before building the output map', () => {
    expect(STATE_KERNEL_WGSL).toContain('SKILL_BUFF_COST_ID');
    expect(STATE_KERNEL_WGSL).toContain('SKILL_BUFF_COST_AMOUNT');
    expect(STATE_KERNEL_WGSL).toContain('skills.ts:4297-4322');
    expect(STATE_KERNEL_WGSL).toContain('if (stacks <= 0.0) { continue; }');
    // The owner's live stacks are the `consumeAll` amount.
    expect(STATE_KERNEL_WGSL).toContain('buffCostRemoval = floor(inState[costEntry + 1u])');
  });
});

describe('GPU stateful skills: measured capacities', () => {
  // Per-skill maxima over the 17 fixtures + hardest-54 (329 skills).
  const observed = {
    effects: { value: 9, source: 'user-report-pattern-step-1 / Stellar Infusion' },
    costBranches: { value: 2, source: 'forge-heat-runway-step-2 / Delayed Fusion' },
    costCreations: { value: 3, source: 'user-report-fairy-recovery / Skillful Manipulation' },
    masteryEntries: { value: 4, source: 'user-report-alchemical-sequence / Brilliant Respite' },
    masteryRecords: { value: 4, source: 'user-report-alchemical-sequence / Brilliant Respite' },
    createdBuffs: { value: 5, source: 'user-report-pattern-step-1 / Focus' },
    stacksScalingKeys: { value: 2, source: 'forge-heat-runway-step-2 / Corrupted Stabilization' },
    cooldown: { value: 99999999, source: 'user-report-fairy-recovery / Golden Path: Third Peak' },
  } as const;
  const headroom = (capacity: number, max: number): boolean => capacity >= Math.ceil(max * 1.5);

  it('keeps >= 50% headroom over every observed maximum', () => {
    expect(headroom(SKILL_EFFECT_CAPACITY, observed.effects.value)).toBe(true);
    expect(headroom(SKILL_COST_BRANCH_CAPACITY, observed.costBranches.value)).toBe(true);
    expect(headroom(SKILL_COST_BRANCH_CAPACITY, observed.costCreations.value)).toBe(true);
    expect(headroom(SKILL_MASTERY_CAPACITY, observed.masteryEntries.value)).toBe(true);
    expect(headroom(SKILL_MASTERY_CAPACITY, observed.masteryRecords.value)).toBe(true);
    expect(headroom(SKILL_CREATED_BUFF_CAPACITY, observed.createdBuffs.value)).toBe(true);
    expect(headroom(SKILL_SCALING_WORDS, observed.stacksScalingKeys.value)).toBe(true);
    expect(SKILL_SCALING_WORDS).toBe(BUFF_SCALING_WORDS);
    // Cooldown is a numeric header field, not a pool; f32 must hold the max.
    expect(Number.isFinite(Math.fround(observed.cooldown.value))).toBe(true);
    expect(BUFF_SKILL_CAPACITY).toBeGreaterThanOrEqual(54);
  });

  it('re-derives the maxima from the fixture payloads themselves', () => {
    let effectsMax = 0;
    let masteryMax = 0;
    let createdMax = 0;
    let cooldownMax = 0;
    let stacksMax = 0;
    let branchesMax = 0;
    for (const file of FILES) {
      const { skills } = load(file);
      for (const skill of skills) {
        const raw = obj(skill) ?? {};
        const effects = list(raw.effects);
        const masteries = list(raw.masteryEntries);
        effectsMax = Math.max(effectsMax, effects.length);
        masteryMax = Math.max(masteryMax, masteries.length);
        cooldownMax = Math.max(cooldownMax, Number(raw.cooldown ?? 0));
        let created = 0;
        let varies = 0;
        const inspect = (effect: Raw, factor: number): void => {
          if (effect.kind !== 'createBuff') return;
          created++;
          stacksMax = Math.max(stacksMax, Object.keys(obj(effect.stacks) ?? {}).length);
          const def = obj(obj(effect.buff)?.definition) ?? obj(effect.buff) ?? {};
          const stats = obj(def.stats) ?? {};
          if (Object.keys(stats).some((key) => key === 'poolCostPercentage' || key === 'stabilityCostPercentage')) {
            const c = obj(effect.condition);
            const p = c?.kind === 'chance' ? Number(c.percentage ?? 0) / 100 : 1;
            if (p > 0 && p < 1 && factor * p > 0 && factor * p < 1) varies++;
          }
        };
        effects.forEach((rawEffect) => inspect(obj(rawEffect) ?? {}, 1));
        for (const mastery of masteries) {
          const m = obj(mastery) ?? {};
          const c = obj(m.condition);
          const factor = c?.kind === 'chance' ? Number(c.percentage ?? 0) / 100 : 1;
          list(m.effects).forEach((rawEffect) => inspect(obj(rawEffect) ?? {}, factor));
        }
        createdMax = Math.max(createdMax, created);
        branchesMax = Math.max(branchesMax, 2 ** varies);
      }
    }
    expect(effectsMax).toBe(observed.effects.value);
    expect(masteryMax).toBe(observed.masteryEntries.value);
    expect(createdMax).toBe(observed.createdBuffs.value);
    expect(cooldownMax).toBe(observed.cooldown.value);
    expect(stacksMax).toBe(observed.stacksScalingKeys.value);
    expect(branchesMax).toBe(observed.costBranches.value);
  });

  it('keeps the layout dense and aligned', () => {
    expect(SKILL_STATE_WORDS % 4).toBe(0);
    expect(SKILL_STATE_BYTES).toBe(SKILL_STATE_WORDS * 4);
    expect(SKILL_STATE_EFFECT_OFF + BUFF_SKILL_CAPACITY * SKILL_EFFECT_CAPACITY * SKILL_EFFECT_WORDS)
      .toBe(SKILL_STATE_COST_OFF);
    expect(SKILL_COST_BRANCH.requiredPostStability).toBe(1 + SKILL_SCALING_WORDS * 2);
    expect(SKILL_EFFECT.flags).toBe(SKILL_EFFECT_WORDS - 1);
    expect(SKILL_MASTERY.flags).toBe(SKILL_MASTERY_WORDS - 2);
    expect(SKILL_MASTERY.upgradeKeyHash).toBe(SKILL_MASTERY_WORDS - 1);
    // Condition fields follow flags so a created buff's buff-condition gate
    // can be re-evaluated against each node (S3 carry).
    expect(SKILL_CREATED_BUFF.flags).toBe(14);
    expect(SKILL_CREATED_BUFF.condFactor).toBe(SKILL_CREATED_BUFF_WORDS - 1);
    expect(SKILL_STATE_FLAGS.hasEffects).toBe(1);
    expect(BUFF_EFFECT_OPS.createBuff).toBeGreaterThan(0);
  });
});

describe('GPU stateful skills: loud overflow', () => {
  const definition: BuffDefinition = {
    name: 'Probe Buff',
    canStack: true,
    maxStacks: 10,
    effects: [],
    stats: { completion: { value: 1 } },
  };

  it('notes skillOverflow when technique effects exceed capacity', () => {
    const skill = makeSkill({
      name: 'Effect Overflow',
      effects: Array.from({ length: SKILL_EFFECT_CAPACITY + 2 }, () => ({
        kind: 'completion' as const,
        amount: { value: 1 },
      })),
    });
    const config = makeConfig([skill]);
    const state = createStateFromGame({
      pool: 100, stability: 100, initialMaxStability: 100, completion: 0, perfection: 0,
      buffs: new Map<string, TrackedBuff>([[norm(definition.name), { name: definition.name, stacks: 1, definition }]]),
    });
    const catalog = buildBuffCatalog(state, config, [skill]);
    const records = buildStateSkillRecords(state, config, [skill], catalog);
    expect(skillOverflowNotes(records.notes)).toEqual(['skillOverflow:Effect Overflow']);
    expect(records.records[SKILL_STATE.effectCount]).toBe(SKILL_EFFECT_CAPACITY);
  });

  it('notes skillOverflow when created buffs exceed capacity', () => {
    const skill = makeSkill({
      name: 'Created Overflow',
      effects: Array.from({ length: SKILL_CREATED_BUFF_CAPACITY + 2 }, () => ({
        kind: 'createBuff' as const,
        stacks: { value: 1 },
        buff: definition,
      })),
    });
    const config = makeConfig([skill]);
    const state = createStateFromGame({
      pool: 100, stability: 100, initialMaxStability: 100, completion: 0, perfection: 0,
      buffs: new Map<string, TrackedBuff>([[norm(definition.name), { name: definition.name, stacks: 1, definition }]]),
    });
    const catalog = buildBuffCatalog(state, config, [skill]);
    const records = buildStateSkillRecords(state, config, [skill], catalog);
    expect(skillOverflowNotes(records.notes)).toEqual(['skillOverflow:Created Overflow']);
    expect(records.records[SKILL_STATE.createdBuffCount]).toBe(SKILL_CREATED_BUFF_CAPACITY);
  });
});

describe('GPU compact kernel guard: byte-identical output', () => {
  // Fingerprints captured before the stateful record family was added, over
  // all 17 fixtures + hardest-54 with nodeCount=2, beamWidth=3, topK=4.
  const expected: Readonly<Record<string, { readonly skills: string; readonly nodes: string }>> = {
    'forge-heat-runway-step-2.snapshot.json': { skills: 'e357ddba', nodes: '6acf7ef8' },
    'forge-heat-runway-step-3.snapshot.json': { skills: 'e357ddba', nodes: 'c67f2af0' },
    'low-stability-regression.snapshot.json': { skills: 'e357ddba', nodes: '80af5e70' },
    'low-stability-step-before.snapshot.json': { skills: 'db8e343b', nodes: '943964ac' },
    'premature-finish-proc-floor.snapshot.json': { skills: 'd9ebb42a', nodes: '616468a0' },
    'skyfall-bow-heat-regression.snapshot.json': { skills: '619a3f8d', nodes: '73336b74' },
    'user-report-alchemical-sequence.snapshot.json': { skills: '0be716b1', nodes: 'bbe8a190' },
    'user-report-fairy-recovery.snapshot.json': { skills: '144a3532', nodes: 'ac1a4f44' },
    'user-report-focused-fusion-prereq.snapshot.json': { skills: 'c56638a4', nodes: 'deb8fd6c' },
    'user-report-forge-frozen-opening.snapshot.json': { skills: '1479e470', nodes: 'f5176138' },
    'user-report-live-workshop-step-1.snapshot.json': { skills: '0be716b1', nodes: 'bbe8a190' },
    'user-report-live-workshop-step-2.snapshot.json': { skills: '89d6db25', nodes: 'ee4c4448' },
    'user-report-overcraft-endgame.snapshot.json': { skills: '542165b2', nodes: '7cc29280' },
    'user-report-pattern-step-1.snapshot.json': { skills: 'f7e93d16', nodes: 'cdc74b14' },
    'user-report-pattern-step-2.snapshot.json': { skills: '0be716b1', nodes: 'bbe8a190' },
    'user-report-premature-finish-runway.snapshot.json': { skills: '80373164', nodes: '1fa70ff0' },
    'user-report-resonance-regression.snapshot.json': { skills: 'f7e93d16', nodes: 'cdc74b14' },
    'hardest-54.snapshot.json': { skills: 'cca0ef9a', nodes: '96a26418' },
  };

  it('still emits the same bytes and layout constants', () => {
    expect({ REC, NODE, NS, OUT, CAND }).toEqual({ REC: 488, NODE: 38, NS: 40, OUT: 5, CAND: 8 });
    expect(OPS).toEqual({ perfection: 1, completion: 2, stability: 3, maxStability: 4, pool: 5, createBuff: 6 });
    expect(STATS).toEqual({
      flat: 0, control: 1, intensity: 2, maxpool: 3, maxstability: 4,
      completion: 5, perfection: 6, critchance: 7, critmultiplier: 8, stacks: 9,
    });
    for (const file of FILES) {
      if (!RUN_HARDEST && file === HARDEST) continue;
      const { state, config, skills } = load(file);
      const records = buildGpuRecords(state, config, skills, { nodeCount: 2, beamWidth: 3, topK: 4 });
      const name = file.split(/[\\/]/).pop() ?? file;
      expect({ name, skills: digest(records.skills), nodes: digest(records.nodes) }).toEqual({
        name, ...expected[name],
      });
    }
  });
});
