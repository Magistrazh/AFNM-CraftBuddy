import { existsSync, readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import {
  BUFF_CATALOG_BYTES,
  BUFF_CONDITION_OPS,
  BUFF_DEFS_CAPACITY,
  BUFF_DEF,
  BUFF_EFFECTS_PER_DEF,
  BUFF_INITIAL_STATE_CAPACITY,
  BUFF_SLOT_CAPACITY,
  BUFF_SLOT_WORDS,
  BUFF_SKILL_CAPACITY,
  BUFF_STATS_CAPACITY,
  BUFF_TRIGGER_SLOTS,
  BUFF_VAR_CAPACITY,
  MAX_OUTCOMES,
  NS_STATE,
  NS_STATE_BYTES,
  buildBuffCatalog,
  buildStateNodes,
} from '../optimizer/gpu/records';
import type { BuffDefinition } from '../optimizer/gameTypes';
import type { OptimizerConfig, SkillDefinition } from '../optimizer/skills';
import { reviveConfigSnapshot, reviveStateSnapshot } from '../modContent/replaySnapshot';
import { createStateFromGame, type CraftingState, type TrackedBuff } from '../optimizer/state';

const FIXTURE_DIR = join(__dirname, '__fixtures__/replay-snapshots');
const HARDEST = 'S:/Temp/opencode/scenario/hardest-54.snapshot.json';
const FIXTURES = readdirSync(FIXTURE_DIR)
  .filter((file) => file.endsWith('.snapshot.json'))
  .sort();
const RUN_HARDEST = existsSync(HARDEST);

type Raw = Record<string, unknown>;

function obj(value: unknown): Raw | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Raw)
    : undefined;
}
function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
function norm(value: unknown): string {
  return String(value ?? '').trim().toLowerCase().replace(/\s+/g, '_');
}
function load(file: string): { state: CraftingState; config: OptimizerConfig; skills: SkillDefinition[] } {
  const snapshot = obj(JSON.parse(readFileSync(file, 'utf8')));
  const input = obj(snapshot?.input) ?? {};
  const config = reviveConfigSnapshot(obj(input.config) ?? {});
  return {
    state: reviveStateSnapshot(obj(input.state) ?? {}),
    config,
    skills: config.skills,
  };
}

/** Independent walk of every definition the catalog must cover. */
function reachableDefinitions(state: CraftingState, skills: SkillDefinition[]): Map<string, Raw> {
  const defs = new Map<string, Raw>();
  const visited = new Set<object>();
  const add = (raw: unknown): void => {
    const holder = obj(raw);
    if (!holder || visited.has(holder)) return;
    visited.add(holder);
    const def = obj(holder.definition) ?? holder;
    const name = norm(def.name);
    if (!name) return;
    const previous = defs.get(name);
    if (!previous || Object.keys(def.stats as object ?? {}).length > Object.keys(previous.stats as object ?? {}).length) {
      defs.set(name, def);
    }
    const lists: unknown[] = [
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
  state.buffs.forEach((tracked, key) => add(tracked.definition ?? { ...tracked, name: tracked.name || key }));
  for (const skill of skills) {
    const raw = obj(skill) ?? {};
    if (raw.grantedBuff) add(raw.grantedBuff);
    for (const effect of list(raw.effects)) {
      const entry = obj(effect);
      if (entry?.buff) add(entry.buff);
    }
    for (const mastery of list(raw.masteryEntries)) {
      for (const effect of list(obj(mastery)?.effects)) {
        const entry = obj(effect);
        if (entry?.buff) add(entry.buff);
      }
    }
  }
  return defs;
}

function overflowNotes(notes: Record<string, number>): string[] {
  return Object.keys(notes).filter((key) => key.startsWith('buffOverflow:'));
}

describe('GPU stateful catalog: fixture coverage', () => {
  it('collects every reachable buff from all 17 fixtures without overflow', () => {
    expect(FIXTURES).toHaveLength(17);
    for (const file of FIXTURES) {
      const { state, config, skills } = load(join(FIXTURE_DIR, file));
      const catalog = buildBuffCatalog(state, config, skills);
      const expected = reachableDefinitions(state, skills);
      expected.forEach((_def, name) => {
        expect(catalog.ids.has(name)).toBe(true);
      });
      state.buffs.forEach((tracked, key) => {
        expect(catalog.ids.has(norm(tracked.name || key))).toBe(true);
      });
      expect({ file, notes: overflowNotes(catalog.notes) }).toEqual({ file, notes: [] });
      expect(catalog.records.byteLength).toBe(BUFF_CATALOG_BYTES);
      expect(catalog.count).toBe(expected.size);
    }
  });

  (RUN_HARDEST ? it : it.skip)('collects every reachable buff from hardest-54 without overflow', () => {
    const { state, config, skills } = load(HARDEST);
    const catalog = buildBuffCatalog(state, config, skills);
    reachableDefinitions(state, skills).forEach((_def, name) => {
      expect(catalog.ids.has(name)).toBe(true);
    });
    expect(overflowNotes(catalog.notes)).toEqual([]);
    expect(skills.length).toBeLessThanOrEqual(BUFF_SKILL_CAPACITY);
  });
});

describe('GPU stateful catalog: measured capacities', () => {
  // Observed over the 17 fixtures + hardest-54 (and their replay traces).
  const observed = {
    activeBuffs: { value: 23, source: 'user-report-overcraft-endgame.trace.json' },
    definitions: { value: 99, source: '17 fixtures + hardest-54' },
    stats: { value: 3, source: 'weak_cauldron_construct' },
    effectsPerList: { value: 4, source: 'delayed_fusion.onAction' },
    effectsTotal: { value: 6, source: 'soul_fragment_(iv)' },
    triggers: { value: 4, source: 'onAction (max across five kinds)' },
    eventBlocks: { value: 1, source: 'true_bifang_flame_(consuming)' },
    initialState: { value: 4, source: 'stabilizing_circuit' },
    additionalVariables: { value: 0, source: 'no definition carries any' },
    skills: { value: 54, source: 'hardest-54' },
  } as const;
  const headroom = (capacity: number, max: number): boolean => max === 0 ? capacity >= 4 : capacity >= Math.ceil(max * 1.5);

  it('keeps >= 50% headroom over every observed maximum', () => {
    expect(headroom(BUFF_SLOT_CAPACITY, observed.activeBuffs.value)).toBe(true);
    expect(headroom(BUFF_DEFS_CAPACITY, observed.definitions.value)).toBe(true);
    expect(headroom(BUFF_STATS_CAPACITY, observed.stats.value)).toBe(true);
    expect(headroom(BUFF_EFFECTS_PER_DEF, observed.effectsTotal.value)).toBe(true);
    expect(headroom(BUFF_EFFECTS_PER_DEF, observed.effectsPerList.value)).toBe(true);
    expect(headroom(BUFF_INITIAL_STATE_CAPACITY, observed.initialState.value)).toBe(true);
    expect(headroom(BUFF_VAR_CAPACITY, observed.additionalVariables.value)).toBe(true);
    expect(headroom(BUFF_TRIGGER_SLOTS - 5, observed.eventBlocks.value)).toBe(true);
    expect(BUFF_SKILL_CAPACITY).toBeGreaterThanOrEqual(observed.skills.value);
  });

  it('derives the maxima from the fixture payloads themselves', () => {
    let activeMax = 0;
    let statMax = 0;
    let listMax = 0;
    let totalMax = 0;
    let initialMax = 0;
    let varsMax = 0;
    let skillMax = 0;
    const files = [...FIXTURES.map((file) => join(FIXTURE_DIR, file)), ...(RUN_HARDEST ? [HARDEST] : [])];
    for (const file of files) {
      const snapshot = obj(JSON.parse(readFileSync(file, 'utf8'))) ?? {};
      const input = obj(snapshot.input) ?? {};
      const stateBuffs = obj(input.state)?.buffs;
      activeMax = Math.max(activeMax, Object.keys(obj(stateBuffs) ?? {}).length);
      skillMax = Math.max(skillMax, list(obj(input.config)?.skills).length);
      const { state, skills } = load(file);
      reachableDefinitions(state, skills).forEach((def) => {
        statMax = Math.max(statMax, Object.keys(obj(def.stats) ?? {}).length);
        initialMax = Math.max(initialMax, Object.keys(obj(def.initialState) ?? {}).length);
        varsMax = Math.max(varsMax, Object.keys(obj(def.additionalVariables) ?? {}).length);
        const lists: number[] = [
          def.effects,
          def.onFusion,
          def.onRefine,
          def.onStabilize,
          def.onSupport,
          ...list(def.triggeredEffects).map((block) => obj(block)?.effects),
        ].map((effects) => list(effects).length);
        listMax = Math.max(listMax, ...lists);
        totalMax = Math.max(totalMax, lists.reduce((sum, count) => sum + count, 0));
      });
    }
    expect(activeMax).toBeLessThanOrEqual(BUFF_SLOT_CAPACITY / 1.5);
    expect(statMax).toBeLessThanOrEqual(BUFF_STATS_CAPACITY / 1.5);
    expect(listMax).toBeLessThanOrEqual(BUFF_EFFECTS_PER_DEF / 1.5);
    expect(totalMax).toBeLessThanOrEqual(BUFF_EFFECTS_PER_DEF / 1.5);
    expect(initialMax).toBeLessThanOrEqual(BUFF_INITIAL_STATE_CAPACITY / 1.5);
    expect(varsMax).toBeLessThanOrEqual(BUFF_VAR_CAPACITY);
    expect(skillMax).toBeLessThanOrEqual(BUFF_SKILL_CAPACITY);
  });
});

describe('GPU stateful nodes: round-trip', () => {
  it('reads bars, buff stacks, internalState and cooldowns back', () => {
    const { state, config, skills } = load(join(FIXTURE_DIR, 'user-report-overcraft-endgame.snapshot.json'));
    const nodes = buildStateNodes(state, config, skills, 2);
    expect(nodes.notes).toEqual({});
    expect(nodes.nodes.byteLength).toBe(2 * NS_STATE_BYTES);
    expect(overflowNotes(nodes.notes)).toEqual([]);

    for (let node = 0; node < nodes.nodeCount; node++) {
      const b = node * NS_STATE.WORDS;
      expect(nodes.nodes[b + NS_STATE.completion]).toBe(state.completion);
      expect(nodes.nodes[b + NS_STATE.perfection]).toBe(state.perfection);
      expect(nodes.nodes[b + NS_STATE.pool]).toBe(state.qi);
      expect(nodes.nodes[b + NS_STATE.stability]).toBe(state.stability);
      expect(nodes.nodes[b + NS_STATE.maxStability]).toBe(state.maxStability);
      expect(nodes.nodes[b + NS_STATE.stabilityPenalty]).toBe(state.stabilityPenalty);
      expect(nodes.nodes[b + NS_STATE.step]).toBe(state.step);
      expect(nodes.nodes[b + NS_STATE.alive]).toBe(1);
      expect(nodes.nodes[b + NS_STATE.control]).toBe(config.baseControl);
      expect(nodes.nodes[b + NS_STATE.intensity]).toBe(config.baseIntensity);
      expect(nodes.nodes[b + NS_STATE.critChance]).toBe(state.critChance);
      expect(nodes.nodes[b + NS_STATE.critMultiplier]).toBe(state.critMultiplier);
      expect(nodes.nodes[b + NS_STATE.overcrit]).toBe(state.overcrit);
      expect(nodes.nodes[b + NS_STATE.overcritCapBonus]).toBe(state.overcritCapBonus);
      expect(nodes.nodes[b + NS_STATE.poolCostFlat]).toBe(state.poolCostFlat);
      expect(nodes.nodes[b + NS_STATE.poolCostPercentage]).toBe(state.poolCostPercentage);
      expect(nodes.nodes[b + NS_STATE.stabilityCostPercentage]).toBe(state.stabilityCostPercentage);
      expect(nodes.nodes[b + NS_STATE.qiCostReduction]).toBe(state.qiCostReduction);
      expect(nodes.nodes[b + NS_STATE.stabilityCostReduction]).toBe(state.stabilityCostReduction);
      expect(nodes.nodes[b + NS_STATE.stabilityCostPenalty]).toBe(state.stabilityCostPenalty);
      expect(nodes.nodes[b + NS_STATE.successChanceBonus]).toBe(state.successChanceBonus);

      const slots = new Map<number, number>();
      for (let slot = 0; slot < BUFF_SLOT_CAPACITY; slot++) {
        const id = nodes.nodes[b + NS_STATE.buffs + slot * BUFF_SLOT_WORDS];
        if (id >= 0) slots.set(id, b + NS_STATE.buffs + slot * BUFF_SLOT_WORDS);
      }
      expect(slots.size).toBe(state.buffs.size);
      state.buffs.forEach((tracked, key) => {
        const id = nodes.buffIds.indexOf(norm(tracked.name || key));
        expect(id).toBeGreaterThanOrEqual(0);
        const entry = slots.get(id);
        expect(entry).toBeDefined();
        expect(nodes.nodes[(entry ?? 0) + 1]).toBe(tracked.stacks);
      });
      for (let i = 0; i < skills.length; i++) {
        const skill = skills[i];
        const want = state.cooldowns.get(skill.key) ?? state.cooldowns.get(skill.name) ?? 0;
        expect(nodes.nodes[b + NS_STATE.cooldowns + i]).toBe(want);
      }
    }
    for (let word = 0; word < NS_STATE.WORDS; word++) {
      expect(nodes.nodes[word]).toBe(nodes.nodes[NS_STATE.WORDS + word]);
    }
  });

  it('exposes blaze-style internalState through the buff slot', () => {
    const definition: BuffDefinition = {
      name: 'True Bifang Flame (Consuming)',
      canStack: true,
      maxStacks: 1,
      effects: [],
      initialState: { blaze: '0' },
      stats: { control: { value: 0.03, stat: 'control', scaling: 'blaze' } },
    };
    const buffs = new Map<string, TrackedBuff>([
      [norm(definition.name), { name: definition.name, stacks: 1, definition, internalState: { blaze: 7 } }],
    ]);
    const state = createStateFromGame({ pool: 100, stability: 100, initialMaxStability: 100, completion: 0, perfection: 0, buffs });
    const config: OptimizerConfig = {
      maxQi: 100,
      maxStability: 100,
      baseControl: 10,
      baseIntensity: 10,
      minStability: 0,
      skills: [],
      defaultBuffMultiplier: 1,
    };
    const nodes = buildStateNodes(state, config, [], 1);
    const slot = nodes.nodes[NS_STATE.buffs];
    expect(slot).toBe(0);
    expect(nodes.nodes[NS_STATE.buffs + 3]).toBe(7);
  });
});

describe('GPU stateful catalog: loud overflow', () => {
  it('notes buffOverflow when a synthetic buff exceeds the slot capacity', () => {
    const definition: BuffDefinition = {
      name: 'Overflow Probe',
      canStack: true,
      effects: [],
    };
    const buffs = new Map<string, TrackedBuff>();
    for (let i = 0; i < BUFF_SLOT_CAPACITY + 2; i++) {
      const name = `Overflow Probe ${i}`;
      buffs.set(norm(name), { name, stacks: 1, definition: { ...definition, name } });
    }
    const state = createStateFromGame({ pool: 100, stability: 100, initialMaxStability: 100, completion: 0, perfection: 0, buffs });
    const config: OptimizerConfig = {
      maxQi: 100,
      maxStability: 100,
      baseControl: 10,
      baseIntensity: 10,
      minStability: 0,
      skills: [],
      defaultBuffMultiplier: 1,
    };
    const catalog = buildBuffCatalog(state, config, []);
    expect(overflowNotes(catalog.notes)).toEqual([]);
    const nodes = buildStateNodes(state, config, [], 1);
    expect(overflowNotes(nodes.notes)).toEqual([
      `buffOverflow:${norm('Overflow Probe 36')}`,
      `buffOverflow:${norm('Overflow Probe 37')}`,
    ]);
    let filled = 0;
    for (let slot = 0; slot < BUFF_SLOT_CAPACITY; slot++) {
      if (nodes.nodes[NS_STATE.buffs + slot * BUFF_SLOT_WORDS] >= 0) filled++;
    }
    expect(filled).toBe(BUFF_SLOT_CAPACITY);
  });

  it('notes buffOverflow when one definition exceeds a fixed pool', () => {
    const definition: BuffDefinition = {
      name: 'Stats Overflow Probe',
      canStack: false,
      effects: [],
      stats: {},
    };
    const statKeys = ['control', 'intensity', 'maxpool', 'completion', 'perfection', 'critchance', 'critmultiplier'];
    for (let i = 0; i < BUFF_STATS_CAPACITY + 1; i++) {
      definition.stats![statKeys[i]] = { value: 1 };
    }
    const buffs = new Map<string, TrackedBuff>([
      [norm(definition.name), { name: definition.name, stacks: 1, definition }],
    ]);
    const state = createStateFromGame({ pool: 100, stability: 100, initialMaxStability: 100, completion: 0, perfection: 0, buffs });
    const config: OptimizerConfig = {
      maxQi: 100,
      maxStability: 100,
      baseControl: 10,
      baseIntensity: 10,
      minStability: 0,
      skills: [],
      defaultBuffMultiplier: 1,
    };
    const catalog = buildBuffCatalog(state, config, []);
    expect(catalog.notes[`buffOverflow:${norm(definition.name)}`]).toBeGreaterThan(0);
    const id = catalog.ids.get(norm(definition.name));
    expect(id).toBe(0);
    expect(catalog.records[(id ?? 0) * BUFF_DEF.WORDS + BUFF_DEF.statCount]).toBe(BUFF_STATS_CAPACITY);
  });
});

describe('GPU stateful layout: invariants', () => {
  it('keeps the compact kernel layout and new opcode tables distinct', () => {
    expect(BUFF_CATALOG_BYTES % 4).toBe(0);
    expect(NS_STATE_BYTES).toBe(NS_STATE.WORDS * 4);
    // Blocks must tile the node without overlap. Cooldowns used to be last;
    // the EV round appended six score/condition inputs after them, so assert
    // the tiling and the tail contents instead of a "cooldowns last" order.
    expect(NS_STATE.buffs).toBeGreaterThan(NS_STATE.harmonyRecommendMask);
    expect(
      NS_STATE.buffs + BUFF_SLOT_CAPACITY * BUFF_SLOT_WORDS,
    ).toBeLessThanOrEqual(NS_STATE.cooldowns);
    expect(NS_STATE.cooldowns + BUFF_SKILL_CAPACITY).toBe(
      NS_STATE.conditionGenerationHarmony,
    );
    const tail = [
      NS_STATE.conditionGenerationHarmony,
      NS_STATE.toxicity,
      NS_STATE.maxToxicity,
      NS_STATE.controlBuffMultiplier,
      NS_STATE.intensityBuffMultiplier,
      NS_STATE.defaultBuffMultiplier,
      NS_STATE.floorStability,
      NS_STATE.floorMaxStability,
      NS_STATE.survivalProbability,
      NS_STATE.survivingStability,
      NS_STATE.survivingPenalty,
      NS_STATE.condStability,
      NS_STATE.condPenalty,
      NS_STATE.outcomeOverflow,
      NS_STATE.projectionDebug,
      NS_STATE.projectionCount,
      NS_STATE.survivalCount,
      NS_STATE.survivalTwin,
    ];
    expect(new Set(tail).size).toBe(tail.length);
    expect(Math.min(...tail)).toBeGreaterThanOrEqual(
      NS_STATE.cooldowns + BUFF_SKILL_CAPACITY,
    );
    // The debug block is the last variable-sized region; the three counters
    // follow it and the node ends immediately after them.
    expect(NS_STATE.projectionDebug + MAX_OUTCOMES * 3).toBe(
      NS_STATE.projectionCount,
    );
    expect(NS_STATE.survivalTwin + 1).toBe(NS_STATE.WORDS);
    expect(BUFF_CONDITION_OPS.chance).toBe(1);
    expect(BUFF_DEF.WORDS).toBe(24);
    expect(BUFF_TRIGGER_SLOTS).toBeGreaterThanOrEqual(5);
    expect(BUFF_INITIAL_STATE_CAPACITY).toBeGreaterThan(0);
    expect(BUFF_VAR_CAPACITY).toBeGreaterThan(0);
  });
});
