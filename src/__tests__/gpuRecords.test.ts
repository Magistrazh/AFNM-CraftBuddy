import {
  EFF_OFF,
  FMOD_OFF,
  NODE_ROOT_SLOT,
  NODE_SUCCESS_BONUS,
  NS,
  OPS,
  SUCCESS_WEIGHT,
  buildGpuRecords,
} from '../optimizer/gpu/records';
import type { BuffDefinition, BuffEffect } from '../optimizer/gameTypes';
import type { OptimizerConfig, SkillDefinition } from '../optimizer/skills';
import { BuffType, createStateFromGame, type TrackedBuff } from '../optimizer/state';

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

function makeConfig(skill: SkillDefinition): OptimizerConfig {
  return {
    maxQi: 100,
    maxStability: 100,
    baseIntensity: 10,
    baseControl: 10,
    minStability: 0,
    skills: [skill],
    defaultBuffMultiplier: 1,
  };
}

function makeState(buffs: Map<string, TrackedBuff> = new Map()) {
  return createStateFromGame({
    pool: 50,
    stability: 50,
    initialMaxStability: 100,
    completion: 0,
    perfection: 0,
    buffs,
  });
}

function buffMap(definition: BuffDefinition, stacks = 1): Map<string, TrackedBuff> {
  return new Map([
    [
      definition.name.toLowerCase().replace(/\s+/g, '_'),
      { name: definition.name, stacks, definition },
    ],
  ]);
}

describe('GPU record bridge: live payload kinds', () => {
  it('evaluates eqn inside a buff stat scaling against the full variable frame', () => {
    const definition: BuffDefinition = {
      name: 'Pool Insight',
      canStack: false,
      effects: [],
      stats: {
        poolCostFlat: {
          value: 1,
          eqn: 'maxpool > 0 ? max(0, pool / maxpool) * 30 : 0',
        },
      },
    };
    const skill = makeSkill();
    const records = buildGpuRecords(
      makeState(buffMap(definition)),
      makeConfig(skill),
      [skill],
    );

    expect(records.notes).toEqual({});
    // pool 50 / maxpool 100 = 0.5, times 30 -> 15 flat pool cost.
    expect(records.nodes[36]).toBe(15);
  });

  it('evaluates string conditions deterministically', () => {
    const withCondition = (condition: string): SkillDefinition =>
      makeSkill({
        effects: [
          {
            kind: 'completion',
            amount: { value: 10 },
            condition: { kind: 'condition', condition },
          },
        ],
      });
    const cases: Array<[string, number]> = [
      ['1', OPS.completion],
      ['0', 0],
      ['pool >= maxpool / 2', OPS.completion],
      ['pool > maxpool / 2', 0],
      ['stacks >= 1', 0], // technique effects have selfStacks = 0
    ];
    for (const [condition, expectedOp] of cases) {
      const skill = withCondition(condition);
      const records = buildGpuRecords(
        makeState(),
        makeConfig(skill),
        [skill],
      );
      expect(records.notes).toEqual({});
      expect(records.skills[EFF_OFF]).toBe(expectedOp);
    }
  });

  it('folds a self addStack into the stacks seen by later buff effects', () => {
    const definitionFor = (effects: BuffEffect[]): BuffDefinition => ({
      name: 'Momentum',
      canStack: true,
      maxStacks: 10,
      stats: { critchance: { value: 10, scaling: 'stacks' } },
      effects,
    });
    const bar: BuffEffect = { kind: 'completion', amount: { value: 10 } };
    const skill = makeSkill();
    const config = makeConfig(skill);
    const build = (effects: BuffEffect[]) =>
      buildGpuRecords(makeState(buffMap(definitionFor(effects), 1)), config, [skill]);

    const base = build([bar]);
    const grown = build([{ kind: 'addStack', stacks: { value: 2 } }, bar]);
    const removed = build([{ kind: 'addStack', stacks: { value: -5 } }, bar]);
    const capped = build([{ kind: 'addStack', stacks: { value: 99 } }, bar]);

    expect(base.notes).toEqual({});
    expect(grown.notes).toEqual({});
    expect(removed.notes).toEqual({});
    expect(capped.notes).toEqual({});

    // critRatio slot of the first (and only) buff effect record.
    const ratio = (records: ReturnType<typeof build>): number => records.skills[EFF_OFF + 6];
    expect(ratio(base)).toBeCloseTo(1, 5);
    expect(ratio(grown)).toBeGreaterThan(ratio(base));
    expect(ratio(removed)).toBeLessThan(ratio(base));
    expect(ratio(capped)).toBeGreaterThan(ratio(grown));
  });

  it('keeps the base skill roll, node bonus and created-buff success bonus separate', () => {
    const buffDefinition: BuffDefinition = {
      name: 'Luck Enhancement',
      canStack: true,
      maxStacks: 5,
      effects: [],
      stats: { successChanceBonus: { value: 0.1 } },
    };
    const skill = makeSkill({
      successChance: 0.8,
      masteryEntries: [
        {
          kind: 'effect',
          effects: [
            { kind: 'createBuff', stacks: { value: 1 }, buff: buffDefinition },
          ],
        },
      ],
    });
    const state = createStateFromGame({
      pool: 50,
      stability: 50,
      initialMaxStability: 100,
      completion: 0,
      perfection: 0,
      successChanceBonus: 0.1,
    });
    const records = buildGpuRecords(state, makeConfig(skill), [skill]);

    // Head slot 18 is the BASE roll; the craft's own bonus is node slot 27.
    expect(records.skills[SUCCESS_WEIGHT]).toBeCloseTo(0.8, 6);
    expect(records.nodes[NODE_SUCCESS_BONUS]).toBeCloseTo(0.1, 6);
    // The created buff's stat becomes fmod field 11 (F.successBonus).
    expect(records.skills[FMOD_OFF]).toBe(11);
    expect(records.skills[FMOD_OFF + 1]).toBeCloseTo(0.1, 6);
    // No skipped-stat note for the key any more.
    expect(records.notes['buffStat:successChanceBonus']).toBeUndefined();
    expect(records.notes).toEqual({});
  });

  it('seeds one opening skill per node so the beam can see every root', () => {
    const skills = [
      makeSkill({ name: 'A', key: 'a' }),
      makeSkill({ name: 'B', key: 'b' }),
      makeSkill({ name: 'C', key: 'c' }),
    ];
    const state = makeState();
    const config: OptimizerConfig = { ...makeConfig(skills[0]), skills };
    const nodeCount = 7;
    const records = buildGpuRecords(state, config, skills, { nodeCount });

    for (let i = 0; i < nodeCount; i += 1) {
      expect(records.nodes[i * NS + NODE_ROOT_SLOT]).toBe(i % skills.length);
    }
  });
});
