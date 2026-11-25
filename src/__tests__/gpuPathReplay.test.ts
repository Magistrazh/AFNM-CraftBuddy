import { replayGpuPaths } from '../modContent/gpuPathSearch';
import { BuffType, CraftingState } from '../optimizer/state';
import type {
  OptimizerConfig,
  SkillDefinition,
} from '../optimizer/skills';
import { __testing, findBestSkill } from '../optimizer/search';

/**
 * Deterministic fixture: no RNG, no cooldowns, no condition effects. The
 * engine's rotation projection therefore uses the same `applySkill` states as
 * the replay, so all projected bars must match exactly.
 */
function makeFixture(): {
  state: CraftingState;
  config: OptimizerConfig;
} {
  const base: SkillDefinition = {
    name: 'Base',
    key: 'base',
    qiCost: 0,
    stabilityCost: 0,
    baseCompletionGain: 0,
    basePerfectionGain: 0,
    stabilityGain: 0,
    maxStabilityChange: 0,
    buffType: BuffType.NONE,
    buffDuration: 0,
    buffMultiplier: 1,
    type: 'support',
  };
  const skills: SkillDefinition[] = [
    {
      ...base,
      name: 'Grow',
      key: 'grow',
      baseCompletionGain: 20,
      type: 'fusion',
    },
    {
      ...base,
      name: 'Polish',
      key: 'polish',
      basePerfectionGain: 20,
      type: 'refine',
    },
    {
      ...base,
      name: 'Steady',
      key: 'steady',
      stabilityGain: 5,
      type: 'stabilize',
    },
  ];
  const config: OptimizerConfig = {
    maxQi: 100,
    maxStability: 60,
    baseIntensity: 10,
    baseControl: 10,
    minStability: 0,
    defaultBuffMultiplier: 1.4,
    maxCompletion: 1000,
    maxPerfection: 1000,
    isSublimeCraft: false,
    skills,
  };
  const state = new CraftingState({
    qi: 100,
    stability: 60,
    initialMaxStability: 60,
    completion: 0,
    perfection: 0,
  });
  return { state, config };
}

const targets = { completion: 100, perfection: 100 };

describe('GPU path replay', () => {
  it('replays the engine rotation and matches its expected final state', () => {
    const { state, config } = makeFixture();
    const result = findBestSkill(
      state,
      config,
      targets.completion,
      targets.perfection,
      false,
      3,
      'neutral',
      [],
      { timeBudgetMs: 600_000, maxNodes: 200_000 },
    );
    const rotation = result.optimalRotation ?? [];
    const expected = result.expectedFinalState;
    expect(rotation.length).toBeGreaterThan(0);
    expect(expected).toBeDefined();

    const replays = replayGpuPaths(
      state,
      config,
      'neutral',
      targets,
      [{ skills: rotation }],
    );
    expect(replays).toHaveLength(1);
    const final = replays[0].endState;
    // Every projected bar comes from the same applySkill chain; the engine's
    // survival floor is a no-op for these deterministic techniques, so there
    // is no field that needs a different tolerance or an exception.
    expect(final.completion).toBeCloseTo(expected!.completion, 6);
    expect(final.perfection).toBeCloseTo(expected!.perfection, 6);
    expect(final.stability).toBeCloseTo(expected!.stability, 6);
    expect(final.maxStability).toBeCloseTo(expected!.maxStability, 6);
    expect(final.qi).toBeCloseTo(expected!.qi, 6);
  });

  it('drops a path containing an inapplicable technique instead of scoring it', () => {
    const { state, config } = makeFixture();
    config.skills.push({
      ...config.skills[0],
      name: 'Impossible',
      key: 'impossible',
      qiCost: 1000,
    });
    const valid = { skills: ['Grow', 'Polish'] };
    // A high GPU score must not let an illegal path survive into the result.
    const invalid = { skills: ['Grow', 'Impossible'], score: 1e9 };
    const replays = replayGpuPaths(
      state,
      config,
      'neutral',
      targets,
      [invalid, valid],
    );
    expect(replays).toHaveLength(1);
    expect(replays[0].path).toBe(valid);
    expect(replays.some((entry) => entry.path === invalid)).toBe(false);
  });

  it('fails if applySkill is skipped and the start state is scored instead', () => {
    const { state, config } = makeFixture();
    const paths = [
      { skills: ['Polish'] },
      { skills: ['Grow', 'Polish'] },
      { skills: ['Grow', 'Grow'] },
    ];
    const replays = replayGpuPaths(state, config, 'neutral', targets, paths);
    expect(replays[0].path).toBe(paths[1]);

    // Mutation proof: scoring the start state returns the same number for
    // every proposal, so a replay that skipped applySkill would keep proposal
    // order and pick paths[0]. That disagrees with the assertion above.
    const context = __testing.buildScoringContext(config, state, 'neutral');
    const startOnlyScores = paths.map(() =>
      __testing.scoreState(
        state,
        targets.completion,
        targets.perfection,
        config.isSublimeCraft || false,
        config.targetMultiplier || 2.0,
        config.trainingMode || false,
        config.maxCompletion,
        config.maxPerfection,
        context,
      ),
    );
    expect(startOnlyScores[0]).toBe(startOnlyScores[1]);
    expect(paths[0].skills).not.toEqual(replays[0].path.skills);
  });
});
