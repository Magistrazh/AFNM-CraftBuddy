/**
 * S7b depth router: depth <= 2 stays on the validated stateful pipeline,
 * depth > 2 uses the resident GPU beam. No real WebGPU is involved here.
 */

import {
  tryGpuRecommendation,
  type StatefulRecommendationInput,
} from '../modContent/statefulGpuRecommendation';
import { findBestSkill } from '../optimizer';
import { DEFAULT_SEARCH_CONFIG } from '../optimizer/search/config';
import type { SearchResult } from '../optimizer';
import type { OptimizerConfig, SkillDefinition } from '../optimizer/skills';
import type { CraftingState } from '../optimizer/state';
import type { StatefulSearchResult } from '../optimizer/gpu/statefulSearch';
import { setDebugSink } from '../utils/debugSink';

/** The router injects the resident function; these fakes never touch a device. */
function makeInput(depth = 4): StatefulRecommendationInput {
  const skill = { key: 'forge', name: 'Forge' } as unknown as SkillDefinition;
  return {
    state: { qiDroplets: 10 } as unknown as CraftingState,
    config: { skills: [skill] } as unknown as OptimizerConfig,
    skills: [skill],
    completion: 100,
    perfection: 100,
    lookaheadDepth: depth,
    condition: 'neutral',
    forecast: ['neutral', 'neutral', 'neutral'],
    searchConfig: {
      ...DEFAULT_SEARCH_CONFIG,
      timeBudgetMs: 4000,
      maxNodes: 123456,
      beamWidth: 20,
      goalPriorityBias: -50,
      overcraftAmbition: false,
    },
  };
}

const fakeResult = {
  recommendation: { skill: { name: 'Forge' }, score: 12 },
} as unknown as SearchResult;

function residentOk() {
  return {
    chosen: 'forge',
    comparableScore: -21_000_000,
    path: ['forge'],
    nodeCounts: [20, 20, 20],
    totalNodes: 60,
    completedLevels: 3,
    stoppedByNodeCap: false,
    timing: { gpuMs: 12, totalMs: 20 },
    notes: { requestedDepth: 4 },
  };
}

function statefulRefusal(): StatefulSearchResult {
  return {
    chosen: null,
    score: Number.NaN,
    path: [],
    order: [],
    scores: {},
    timing: { encodeMs: 0, gpuMs: 0, readbackMs: 0, totalMs: 0 },
    notes: {},
    refusal: 'WebGPU недоступен',
  };
}

describe('depth router for GPU recommendations', () => {
  let events: Array<Record<string, unknown>> = [];

  beforeEach(() => {
    events = [];
    setDebugSink((body) => {
      events.push(JSON.parse(body) as Record<string, unknown>);
    });
  });

  afterEach(() => {
    setDebugSink(null);
  });

  it('sends depth 4 to the resident search and depth 2 to the old search', async () => {
    const resident = jest.fn().mockResolvedValue({
      ...residentOk(),
      refusal: 'WebGPU недоступен',
    });
    const oldSearch = jest.fn().mockResolvedValue(statefulRefusal());
    const log = () => undefined;

    await tryGpuRecommendation(makeInput(4), {
      runResidentSearch: resident as never,
      runSearch: oldSearch as never,
      log,
    });
    await tryGpuRecommendation(makeInput(2), {
      runResidentSearch: resident as never,
      runSearch: oldSearch as never,
      log,
    });

    expect(resident).toHaveBeenCalledTimes(1);
    expect(oldSearch).toHaveBeenCalledTimes(1);
    const deepInput = resident.mock.calls[0][0];
    expect(deepInput.lookaheadDepth).toBe(4);
    expect(deepInput.searchConfig.timeBudgetMs).toBe(4000);
    expect(deepInput.searchConfig.maxNodes).toBe(123456);
    expect(deepInput.searchConfig.beamWidth).toBe(20);
    expect(deepInput.searchConfig.goalPriorityBias).toBe(-50);
  });

  it('passes the engine settings through unchanged on a successful deep search', async () => {
    const find = jest.fn().mockReturnValue(fakeResult);
    const result = await tryGpuRecommendation(makeInput(6), {
      runResidentSearch: jest.fn().mockResolvedValue(residentOk()) as never,
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
    });

    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error('expected success');
    expect(result.chosenKey).toBe('forge');
    expect(result.gpuScore).toBe(-21_000_000);
    expect(result.status).toContain('глубина 6');
    expect(result.status).toContain('уровней 3');
    const searchCfg = find.mock.calls[0][8] as Record<string, unknown>;
    expect(searchCfg.rootSkillKeys).toEqual(['forge']);
    expect(searchCfg.goalPriorityBias).toBe(-50);
    expect(find.mock.calls[0][5]).toBe(6);
    const event = events.find((e) => e.kind === 'gpu-resident');
    expect(event?.ok).toBe(true);
    expect(event?.depth).toBe(6);
  });

  it('refuses loudly on a device refusal without calling the engine', async () => {
    const find = jest.fn().mockReturnValue(fakeResult);
    const result = await tryGpuRecommendation(makeInput(), {
      runResidentSearch: jest.fn().mockResolvedValue({
        ...residentOk(),
        chosen: null,
        refusal: 'device lost',
        notes: { refusals: 1 },
      }) as never,
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain('device lost');
    expect(find).not.toHaveBeenCalled();
    const event = events.find((e) => e.kind === 'gpu-resident');
    expect(event?.ok).toBe(false);
    expect(event?.notes).toEqual({ refusals: 1 });
  });

  it('refuses an incomplete level fold instead of answering with a shallow plan', async () => {
    const find = jest.fn().mockReturnValue(fakeResult);
    const result = await tryGpuRecommendation(makeInput(10), {
      runResidentSearch: jest.fn().mockResolvedValue({
        ...residentOk(),
        completedLevels: 2,
      }) as never,
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain('уровней 2');
    expect(find).not.toHaveBeenCalled();
  });

  it('converts an unexpected backend exception into a refusal', async () => {
    const result = await tryGpuRecommendation(makeInput(), {
      runResidentSearch: jest.fn().mockRejectedValue(new Error('device lost')) as never,
      log: () => undefined,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain('device lost');
  });
});
