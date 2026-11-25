/**
 * S7b wiring contract: on any stateful GPU refusal no GPU result is published,
 * so `runRecommendationSearch` keeps its existing GPU/JS ladder untouched.
 */

import {
  tryStatefulRecommendation,
  GPU_STATEFUL_PARITY_LIMIT,
  GPU_STATEFUL_VERIFY_BUDGET_MS,
  GPU_STATEFUL_TIMEOUT_MS,
  GPU_STATEFUL_COLD_TIMEOUT_MS,
  type StatefulRecommendationInput,
} from '../modContent/statefulGpuRecommendation';
import { findBestSkill } from '../optimizer';
import { DEFAULT_SEARCH_CONFIG } from '../optimizer/search/config';
import type { SearchResult } from '../optimizer';
import type { OptimizerConfig, SkillDefinition } from '../optimizer/skills';
import type { CraftingState } from '../optimizer/state';
import type {
  StatefulSearchOptions,
  StatefulSearchResult,
} from '../optimizer/gpu/statefulSearch';
import { setDebugSink } from '../utils/debugSink';

const fakeResult = {
  recommendation: { skill: { name: 'Forge' }, score: 12 },
} as unknown as SearchResult;

function makeInput(): StatefulRecommendationInput {
  const skill = { key: 'forge', name: 'Forge' } as unknown as SkillDefinition;
  return {
    state: {} as CraftingState,
    config: { skills: [skill] } as unknown as OptimizerConfig,
    skills: [skill],
    completion: 100,
    perfection: 100,
    lookaheadDepth: 4,
    condition: 'neutral',
    forecast: [],
    searchConfig: { ...DEFAULT_SEARCH_CONFIG, timeBudgetMs: 4000 },
  };
}

function gpuResult(
  patch: Partial<StatefulSearchResult> = {},
): StatefulSearchResult {
  return {
    chosen: null,
    score: Number.NaN,
    path: [],
    order: [],
    scores: {},
    timing: { encodeMs: 1, gpuMs: 2, readbackMs: 3, totalMs: 6 },
    notes: {},
    refusal: 'WebGPU недоступен',
    ...patch,
  };
}

describe('stateful GPU wiring in the recommendation path', () => {
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

  it('gives cold device compilation its own timeout, warm searches the tight one', async () => {
    const seen: number[] = [];
    const runSearch = async (
      _state: CraftingState,
      _config: OptimizerConfig,
      _skills: readonly SkillDefinition[],
      options?: StatefulSearchOptions,
    ): Promise<StatefulSearchResult> => {
      seen.push(options?.timeoutMs ?? -1);
      return gpuResult();
    };
    await tryStatefulRecommendation(makeInput(), {
      runSearch,
      backendStatus: () => ({ active: false }),
      log: () => undefined,
    });
    await tryStatefulRecommendation(makeInput(), {
      runSearch,
      backendStatus: () => ({ active: true }),
      log: () => undefined,
    });
    expect(seen).toEqual([GPU_STATEFUL_COLD_TIMEOUT_MS, GPU_STATEFUL_TIMEOUT_MS]);
  });

  it('falls through to the JS path on refusal: no result, no engine search', async () => {
    const find = jest.fn().mockReturnValue(fakeResult);
    const result = await tryStatefulRecommendation(makeInput(), {
      runSearch: async () => gpuResult(),
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.refusal).toBe('WebGPU недоступен');
    expect(find).not.toHaveBeenCalled();
    const event = events.find((e) => e.kind === 'gpu-stateful');
    expect(event).toBeDefined();
    expect(event?.ok).toBe(false);
    expect(event?.parityLimit).toBe(GPU_STATEFUL_PARITY_LIMIT);
    expect(event?.timing).toEqual({
      encodeMs: 1,
      gpuMs: 2,
      readbackMs: 3,
      totalMs: 6,
    });
  });

  it('refuses on incomplete device records without calling the engine', async () => {
    const find = jest.fn().mockReturnValue(fakeResult);
    const result = await tryStatefulRecommendation(makeInput(), {
      runSearch: async () =>
        gpuResult({
          refusal: undefined,
          chosen: 'forge',
          notes: { terminalityMismatched: 1 },
        }),
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain('неполные записи stateful GPU');
    }
    expect(find).not.toHaveBeenCalled();
  });

  it('converts an unexpected backend exception into a refusal', async () => {
    const result = await tryStatefulRecommendation(makeInput(), {
      runSearch: async () => {
        throw new Error('device lost');
      },
      log: () => undefined,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain('device lost');
  });

  it('falls through when engine verification throws instead of throwing into the UI', async () => {
    const find = jest.fn(() => {
      throw new Error('engine broken');
    });
    const result = await tryStatefulRecommendation(makeInput(), {
      runSearch: async () =>
        gpuResult({ refusal: undefined, chosen: 'forge', notes: {} }),
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain('engine broken');
    expect(events.some((e) => e.kind === 'gpu-stateful' && e.ok === false)).toBe(true);
  });

  it('builds the panel result from one single-root engine search on success', async () => {
    const find = jest.fn().mockReturnValue(fakeResult);
    const input = makeInput();
    const result = await tryStatefulRecommendation(input, {
      runSearch: async (
        _state: CraftingState,
        _config: OptimizerConfig,
        _skills: SkillDefinition[],
        options?: StatefulSearchOptions,
      ) => {
        expect(options?.currentCondition).toBe('neutral');
        expect(options?.conditionQueue).toEqual([]);
        expect(options?.targetCompletion).toBe(100);
        expect(options?.targetPerfection).toBe(100);
        return gpuResult({
          refusal: undefined,
          chosen: 'forge',
          score: 42,
          path: ['forge'],
          order: ['forge'],
          notes: { stateNodes: 12 },
        });
      },
      find: find as unknown as typeof findBestSkill,
      log: () => undefined,
      now: (() => {
        let t = 0;
        return () => (t += 10);
      })(),
    });

    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error('expected success');
    expect(result.result).toBe(fakeResult);
    expect(result.chosenName).toBe('Forge');
    expect(result.status).toContain('Forge');
    expect(result.status).toContain(GPU_STATEFUL_PARITY_LIMIT);
    expect(find).toHaveBeenCalledTimes(1);
    const searchCfg = find.mock.calls[0][8] as Record<string, unknown>;
    expect(searchCfg.rootSkillKeys).toEqual(['forge']);
    expect(searchCfg.timeBudgetMs).toBe(
      Math.min(input.searchConfig.timeBudgetMs, GPU_STATEFUL_VERIFY_BUDGET_MS),
    );
    const event = events.find((e) => e.kind === 'gpu-stateful');
    expect(event?.ok).toBe(true);
    expect(event?.timing).toEqual({
      encodeMs: 1,
      gpuMs: 2,
      readbackMs: 3,
      totalMs: 6,
    });
  });
});
