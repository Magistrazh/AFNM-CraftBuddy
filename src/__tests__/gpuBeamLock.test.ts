import {
  DEFAULT_SETTINGS,
  getSettings,
  resetSettings,
  saveSettings,
  SEARCH_PRESET_BUDGETS,
} from '../settings';
import {
  allowSliderCommit,
  budgetMatchesSettings,
  engineChangePatch,
  GPU_BEAM_WIDTH,
  isGpuSearchEngine,
  presetChangePatch,
} from '../ui/gpuBeamLock';

describe('GPU beam lock', () => {
  let logSpy: jest.SpyInstance;
  let storageData: Record<string, string>;

  beforeEach(() => {
    logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    storageData = {};
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: (key: string) => storageData[key] ?? null,
        setItem: (key: string, value: string) => {
          storageData[key] = String(value);
        },
        removeItem: (key: string) => {
          delete storageData[key];
        },
      } as Storage,
    });
    resetSettings();
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  it('recognises both GPU engines but not JS', () => {
    expect(isGpuSearchEngine('gpu')).toBe(true);
    expect(isGpuSearchEngine('gpuVerified')).toBe(true);
    expect(isGpuSearchEngine('js')).toBe(false);
  });

  it('writes 20 into the saved patch when a GPU engine is selected', () => {
    expect(engineChangePatch('gpu')).toEqual({
      searchEngine: 'gpu',
      searchBeamWidth: GPU_BEAM_WIDTH,
    });
    expect(engineChangePatch('gpuVerified')).toEqual({
      searchEngine: 'gpuVerified',
      searchBeamWidth: GPU_BEAM_WIDTH,
    });
    expect(engineChangePatch('js')).toEqual({ searchEngine: 'js' });
  });

  it('blocks only the branches slider while GPU is selected', () => {
    expect(allowSliderCommit('searchBeamWidth', true)).toBe(false);
    expect(allowSliderCommit('lookaheadDepth', true)).toBe(true);
    expect(allowSliderCommit('searchBeamWidth', false)).toBe(true);
  });

  it('keeps presets from overwriting the locked beam width', () => {
    const budget = SEARCH_PRESET_BUDGETS.max;
    expect(presetChangePatch(budget, true)).toEqual({
      lookaheadDepth: budget.lookaheadDepth,
      searchTimeBudgetMs: budget.searchTimeBudgetMs,
      searchMaxNodes: budget.searchMaxNodes,
    });
    expect(presetChangePatch(budget, false)).toEqual(budget);
  });

  it('still shows the matching preset active when the GPU beam differs', () => {
    const budget = SEARCH_PRESET_BUDGETS.fast;
    const settings = {
      ...DEFAULT_SETTINGS,
      ...budget,
      searchBeamWidth: GPU_BEAM_WIDTH,
    };
    expect(budgetMatchesSettings(budget, settings, true)).toBe(true);
    expect(budgetMatchesSettings(budget, settings, false)).toBe(false);
  });

  it('persists 20 on GPU selection and keeps it after returning to JS', () => {
    saveSettings(engineChangePatch('gpu'));
    expect(getSettings().searchBeamWidth).toBe(GPU_BEAM_WIDTH);

    saveSettings(engineChangePatch('js'));
    expect(getSettings().searchEngine).toBe('js');
    expect(getSettings().searchBeamWidth).toBe(GPU_BEAM_WIDTH);
  });
});
