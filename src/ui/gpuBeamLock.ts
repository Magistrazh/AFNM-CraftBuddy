import type {
  CraftBuddySettings,
  SearchEngine,
  SearchPresetBudget,
} from '../settings';

/** The resident GPU kernel is compiled with this beam width. */
export const GPU_BEAM_WIDTH = 20;

/** The resident GPU kernel is compiled with one beam width. */
export function isGpuSearchEngine(engine: SearchEngine): boolean {
  return engine === 'gpu';
}

/** Selecting an engine also pins the beam width when a GPU backend is chosen. */
export function engineChangePatch(
  engine: SearchEngine,
): Partial<CraftBuddySettings> {
  return isGpuSearchEngine(engine)
    ? { searchEngine: engine, searchBeamWidth: GPU_BEAM_WIDTH }
    : { searchEngine: engine };
}

/** Presets must not write a beam width while the GPU kernel is selected. */
export function presetChangePatch(
  budget: SearchPresetBudget,
  gpuMode: boolean,
): Partial<CraftBuddySettings> {
  if (!gpuMode) return budget;
  const { searchBeamWidth: _locked, ...rest } = budget;
  return rest;
}

/** Show the preset as active even though the GPU beam never matches its own. */
export function budgetMatchesSettings(
  budget: SearchPresetBudget,
  settings: CraftBuddySettings,
  gpuMode: boolean,
): boolean {
  return (
    budget.lookaheadDepth === settings.lookaheadDepth &&
    budget.searchTimeBudgetMs === settings.searchTimeBudgetMs &&
    budget.searchMaxNodes === settings.searchMaxNodes &&
    (gpuMode || budget.searchBeamWidth === settings.searchBeamWidth)
  );
}

/** The branches slider is read-only while a GPU backend is selected. */
export function allowSliderCommit(key: string, gpuMode: boolean): boolean {
  return !(gpuMode && key === 'searchBeamWidth');
}
