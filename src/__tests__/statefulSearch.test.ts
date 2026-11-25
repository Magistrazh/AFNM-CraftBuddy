import { finalKernelSources, runStatefulSearch } from '../optimizer/gpu/statefulSearch';
import type { CraftingState } from '../optimizer/state';
import type { OptimizerConfig } from '../optimizer/skills';

describe('statefulSearch self-contained GPU backend', () => {
  it('builds every patched WGSL source at module load, never at call time', () => {
    const sources = finalKernelSources();
    expect(sources.patchApplied).toBe(true);
    expect(sources.state).toContain('let appliedSuccess = derivedSuccess;');
    expect(sources.state).toContain('const EV_MODE: bool = true;');
    expect(sources.state).not.toContain('let appliedSuccess = 1.0;');
    expect(sources.state).toContain('@group(0) @binding(6) var<storage, read> nodeParams');
    expect(sources.state).toContain('np(nodeBase, 32u)');
    expect(sources.condition).toContain('nodeParams[node * 6u + ');
    expect(sources.blend).toContain('fn blendKernel');
  });

  it('refuses loudly without WebGPU instead of throwing into the UI', async () => {
    const result = await runStatefulSearch(
      {} as CraftingState,
      {} as OptimizerConfig,
      [],
      { timeoutMs: 500 },
    );
    expect(result.chosen).toBeNull();
    expect(result.refusal).toBeTruthy();
    expect(result.notes.refusals).toBe(1);
    expect(result.path).toEqual([]);
    expect(Number.isFinite(result.timing.totalMs)).toBe(true);
  });
});
