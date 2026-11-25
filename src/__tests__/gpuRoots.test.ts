import { NODE, NODE_ROOT, NS } from '../optimizer/gpu/records';
import { summarizeRoots } from '../optimizer/gpu/search';

interface NodeSpec {
  readonly root: number;
  readonly step: number;
  readonly alive: number;
  readonly stability: number;
  readonly completion: number;
  readonly perfection: number;
}

function makeNodes(specs: readonly NodeSpec[]): Float32Array {
  const values = new Float32Array(specs.length * NS);
  specs.forEach((spec, index) => {
    const base = index * NS;
    values[base + NODE_ROOT] = spec.root;
    values[base + NODE] = spec.step;
    values[base + NODE + 1] = spec.alive;
    values[base + 1] = spec.stability;
    values[base + 13] = spec.completion;
    values[base + 14] = spec.perfection;
  });
  return values;
}

const SKILLS = [
  { name: 'Alpha' },
  { name: 'Beta' },
  { name: 'Gamma' },
  { name: 'Delta' },
];

describe('GPU root ranking', () => {
  it('keeps the best living node per root and counts dead nodes too', () => {
    const values = makeNodes([
      // Level-0 root: step 0, freshly stamped, best Alpha (10*3 + 5 + 20*0.05).
      { root: 0, step: 0, alive: 1, stability: 20, completion: 10, perfection: 5 },
      // Inherited root (step > 0); 1.4 must round to skill 1 (Beta).
      { root: 1.4, step: 2, alive: 1, stability: 10, completion: 8, perfection: 4 },
      // Dead-only root: would outscore everything, but must never appear.
      { root: 2, step: 1, alive: 0, stability: 100, completion: 100, perfection: 100 },
      // Out-of-range id: dropped completely, not even counted.
      { root: 99, step: 0, alive: 1, stability: 100, completion: 100, perfection: 100 },
      // Second Beta node, strictly worse; must not replace the first one.
      { root: 1, step: 3, alive: 1, stability: 1, completion: 1, perfection: 0 },
      // Dead Alpha node: counted in Alpha's `nodes`, cannot win the ranking.
      { root: 0, step: 1, alive: 0, stability: 100, completion: 100, perfection: 100 },
      { root: 3, step: 0, alive: 1, stability: 40, completion: 5, perfection: 2 },
    ]);

    expect(summarizeRoots(values, 7, SKILLS)).toEqual([
      { name: 'Alpha', score: 36, nodes: 2, completion: 10, perfection: 5 },
      { name: 'Beta', score: 28.5, nodes: 2, completion: 8, perfection: 4 },
      { name: 'Delta', score: 19, nodes: 1, completion: 5, perfection: 2 },
    ]);
  });

  it('returns nothing when no root has a living node', () => {
    const values = makeNodes([
      { root: 0, step: 2, alive: 0, stability: 50, completion: 50, perfection: 50 },
      { root: 7, step: 0, alive: 1, stability: 50, completion: 50, perfection: 50 },
    ]);
    expect(summarizeRoots(values, 2, SKILLS)).toEqual([]);
  });
});
