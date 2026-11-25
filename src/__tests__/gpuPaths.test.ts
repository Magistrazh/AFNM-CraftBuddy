import { NODE, NODE_ROOT, NS } from '../optimizer/gpu/records';
import { reconstructPaths } from '../optimizer/gpu/search';

const DEPTH = 3;
const WIDTH = 4;
const SKILLS = [
  { name: 'S0' },
  { name: 'S1' },
  { name: 'S2' },
  { name: 'S3' },
  { name: 'S4' },
];

interface FinalSpec {
  readonly alive: number;
  readonly step: number;
  readonly completion: number;
  readonly perfection: number;
  readonly stability: number;
}

function makeNodes(specs: readonly FinalSpec[]): Float32Array {
  const values = new Float32Array(specs.length * NS);
  specs.forEach((spec, index) => {
    const base = index * NS;
    values[base + NODE_ROOT] = 0;
    values[base + NODE] = spec.step;
    values[base + NODE + 1] = spec.alive;
    values[base + 1] = spec.stability;
    values[base + 13] = spec.completion;
    values[base + 14] = spec.perfection;
  });
  return values;
}

function put(trace: Float32Array, level: number, node: number, parent: number, skill: number): void {
  const entry = (level * WIDTH + node) * 2;
  trace[entry] = parent;
  trace[entry + 1] = skill;
}

/**
 * 3 levels, 4 nodes wide. Each (level, node) entry is written once; beam nodes
 * may share ancestors, so node 3 reuses the level-1/level-0 entries of node 0.
 *
 *   node 0: [1, 2, 3], score 30 (winner)
 *   node 2: [1, 2, 3], score 24 (duplicate of node 0)
 *   node 1: [4, 0, 2], score 20 (second distinct path)
 *   node 3: [1, 2, 4], score 355 but step 2 -> dead ancestor, skipped
 */
function makeTrace(): Float32Array {
  const trace = new Float32Array(DEPTH * WIDTH * 2);
  // Final node 0: level 2 <- node 2 (S3), level 1 <- node 1 (S2), level 0 <- node 0 (S1).
  put(trace, 2, 0, 2, 3);
  put(trace, 1, 2, 1, 2);
  put(trace, 0, 1, 0, 1);
  // Final node 1: [4, 0, 2].
  put(trace, 2, 1, 3, 2);
  put(trace, 1, 3, 0, 0);
  put(trace, 0, 0, 2, 4);
  // Final node 2: [1, 2, 3] again, to exercise dedupe.
  put(trace, 2, 2, 0, 3);
  put(trace, 1, 0, 3, 2);
  put(trace, 0, 3, 1, 1);
  // Final node 3: [1, 2, 4], but its node record has a step deficit.
  put(trace, 2, 3, 2, 4);
  return trace;
}

function makeFinalNodes(): Float32Array {
  return makeNodes([
    { alive: 1, step: 3, completion: 10, perfection: 0, stability: 0 }, // 30
    { alive: 1, step: 3, completion: 5, perfection: 5, stability: 0 }, // 20
    { alive: 1, step: 3, completion: 8, perfection: 0, stability: 0 }, // 24, duplicate
    { alive: 1, step: 2, completion: 100, perfection: 100, stability: 100 }, // 355, dead ancestor
  ]);
}

const EXPECTED = [
  { skills: ['S1', 'S2', 'S3'], score: 30, completion: 10, perfection: 0, stability: 0 },
  { skills: ['S4', 'S0', 'S2'], score: 20, completion: 5, perfection: 5, stability: 0 },
];

describe('GPU replay paths', () => {
  it('returns ranked, deduped skill sequences and skips dead-ancestor paths', () => {
    expect(reconstructPaths(makeTrace(), makeFinalNodes(), DEPTH, WIDTH, SKILLS, 16)).toEqual(
      EXPECTED,
    );
  });

  it('caps the result with limit', () => {
    expect(reconstructPaths(makeTrace(), makeFinalNodes(), DEPTH, WIDTH, SKILLS, 1)).toEqual(
      [EXPECTED[0]],
    );
  });

  it('returns nothing for a zero-depth search', () => {
    expect(reconstructPaths(new Float32Array(0), new Float32Array(0), 0, WIDTH, SKILLS, 16)).toEqual(
      [],
    );
  });

  it('skips paths with an out-of-range parent or skill index', () => {
    const badParent = makeTrace();
    put(badParent, 2, 3, 99, 4); // would outscore everything, parent 99 >= WIDTH
    const nodes = makeFinalNodes();
    const base = 3 * NS;
    nodes[base + NODE] = 3; // alive, complete step: only the index is wrong
    expect(reconstructPaths(badParent, nodes, DEPTH, WIDTH, SKILLS, 16)).toEqual(EXPECTED);

    const badSkill = makeTrace();
    put(badSkill, 2, 3, 2, SKILLS.length + 73); // skill 78 out of range
    expect(reconstructPaths(badSkill, nodes, DEPTH, WIDTH, SKILLS, 16)).toEqual(EXPECTED);
  });

  /**
   * Fixture discrimination: the expected sequences are only produced by a
   * stride-2 read of `(parent, skill)` walked from the last level backwards.
   * Both transformed traces below feed the real function and must disagree
   * with the correct result, so a regression to either bug fails the suite.
   */
  it('fails if the trace is read with the wrong stride or the wrong direction', () => {
    const trace = makeTrace();
    const nodes = makeFinalNodes();
    const correct = reconstructPaths(trace, nodes, DEPTH, WIDTH, SKILLS, 16);

    // Stride 1 reads entry i from trace[i] / trace[i+1]. Repack the fixture so
    // the stride-2 reader sees exactly those values.
    const wrongStride = new Float32Array(trace.length);
    for (let i = 0; i + 1 < trace.length; i++) {
      wrongStride[i * 2] = trace[i];
      wrongStride[i * 2 + 1] = trace[i + 1];
    }
    expect(reconstructPaths(wrongStride, nodes, DEPTH, WIDTH, SKILLS, 16)).not.toEqual(correct);

    // Swap level 0 with level 2: a forward/backwards mix-up yields different
    // skill orders (the correct sequences are not palindromes).
    const wrongDirection = new Float32Array(trace.length);
    for (let level = 0; level < DEPTH; level++) {
      const from = (DEPTH - 1 - level) * WIDTH * 2;
      const to = level * WIDTH * 2;
      for (let i = 0; i < WIDTH * 2; i++) wrongDirection[to + i] = trace[from + i];
    }
    expect(reconstructPaths(wrongDirection, nodes, DEPTH, WIDTH, SKILLS, 16)).not.toEqual(correct);
  });
});
