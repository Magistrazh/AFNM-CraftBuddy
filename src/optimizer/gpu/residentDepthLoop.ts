/**
 * Parameterized device-resident depth loop. Stand-only: no production wiring.
 *
 * Replaces the hard-coded three-level orchestration of
 * `scripts/resident-depth2-stand.ts`. The loop is parameterized by depth,
 * node cap and beam width; the game-specific tables that must be rebuilt from
 * the CURRENT beam (gains / availability / style) are injected as callbacks,
 * so this module stays independent of the engine and of host scoring code.
 *
 * Every level is one `runOneResidentLevel` submit (A<->B ping-pong is already
 * keyed by level parity there). No per-case SCORE_INPUT is built on the host;
 * only the small constant tables are (re)uploaded between submits. There is
 * deliberately no wall-clock check: the owner excluded time from this run.
 */

import {
  BEAM,
  DEPTH_MAX,
  META,
  runOneResidentLevel,
} from './residentBeam';
import type {
  ResidentBuffers,
  ResidentLevelResult,
  ResidentPipelines,
} from './residentBeam';

/** Constant tables rebuilt from the previous level's device readback. */
export interface ResidentDepthRecompute {
  /** Gains keyed by (inputSlot * SKILL_CAPACITY + skill, conditionKind, word). */
  readonly gains?: (level: number, previous: ResidentLevelResult) => Float32Array;
  /** Availability stock/cost base, keyed by skill. */
  readonly availParamBase?: (level: number, previous: ResidentLevelResult) => Float32Array;
  /** Scalar style buffer used by prepareCondParams / prepareVerdictInputs. */
  readonly style?: (level: number, previous: ResidentLevelResult) => Float32Array;
  /** Per-case style fallback keyed by skill, used by prepareStyleCases. */
  readonly styleCases?: (level: number, previous: ResidentLevelResult) => Float32Array;
}

export interface ResidentDepthOptions {
  /** Requested levels, clamped to 0..DEPTH_MAX (96). */
  readonly depth: number;
  /** Cumulative alive-node cap checked BETWEEN submits, never inside a dispatch. */
  readonly maxNodes: number;
  /**
   * Beam width. The resident WGSL constants are compiled with BEAM=20, so any
   * other value is rejected rather than silently misreporting a narrower beam.
   */
  readonly beamWidth: number;
  readonly recompute?: ResidentDepthRecompute;
  /** Called after each level's readback, before the next level's submit. */
  readonly observe?: (level: number, result: ResidentLevelResult) => void;
  /**
   * Optional engine-style root fold. Defaults to the level-0 beam winner.
   * May be async: a stand re-seed needs extra device submits after the loop.
   * The default synchronous path is unchanged.
   */
  readonly rankRoots?: (
    levels: readonly ResidentLevelResult[],
  ) => { readonly move: number; readonly score: number }
    | Promise<{ readonly move: number; readonly score: number }>;
}

export interface ResidentDepthResult {
  readonly depth: number;
  readonly levels: readonly ResidentLevelResult[];
  /** Device alive-node count per completed level (counters[1]). */
  readonly nodeCounts: readonly number[];
  readonly totalNodes: number;
  readonly completedLevels: number;
  readonly stoppedByNodeCap: boolean;
  /** Engine skill index of the chosen root move. */
  readonly topMove: number;
  readonly topScore: number;
}

function clampInt(value: number, min: number, max: number): number {
  const n = Number.isFinite(value) ? Math.floor(value) : min;
  return Math.max(min, Math.min(max, n));
}

function aliveCount(result: ResidentLevelResult): number {
  const deviceCount = result.counters[1];
  if (Number.isFinite(deviceCount) && deviceCount > 0) return deviceCount;
  let alive = 0;
  for (let i = 0; i < BEAM; i++) {
    if (result.meta[i * META.WORDS + META.alive] >= 0.5) alive++;
  }
  return alive;
}

/**
 * Run `depth` resident levels from the root tables already uploaded in
 * `buffers`. Before each level after the first, the injected callbacks rebuild
 * the constant tables from the PREVIOUS level's nodes — never from the root.
 * The root-reuse bug previously produced a -7873 EV divergence.
 */
export async function runResidentDepth(
  device: GpuDevice,
  pipelines: ResidentPipelines,
  buffers: ResidentBuffers,
  options: ResidentDepthOptions,
): Promise<ResidentDepthResult> {
  const depth = clampInt(options.depth, 0, DEPTH_MAX);
  const beamWidth = Math.floor(options.beamWidth);
  if (beamWidth !== BEAM) {
    throw new Error(`runResidentDepth: beamWidth ${beamWidth} requires kernel rebuild (compiled BEAM=${BEAM})`);
  }
  const maxNodes = Number.isFinite(options.maxNodes)
    ? Math.max(0, Math.floor(options.maxNodes))
    : Number.POSITIVE_INFINITY;

  const levels: ResidentLevelResult[] = [];
  const nodeCounts: number[] = [];
  let totalNodes = 0;
  let stoppedByNodeCap = false;

  for (let level = 0; level < depth; level++) {
    // Node cap is a boundary condition only; no time check anywhere in the loop.
    if (level > 0 && totalNodes > maxNodes) {
      stoppedByNodeCap = true;
      break;
    }
    if (level > 0) {
      const previous = levels[level - 1] as ResidentLevelResult;
      const rebuild = options.recompute;
      if (rebuild?.gains) {
        device.queue.writeBuffer(buffers.gains, 0, rebuild.gains(level, previous));
      }
      if (rebuild?.availParamBase) {
        device.queue.writeBuffer(buffers.availParamBase, 0, rebuild.availParamBase(level, previous));
      }
      if (rebuild?.style) {
        device.queue.writeBuffer(buffers.style, 0, rebuild.style(level, previous));
      }
      if (rebuild?.styleCases) {
        device.queue.writeBuffer(buffers.styleCasesBase, 0, rebuild.styleCases(level, previous));
      }
    }
    const result = await runOneResidentLevel(device, pipelines, buffers, {
      depth,
      level,
    });
    const count = aliveCount(result);
    levels.push(result);
    nodeCounts.push(count);
    totalNodes += count;
    options.observe?.(level, result);
  }

  const ranked = options.rankRoots
    ? await options.rankRoots(levels)
    : (() => {
        const first = levels[0];
        const move = first
          ? Math.round(first.meta[META.skill])
          : -1;
        const score = first?.values[Math.round(first.meta[META.skill])] ?? Number.NaN;
        return { move, score };
      })();

  return {
    depth,
    levels,
    nodeCounts,
    totalNodes,
    completedLevels: levels.length,
    stoppedByNodeCap,
    topMove: ranked.move,
    topScore: ranked.score,
  };
}
