/**
 * GPU-assisted root selection for the recommendation search.
 *
 * The GPU beam cannot be trusted as an oracle: its model freezes buff state and
 * ignores the engine's scoring, so a plan it likes may be wrong. What it can do
 * is see all opening skills at once and score them several levels deep, which
 * the engine's greedy pass does not.
 *
 * This module uses that ranking only to pick a *root set*. The engine still
 * evaluates every chosen root exactly, with its own transitions and scoring, so
 * the numbers the panel shows never come from the GPU.
 *
 * Measured on the 17 replay fixtures: the engine's greedy top-8 preserves the
 * full search's move and score in all 17 cases at a 600 ms budget; 4-6 roots
 * lose on two fixtures, 10 or more lose on one. The GPU may add one root the
 * greedy pass missed; the union stays at nine, inside the measured range.
 */

import { findBestSkill, type OptimizerConfig } from '../optimizer';
import {
  describeProfile,
  rankRoots,
  type GpuProfile,
} from '../optimizer/gpu';
import { gpuDebugLog as gpuLog } from '../optimizer/gpu/debug';
import type { CraftingState } from '../optimizer/state';
import type { SearchConfig } from '../optimizer/search/types';

/** Roots the engine's own greedy pass ranks highest; measured safe baseline. */
const GREEDY_ROOTS = 8;
/** Extra opening the GPU beam may add when the greedy pass does not see it. */
const GPU_EXTRA_ROOTS = 1;
const GREEDY_BUDGET_MS = 300;
const GPU_TIMEOUT_MS = 2500;
/** Ranking depth is fixed: the user's lookahead depth can be 96. */
const GPU_RANK_DEPTH = 6;
const GPU_BEAM_WIDTH = 1000;

export interface GpuRootSelection {
  /** Skill keys to hand to the engine as `rootSkillKeys`. */
  readonly roots: string[];
  readonly greedyRoots: string[];
  readonly gpuRoots: string[];
  /** Top opening according to the GPU beam; '' when it did not report one. */
  readonly gpuTopName: string;
  readonly rankingMs: number;
  readonly greedyMs: number;
  /** GPU profile from the ranking pass, when the backend reported one. */
  readonly profile?: GpuProfile;
  /** One-line status for the settings panel and the logs. */
  readonly status: string;
}

export type GpuRootOutcome =
  | { readonly ok: true; readonly selection: GpuRootSelection }
  | { readonly ok: false; readonly reason: string };

export interface GpuRootRequest {
  readonly state: CraftingState;
  readonly config: OptimizerConfig;
  readonly targetCompletion: number;
  readonly targetPerfection: number;
  readonly depth: number;
  readonly condition?: string;
  readonly forecast: readonly string[];
  readonly searchConfig: Partial<SearchConfig>;
}

const unique = (values: readonly string[]): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    if (!value || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
};

const skillKeyByName = (config: OptimizerConfig): Map<string, string> => {
  const map = new Map<string, string>();
  for (const skill of config.skills) map.set(skill.name, skill.key);
  return map;
};

/**
 * Never throws. Any GPU failure is reported as `{ok:false}` so the caller can
 * fall back to the full-root search with an explicit reason.
 */
export async function selectGpuRoots(
  request: GpuRootRequest,
): Promise<GpuRootOutcome> {
  const started = performance.now();
  let greedyRoots: string[] = [];
  let greedyMs = 0;
  try {
    const greedyStart = performance.now();
    const greedy = findBestSkill(
      request.state,
      request.config,
      request.targetCompletion,
      request.targetPerfection,
      true,
      request.depth,
      request.condition,
      [...request.forecast],
      // No transposition cache here: greedy keys are different and must not
      // pollute the lookahead's cross-step table.
      { timeBudgetMs: GREEDY_BUDGET_MS },
    );
    greedyMs = performance.now() - greedyStart;
    const picks = [
      greedy.recommendation,
      ...greedy.alternativeSkills,
    ].filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
    greedyRoots = unique(picks.map((entry) => entry.skill.key)).slice(
      0,
      GREEDY_ROOTS,
    );
  } catch (error) {
    greedyMs = performance.now() - started;
    return {
      ok: false,
      reason: `жадный проход не удался: ${String(error)}`,
    };
  }

  if (greedyRoots.length === 0) {
    return { ok: false, reason: 'жадный проход не выбрал ни одного корня' };
  }

  const ranking = await rankRoots(request.state, request.config, request.condition, {
    depth: Math.min(GPU_RANK_DEPTH, Math.max(1, request.depth)),
    beamWidth: GPU_BEAM_WIDTH,
    timeoutMs: GPU_TIMEOUT_MS,
  });
  if (!ranking.available) {
    return { ok: false, reason: ranking.reason ?? 'GPU-ранжирование недоступно' };
  }

  const byName = skillKeyByName(request.config);
  const gpuRoots = ranking.roots
    .slice(0, GPU_EXTRA_ROOTS)
    .map((root) => byName.get(root.name) ?? '')
    .filter(Boolean);
  const roots = unique([...greedyRoots, ...gpuRoots]);
  if (roots.length === 0) {
    return { ok: false, reason: 'ранжирование не дало ни одного корня' };
  }

  const wallMs = Math.round(performance.now() - started - greedyMs);
  const rankingMs = Math.round(ranking.timing?.totalMs ?? wallMs);
  if (ranking.profile) {
    gpuLog(
      `ranking wallMs=${wallMs} ` + describeProfile(ranking.profile),
    );
  }
  return {
    ok: true,
    selection: {
      roots,
      greedyRoots,
      gpuRoots,
      gpuTopName: ranking.roots[0]?.name ?? '',
      rankingMs,
      greedyMs: Math.round(greedyMs),
      profile: ranking.profile,
      status:
        `GPU: корней ${ranking.roots.length}, в поиске ${roots.length}` +
        ` (жадный ${greedyMs.toFixed(0)} мс, ранг ${rankingMs} мс)`,
    },
  };
}
