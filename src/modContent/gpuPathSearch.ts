/**
 * Exact JS verification of GPU-proposed skill sequences.
 *
 * The GPU beam is only a proposal generator: its kernel freezes buff state,
 * carries no forecast queue and scores with its own heuristic. Here every
 * proposed sequence is replayed with the engine's own `applySkill` transition,
 * with one fixed condition, and scored with the engine's own `scoreState`.
 * Nothing in this module talks to a GPU, a debug sink or a global singleton.
 */

import {
  applySkill,
  getConditionEffectsForConfig,
  isTerminalState,
  type OptimizerConfig,
  type SkillDefinition,
} from '../optimizer/skills';
import type { CraftingState } from '../optimizer/state';
import { __testing } from '../optimizer/search';
import { DEFAULT_SEARCH_CONFIG } from '../optimizer/search/config';
import { normalizeConditionType } from '../optimizer/search/conditions';
import type { SearchConfig } from '../optimizer/search/types';
import type { GpuPath } from '../optimizer/gpu/search';

/** Minimum shape of a GPU proposal: only the skill names matter for replay. */
export interface GpuReplayPath {
  readonly skills: readonly string[];
}

export interface GpuPathTargets {
  readonly completion: number;
  readonly perfection: number;
  /**
   * Engine search settings that feed the optional `scoreState` arguments.
   * Missing fields fall back to `DEFAULT_SEARCH_CONFIG`, exactly as in
   * `runLookaheadSearch` (`{ ...DEFAULT_SEARCH_CONFIG, ...searchConfig }`).
   */
  readonly searchConfig?: Partial<SearchConfig>;
}

export interface GpuPathReplay<T extends GpuReplayPath = GpuPath> {
  readonly path: T;
  readonly endState: CraftingState;
  /** Engine score, never the score carried by the GPU path. */
  readonly score: number;
}

/**
 * Replay each `{skills: string[]}` proposal against the real engine and keep
 * only the ones that are legal all the way through. Results are sorted by the
 * engine score descending; ties keep GPU proposal order (stable in V8, and ES5
 * sort is stable in every engine the mod targets).
 *
 * The condition is held fixed for the whole path: the GPU kernel generated the
 * sequence under one condition with no forecast queue, so changing it mid-replay
 * would verify a different sequence than the one proposed. The engine re-runs
 * the chosen opening with its real condition transitions afterwards.
 */
export function replayGpuPaths<T extends GpuReplayPath>(
  state: CraftingState,
  config: OptimizerConfig,
  condition: string | undefined,
  targets: GpuPathTargets,
  paths: readonly T[],
): GpuPathReplay<T>[] {
  const currentCondition = normalizeConditionType(condition);
  // Computed once: the fixed-condition assumption above applies to effects too.
  const conditionEffects = getConditionEffectsForConfig(
    config,
    currentCondition,
  );
  const cfg: SearchConfig = { ...DEFAULT_SEARCH_CONFIG, ...targets.searchConfig };
  const context = __testing.buildScoringContext(
    config,
    state,
    currentCondition,
  );

  const skillsByName = new Map<string, SkillDefinition>();
  for (const skill of config.skills) {
    if (!skillsByName.has(skill.name)) skillsByName.set(skill.name, skill);
  }

  const results: GpuPathReplay<T>[] = [];
  for (const path of paths) {
    const skills: SkillDefinition[] = [];
    let known = true;
    for (const name of path.skills) {
      const skill = skillsByName.get(name);
      if (!skill) {
        known = false;
        break;
      }
      skills.push(skill);
    }
    if (!known) continue;

    let cursor = state;
    let valid = true;
    for (const skill of skills) {
      // A terminal state has no legal continuation, so any remaining step of
      // the proposal would be a fiction. `applySkill` would also return null,
      // but checking first makes the reason explicit.
      if (isTerminalState(cursor, config, currentCondition)) {
        valid = false;
        break;
      }
      const next = applySkill(
        cursor,
        skill,
        config,
        conditionEffects,
        targets.completion,
        currentCondition,
        targets.perfection,
      );
      if (next === null) {
        valid = false;
        break;
      }
      cursor = next;
    }
    if (!valid) continue;

    // Same arguments as lookahead.ts:264-278 (live, non-terminal branch).
    const score = __testing.scoreState(
      cursor,
      targets.completion,
      targets.perfection,
      config.isSublimeCraft || false,
      config.targetMultiplier || 2.0,
      config.trainingMode || false,
      config.maxCompletion,
      config.maxPerfection,
      context,
      cfg.goalPriorityBias,
      cfg.overcraftAmbition ?? true,
      cfg.perfectionBandGoal ?? 0,
      cfg.completionBandCeiling ?? 0,
    );
    // A NaN score would poison the ordering; treat it as an unusable proposal.
    if (!Number.isFinite(score)) continue;
    results.push({ path, endState: cursor, score });
  }

  results.sort((a, b) => (a.score !== b.score ? b.score - a.score : 0));
  return results;
}
