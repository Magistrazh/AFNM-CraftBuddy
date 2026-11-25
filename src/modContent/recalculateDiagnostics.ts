/**
 * Diagnostics for the panel's "Update plan" button.
 *
 * Kept separate from `modContent/index.ts` so the event shape can be tested
 * without importing the side-effect-heavy integration module. Nothing here
 * knows about the search engine: callers pass the values already in hand.
 */

import { emitDebugEvent } from '../utils/debugSink';

export interface RecalculateStartInfo {
  ok: boolean;
  reason?: string;
  searchEngine: string;
  lookaheadDepth: number;
  timeBudgetMs: number;
  maxNodes: number;
  beamWidth: number;
  step: number | null;
  condition: string | null;
}

export interface RecalculateResultInfo {
  ok: boolean;
  engine: string | null;
  pick: string | null;
  score: number | null;
  reason?: string;
}

export type RecalculateResultListener = (info: RecalculateResultInfo) => void;

/** Emitted before dispatch, and once more with `ok:false` when there is no craft. */
export function emitRecalculateStart(info: RecalculateStartInfo): void {
  emitDebugEvent({
    kind: 'recalculate',
    trigger: 'settings',
    ok: info.ok,
    ...(info.reason ? { reason: info.reason } : {}),
    searchEngine: info.searchEngine,
    lookaheadDepth: info.lookaheadDepth,
    timeBudgetMs: info.timeBudgetMs,
    maxNodes: info.maxNodes,
    beamWidth: info.beamWidth,
    step: info.step,
    condition: info.condition,
  });
}

/** Emitted from the real recommendation result path, including failure/stale paths. */
export function emitRecalculateResult(
  info: RecalculateResultInfo & { elapsedMs: number },
): void {
  emitDebugEvent({
    kind: 'recalculateResult',
    trigger: 'settings',
    ok: info.ok,
    engine: info.engine,
    pick: info.pick,
    score: info.score,
    elapsedMs: info.elapsedMs,
    ...(info.reason ? { reason: info.reason } : {}),
  });
}
