/**
 * GPU backend debug logging for the first in-game tests.
 *
 * The switch is read once at module load. Default is FALSE for release; set
 * `window.__CRAFTBUDDY_GPU_DEBUG__ = true` before the mod loads to opt in.
 * Every line is single-line and carries the same prefix so it can be grepped
 * out of the game console.
 */

import { emitDebugEvent } from '../../utils/debugSink';

const PREFIX = '[CraftBuddy][gpu]';

function readFlag(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    const holder = window as Window & { __CRAFTBUDDY_GPU_DEBUG__?: unknown };
    const value = holder.__CRAFTBUDDY_GPU_DEBUG__;
    return typeof value === 'boolean' ? value : false;
  } catch {
    return false;
  }
}

export const gpuDebugEnabled: boolean = readFlag();

function singleLine(parts: readonly unknown[]): string {
  return parts
    .map((part) => String(part).replace(/[\r\n\t]+/g, ' '))
    .join(' ');
}

export function gpuDebugLog(...parts: unknown[]): void {
  if (!gpuDebugEnabled) return;
  const line = `${PREFIX} ${singleLine(parts)}`;
  // eslint-disable-next-line no-console
  console.log(line);
  emitDebugEvent({ kind: 'gpu-log', line });
}
