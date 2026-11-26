/**
 * Live diagnostics transport: POSTs compact JSON events to a local receiver.
 *
 * Disabled by default for release. Opt in before the mod loads with
 * `window.__CRAFTBUDDY_DEBUG_SINK__ = true` (the local sink) or a URL string
 * (any other string overrides the base URL). When the receiver is down, every
 * POST fails quietly and the mod carries on unchanged.
 */

import { emitDebugEvent, setDebugSink } from '../utils/debugSink';

const DEFAULT_SINK_URL = 'http://127.0.0.1:43618';

export interface DebugBootContext {
  readonly version: string;
  readonly gpuDebug: boolean;
  readonly searchEngine: string;
}

let initialized = false;

export function initDebugSink(context: DebugBootContext): void {
  if (initialized) return;
  initialized = true;

  let configured: unknown;
  try {
    if (typeof window === 'undefined') return;
    const holder = window as Window & {
      __CRAFTBUDDY_DEBUG_SINK__?: unknown;
    };
    configured = holder.__CRAFTBUDDY_DEBUG_SINK__;
  } catch {
    return;
  }

  let sinkUrl: string | null = null;
  if (
    configured === true ||
    (typeof configured === 'string' && configured.trim().length > 0)
  ) {
    const base =
      typeof configured === 'string' && configured.trim().length > 0
        ? configured.replace(/\/+$/, '')
        : DEFAULT_SINK_URL;
    sinkUrl = base;

    setDebugSink((body) => {
      try {
        void fetch(`${base}/log`, {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body,
          // Fetch rejects keepalive bodies over 64 KiB with a network error.
          // Large snapshots must still be delivered while the page is alive.
          keepalive: body.length <= 64 * 1024,
        }).catch(() => {
          // Local receiver absent: drop the event, keep the game running.
        });
      } catch {
        // No fetch in this environment: diagnostics stay silent.
      }
    });
  }

  emitDebugEvent({
    kind: 'boot',
    version: context.version,
    gpuDebug: context.gpuDebug,
    sinkUrl,
    searchEngine: context.searchEngine,
    at: new Date().toISOString(),
  });
}
