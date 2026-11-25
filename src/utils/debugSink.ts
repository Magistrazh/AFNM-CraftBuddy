/**
 * Transport-neutral debug event sink.
 *
 * The mod renderer has no filesystem access, so diagnostics are handed to a
 * registered function as a single-line JSON string. Nothing here knows about
 * fetch, the local receiver, or the game API: a disabled sink is a no-op and
 * every entry point swallows errors so diagnostics can never break the mod.
 */

/** Longest JSON body passed to the sink; larger events are reduced or dropped. */
const MAX_BODY_LENGTH = 4000;
/** Strings over this length are shortened on the single allowed retry. */
const MAX_STRING_LENGTH = 400;
/** Guard against deep or cyclic payloads while walking an oversized event. */
const MAX_TRAVERSAL_DEPTH = 8;

/** Receives one compact, single-line JSON event. Must not be relied on to run. */
export type DebugSink = (body: string) => void;

export interface EmitDebugEventOptions {
  /**
   * Raise the 4000-byte cap for one deliberately large diagnostic (for example
   * a complete replay snapshot). Still bounded, still dropped if it does not
   * fit. Ordinary events keep the default cap.
   */
  maxBodyLength?: number;
}

let activeSink: DebugSink | undefined;

export function setDebugSink(fn?: DebugSink | null): void {
  activeSink = fn ?? undefined;
}

function truncateStrings(value: unknown, depth: number): unknown {
  if (typeof value === 'string') {
    return value.length > MAX_STRING_LENGTH
      ? `${value.slice(0, MAX_STRING_LENGTH)}...[truncated]`
      : value;
  }
  if (depth >= MAX_TRAVERSAL_DEPTH || value === null) return value;
  if (Array.isArray(value)) {
    return value.map((entry) => truncateStrings(entry, depth + 1));
  }
  if (typeof value === 'object') {
    const source = value as Record<string, unknown>;
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(source)) {
      result[key] = truncateStrings(source[key], depth + 1);
    }
    return result;
  }
  return value;
}

export function emitDebugEvent(
  event: Record<string, unknown>,
  options?: EmitDebugEventOptions,
): void {
  const sink = activeSink;
  if (!sink) return;
  const maxBodyLength = options?.maxBodyLength ?? MAX_BODY_LENGTH;

  try {
    let body = JSON.stringify(event);
    if (typeof body !== 'string') return;

    if (body.length > maxBodyLength) {
      const reduced = truncateStrings(event, 0);
      if (!reduced || typeof reduced !== 'object') return;
      body = JSON.stringify(reduced);
      if (typeof body !== 'string' || body.length > maxBodyLength) return;
    }

    sink(body);
  } catch {
    // Diagnostics are best-effort: never surface a failure to the mod.
  }
}
