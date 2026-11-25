/**
 * Flat timing profile for one GPU search.
 *
 * No dependency on GPU types: this is pure wall-clock measurement around the
 * phases in search.ts. `performance.now()` is used when available (game and
 * browser), with a Date.now() fallback for Node-based tooling.
 */

export interface GpuProfile {
  readonly recordsMs: number;
  readonly deviceInitMs: number;
  readonly uploadMs: number;
  readonly dispatchMs: number;
  readonly readbackMs: number;
  readonly totalMs: number;
  readonly levels: number;
  readonly dispatches: number;
  readonly bufferBytes: number;
  readonly skillCount: number;
  readonly nodeCount: number;
}

/** Named phase boundaries used by search.ts. */
export type GpuProfileMark =
  | 'start'
  | 'records'
  | 'device'
  | 'upload'
  | 'dispatch'
  | 'readback';

export interface GpuProfileShape {
  readonly levels: number;
  readonly skillCount: number;
  readonly nodeCount: number;
}

const defaultNow = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();

/**
 * Tiny profiler: named marks, additive counters, one flat record on finish().
 * Missing marks yield 0 for that phase rather than NaN, so partial runs (device
 * error, timeout) still produce a usable record.
 */
export class GpuProfiler {
  private readonly startedAt: number;
  private readonly marks = new Map<string, number>();
  private dispatchCount = 0;
  private bufferByteCount = 0;
  private shape: GpuProfileShape = { levels: 0, skillCount: 0, nodeCount: 0 };

  constructor(private readonly now: () => number = defaultNow) {
    this.startedAt = now();
    this.marks.set('start', this.startedAt);
  }

  mark(name: GpuProfileMark | string): void {
    this.marks.set(name, this.now());
  }

  elapsedSince(name: GpuProfileMark | string): number {
    const at = this.marks.get(name);
    return at === undefined ? 0 : Math.max(0, this.now() - at);
  }

  addDispatches(count: number): void {
    if (Number.isFinite(count) && count > 0) this.dispatchCount += count;
  }

  addBufferBytes(bytes: number): void {
    if (Number.isFinite(bytes) && bytes > 0) this.bufferByteCount += bytes;
  }

  setShape(shape: GpuProfileShape): void {
    this.shape = shape;
  }

  finish(): GpuProfile {
    const phase = (from: GpuProfileMark, to: GpuProfileMark): number => {
      const a = this.marks.get(from);
      const b = this.marks.get(to);
      return a !== undefined && b !== undefined && b >= a ? b - a : 0;
    };
    return {
      recordsMs: phase('start', 'records'),
      deviceInitMs: phase('records', 'device'),
      uploadMs: phase('device', 'upload'),
      dispatchMs: phase('upload', 'dispatch'),
      readbackMs: phase('dispatch', 'readback'),
      totalMs: Math.max(0, this.now() - this.startedAt),
      levels: this.shape.levels,
      dispatches: this.dispatchCount,
      bufferBytes: this.bufferByteCount,
      skillCount: this.shape.skillCount,
      nodeCount: this.shape.nodeCount,
    };
  }
}

function ms(value: number): string {
  return `${Number.isFinite(value) ? value.toFixed(2) : '0.00'}ms`;
}

function kib(value: number): string {
  return `${Number.isFinite(value) ? (value / 1024).toFixed(1) : '0.0'}KiB`;
}

function int(value: number): string {
  return Number.isFinite(value) ? String(Math.round(value)) : '0';
}

/** One-line, grep-able summary for the debug log and diagnostics channel. */
export function describeProfile(profile: GpuProfile): string {
  return (
    `records=${ms(profile.recordsMs)} device=${ms(profile.deviceInitMs)} ` +
    `upload=${ms(profile.uploadMs)} dispatch=${ms(profile.dispatchMs)} ` +
    `readback=${ms(profile.readbackMs)} total=${ms(profile.totalMs)} ` +
    `levels=${int(profile.levels)} dispatches=${int(profile.dispatches)} ` +
    `buffers=${kib(profile.bufferBytes)} skills=${int(profile.skillCount)} ` +
    `nodes=${int(profile.nodeCount)}`
  );
}
