/**
 * The complete replay snapshot must reach the local jsonl sink. The sink's
 * default 4000-byte cap silently drops a 100-300 KB snapshot, so the emitter
 * opts into a bounded larger body. This pins the event shape, its position
 * after the recommendation event, and the cap itself.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  buildOptimizerSnapshotDebugEvent,
  sanitizeForJson,
  type OptimizerReplaySnapshot,
} from '../modContent/replaySnapshot';
import { emitDebugEvent, setDebugSink } from '../utils/debugSink';

const SNAPSHOT_BODY_LIMIT = 2 * 1024 * 1024;

function makeSnapshot(): OptimizerReplaySnapshot {
  const fixture = JSON.parse(
    readFileSync(
      join(
        __dirname,
        '__fixtures__',
        'replay-snapshots',
        'user-report-pattern-step-1.snapshot.json',
      ),
      'utf8',
    ),
  ) as Record<string, unknown>;

  return {
    ...fixture,
    turn: {
      sequence: 7,
      step: 12,
      capturedAt: '2026-03-03T09:58:52.654Z',
      stateFingerprint: 'test-fingerprint',
      autoCraft: {
        policy: 'assist',
        armed: false,
        phase: 'idle',
        tone: 'neutral',
        statusTitle: '',
        statusDetail: '',
        canArm: true,
        canStop: false,
        isRunning: false,
        stopRequested: false,
      },
    },
  } as unknown as OptimizerReplaySnapshot;
}

describe('optimizerSnapshot debug event', () => {
  let bodies: string[] = [];

  beforeEach(() => {
    bodies = [];
    setDebugSink((body) => {
      bodies.push(body);
    });
  });

  afterEach(() => {
    setDebugSink(null);
  });

  it('emits the complete snapshot after the recommendation event', () => {
    const snapshot = makeSnapshot();
    const event = buildOptimizerSnapshotDebugEvent({
      data: sanitizeForJson(snapshot),
      json: JSON.stringify(snapshot, null, 2),
    });

    // Mirrors the index.ts call order: the recommendation event is emitted in
    // the try block, the snapshot in the finally block once output is present.
    emitDebugEvent({ kind: 'recommendation', pick: 'Forge' });
    emitDebugEvent(event, { maxBodyLength: SNAPSHOT_BODY_LIMIT });

    expect(bodies).toHaveLength(2);
    const recommendation = JSON.parse(bodies[0]) as Record<string, unknown>;
    const emitted = JSON.parse(bodies[1]) as Record<string, unknown>;

    expect(recommendation.kind).toBe('recommendation');
    expect(emitted.kind).toBe('optimizerSnapshot');
    expect(emitted.complete).toBe(true);
    expect(emitted.sequence).toBe(7);
    expect(emitted.step).toBe(12);
    expect(emitted.searchEpoch).toBe(44);
    expect(emitted.capturedAt).toBe('2026-03-03T09:58:52.654Z');
    expect(emitted.snapshotBytes).toBeGreaterThan(4000);

    const payload = emitted.snapshot as Record<string, unknown>;
    expect(payload.input).toBeDefined();
    const input = payload.input as Record<string, unknown>;
    expect(input.lookaheadDepth).toBe(24);
    expect(input.state).toBeDefined();
    expect(input.config).toBeDefined();
    expect(bodies[1].length).toBeGreaterThan(4000);
  });

  it('is dropped under the default cap, proving the raised bound is required', () => {
    const snapshot = makeSnapshot();
    const event = buildOptimizerSnapshotDebugEvent({
      data: sanitizeForJson(snapshot),
      json: JSON.stringify(snapshot, null, 2),
    });

    emitDebugEvent(event);

    expect(bodies).toHaveLength(0);
  });
});
