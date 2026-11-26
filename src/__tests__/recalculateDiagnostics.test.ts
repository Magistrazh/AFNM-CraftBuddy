/**
 * Event shapes for the panel's "Update plan" diagnostics. No GPU and no
 * integration module import: the helpers are pure emitters over debugSink.
 */

import {
  emitRecalculateResult,
  emitRecalculateStart,
} from '../modContent/recalculateDiagnostics';
import { setDebugSink } from '../utils/debugSink';

const startFields = {
  searchEngine: 'gpu',
  lookaheadDepth: 48,
  timeBudgetMs: 2250,
  maxNodes: 800000,
  beamWidth: 5,
  step: 7,
  condition: 'positive',
};

describe('recalculate diagnostics', () => {
  let events: Array<Record<string, unknown>> = [];

  beforeEach(() => {
    events = [];
    setDebugSink((body) => {
      events.push(JSON.parse(body) as Record<string, unknown>);
    });
  });

  afterEach(() => {
    setDebugSink(null);
  });

  it('emits a start event with the effective search settings and craft position', () => {
    emitRecalculateStart({ ...startFields, ok: true });

    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({
      kind: 'recalculate',
      trigger: 'settings',
      ok: true,
      ...startFields,
    });
    expect(Object.keys(events[0]).sort()).toEqual(
      [
        'beamWidth',
        'condition',
        'kind',
        'lookaheadDepth',
        'maxNodes',
        'ok',
        'searchEngine',
        'step',
        'timeBudgetMs',
        'trigger',
      ].sort(),
    );
  });

  it('emits an explicit no-craft failure without hiding the settings', () => {
    emitRecalculateStart({
      ...startFields,
      step: null,
      condition: null,
      ok: false,
      reason: 'no craft',
    });

    expect(events[0]).toEqual({
      kind: 'recalculate',
      trigger: 'settings',
      ok: false,
      reason: 'no craft',
      ...startFields,
      step: null,
      condition: null,
    });
  });

  it('emits a result event from the real recommendation outcome', () => {
    emitRecalculateResult({
      ok: true,
      engine: 'gpu',
      pick: 'Приём',
      score: 123.5,
      elapsedMs: 42,
    });

    expect(events[0]).toEqual({
      kind: 'recalculateResult',
      trigger: 'settings',
      ok: true,
      engine: 'gpu',
      pick: 'Приём',
      score: 123.5,
      elapsedMs: 42,
    });
  });

  it('emits a result event with a reason when the search fails', () => {
    emitRecalculateResult({
      ok: false,
      engine: 'gpu',
      pick: null,
      score: null,
      reason: 'superseded',
      elapsedMs: 3,
    });

    expect(events[0]).toEqual({
      kind: 'recalculateResult',
      trigger: 'settings',
      ok: false,
      engine: 'gpu',
      pick: null,
      score: null,
      reason: 'superseded',
      elapsedMs: 3,
    });
  });
});
