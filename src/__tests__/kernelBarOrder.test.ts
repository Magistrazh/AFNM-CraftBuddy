import { STATE_KERNEL_WGSL } from '../optimizer/gpu/kernelState';

/**
 * skills.ts:3105-3116 — `expectedProgressGain` floors the critted (and boosted)
 * bar gain BEFORE the success chance is applied, then floors again:
 *
 *   safeFloor(clampPredictedProgressGain(safeFloor(gainWithCrit)) * expectedFactor)
 *
 * The kernel used to fold the success chance into the effect weight
 * (`skills[e + E_CHANCE] * appliedSuccess`) and floor once at the end, which
 * moves the floor boundary at fractional bar gains. `state-kernel` numbers from
 * the depth-2 stand (forge-heat-runway-step-2#6, explosive_refinement, success
 * 0.4, raw 471, critFactor 1.099) show the two orders disagreeing by one point.
 */
describe('state kernel bar gain order', () => {
  // Real values from the probe trace (slot 60/2) of a failing depth-2 case.
  const raw = 471;
  const critFactor = 1.099;
  const success = 0.4;

  const engineOrder = Math.floor(Math.floor(raw * critFactor) * success);
  const weightFirstOrder = Math.floor(raw * success * critFactor);

  it('floors the critted gain before the success weighting', () => {
    expect(weightFirstOrder).toBe(207);
    expect(engineOrder).toBe(206);
    expect(engineOrder).not.toBe(weightFirstOrder);
  });

  it('passes the unweighted bar amount into the crit/boost/floor stage', () => {
    // skills.ts:3235-3262 sums bar effects with the condition factor only; the
    // success chance is the expected-value factor applied afterwards.
    expect(STATE_KERNEL_WGSL).toContain(
      'let weight = select(\n      skills[e + E_CHANCE] * appliedWeight,\n      skills[e + E_CHANCE],\n      barEffect);',
    );
    // The applied weight is the success roll, overridden only by the fatal
    // probe (safety.ts:229-238); pin that structure so a rename cannot hide
    // a semantic change.
    expect(STATE_KERNEL_WGSL).toContain('var appliedWeight = appliedSuccess;');
    expect(STATE_KERNEL_WGSL).toContain('if (forceSuccess == 1u) { appliedWeight = 1.0; }');
    expect(STATE_KERNEL_WGSL).toContain('else if (forceSuccess == 2u) { appliedWeight = 0.0; }');
  });

  it('clamps the floored bar gain to the headroom before the success weight', () => {
    // skills.ts:3078-3093 clamps `min(gain, cap - current)` BEFORE the success
    // factor; weighting first would erase the failure risk of an overshoot.
    expect(STATE_KERNEL_WGSL).toContain(
      'boostedBar(barCompRaw * finalFrame.critFactor,\n    finalFrame.boostCompletion, true)',
    );
    expect(STATE_KERNEL_WGSL).toContain(
      'boostedBar(barPerfRaw * finalFrame.critFactor,\n    finalFrame.boostPerfection, true)',
    );
    expect(STATE_KERNEL_WGSL).toContain('if (mainComp > 0.0) { mainComp = min(mainComp, mainCompHeadroom); }');
    expect(STATE_KERNEL_WGSL).toContain('dCompletion = dCompletion + floor(mainComp * appliedWeight);');
    expect(STATE_KERNEL_WGSL).toContain('dPerfection = dPerfection + floor(mainPerf * appliedWeight);');
  });

  it('snaps an f32 crit product that lands one ulp below an integer', () => {
    // user-report-focused-fusion-prereq#99/purifying_intensity: the engine
    // computes 1500 * 1.096 in doubles (1644), the kernel's f32 factor is
    // 1.095999956 and the plain floor returned 1643.
    const f32Crit = Math.fround(1.096);
    expect(1500 * f32Crit).toBeLessThan(1644);
    expect(Math.floor(1500 * f32Crit + Math.max(1e-7, 1500 * 1e-7))).toBe(1644);
    expect(STATE_KERNEL_WGSL).toContain('if (floorUnboosted) { return floor(critted + max(1.0e-7, abs(critted) * 1.0e-7)); }');
  });

  it('resolves the current action mastery upgrade on held buff effects', () => {
    // user-report-fairy-recovery#96/rapid_fusion: the delayed_fusion buff has
    // upgradeKey "completion" and must inherit Rapid Fusion's +15% upgrade
    // (skills.ts:5434-5439). The catalog is shared, so the WGSL matches a
    // stored key hash against the active skill's upgrade mastery records.
    expect(STATE_KERNEL_WGSL).toContain('skills[m + M_UPGRADE_KEY_HASH] == upgradeHash');
    expect(STATE_KERNEL_WGSL).toContain('v = (v + skills[m + M_UPGRADE_ADD]) * skills[m + M_UPGRADE_MUL];');
  });
});
