import {
  AVAIL_OUT,
  AVAIL_PARAM,
  AVAIL_REASON,
  AVAIL_SKILL,
  AVAILABILITY_KERNEL_WGSL,
  CONDITION_KIND_NAMES,
  REASON_NAMES,
  conditionKindCode,
  conditionNameOfKind,
  reasonCodeOf,
} from '../optimizer/gpu/availability';
import { CONDITION_KERNEL_WGSL } from '../optimizer/gpu/kernelConditions';

describe('S3 availability kernel contract', () => {
  it('maps every engine blocked-reason category and fails closed on unknown', () => {
    expect(reasonCodeOf('stability')).toBe(AVAIL_REASON.stability);
    expect(reasonCodeOf('cooldown')).toBe(AVAIL_REASON.cooldown);
    expect(reasonCodeOf('condition')).toBe(AVAIL_REASON.condition);
    expect(reasonCodeOf('qi')).toBe(AVAIL_REASON.qi);
    expect(reasonCodeOf('toxicity')).toBe(AVAIL_REASON.toxicity);
    expect(reasonCodeOf('unexpected')).toBe(AVAIL_REASON.none);
    expect(REASON_NAMES).toHaveLength(6);
    for (const [name, code] of Object.entries(AVAIL_REASON)) {
      expect(REASON_NAMES[code]).toBe(name);
    }
  });

  it('uses canonical condition kinds, including game aliases', () => {
    expect(conditionKindCode('neutral')).toBe(0);
    expect(conditionKindCode('balanced')).toBe(0);
    expect(conditionKindCode('harmonious')).toBe(conditionKindCode('positive'));
    expect(conditionKindCode('resistant')).toBe(conditionKindCode('negative'));
    expect(conditionKindCode('brilliant')).toBe(conditionKindCode('veryPositive'));
    expect(conditionKindCode('corrupted')).toBe(conditionKindCode('veryNegative'));
    expect(CONDITION_KIND_NAMES[conditionKindCode('excellent')]).toBe('veryPositive');
    expect(conditionNameOfKind(99)).toBe('other');
  });

  it('exposes named native-precheck and host cost inputs in the parameter table', () => {
    expect(AVAIL_PARAM.nativePrecheck).not.toBe(AVAIL_PARAM.nativePrecheckPresent);
    expect(AVAIL_PARAM.effectiveQiCost).not.toBe(AVAIL_PARAM.maxToxicity);
    expect(AVAIL_PARAM.currentConditionKind).toBe(AVAIL_PARAM.WORDS - 1);
    expect(AVAIL_OUT.WORDS).toBe(4);
  });

  it('contains the availability entry point and the diagnostic asymmetry', () => {
    expect(AVAILABILITY_KERNEL_WGSL).toContain('fn availability(');
    expect(AVAILABILITY_KERNEL_WGSL).toContain('buffStacks(nodeBase');
    expect(AVAILABILITY_KERNEL_WGSL).toContain('P_NATIVE_PRESENT');
    // The engine's reason list deliberately omits droplet/buff gates.
    expect(AVAILABILITY_KERNEL_WGSL).toContain('does NOT check');
    expect(AVAIL_SKILL.WORDS).toBe(11);
  });

  it('matches the f64 branch-threshold decision at the 0.15 boundary', () => {
    // conditions.ts:165-168 uses >= against an f64 product; f32 arithmetic at
    // harmony -70 puts the positive branch one ulp below the threshold.
    expect(CONDITION_KERNEL_WGSL).toContain('(*d).prob[i] + 1.0e-7 >= minProb');
    expect(CONDITION_KERNEL_WGSL).toContain('conditions.ts:165-168');
  });
});
