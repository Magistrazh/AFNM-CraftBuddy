/**
 * S7d step 8: device-side SCORE_INPUT assembly.
 *
 * Replaces the host `scoreInputFromNode` + `buildScoreWords` call for every
 * score case. Immutable harmonyData lives in `style` (one upload per search);
 * evolving scalars are read from the NS_STATE node. No per-case host input is
 * constructed. The numeric mapping is byte-for-byte the one in
 * `kernelScore.ts:253-323`; the score kernel itself is unchanged.
 */
import { NS_STATE } from './records';
import { SCORE_INPUT } from './kernelScore';

export const STYLE = {
  kind: 0,
  inscriptionStacks: 1,
  inscribedRemaining: 2,
  recommendedCount: 3,
  recommended: 4, // 8 words
  resonanceType: 12,
  pendingResonanceType: 13,
  pendingCount: 14,
  cadenceChain: 15,
  alchemicalModsFinite: 16,
  alchemicalControlMult: 17,
  alchemicalIntensityMult: 18,
  alchemicalCritBonus: 19,
  alchemicalSuccessBonus: 20,
  alchemicalQiReduction: 21,
  alchemicalStabReduction: 22,
  forgeHeat: 23,
  forgeLastBuffedHeat: 24,
  alchemicalCharges: 25,
  resonanceStrength: 26,
  WORDS: 27,
} as const;

const w = (name: keyof typeof SCORE_INPUT): number => SCORE_INPUT[name] as number;
const n = (name: keyof typeof NS_STATE): number => NS_STATE[name] as number;

export const SCORE_INPUT_KERNEL_WGSL = `
@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read> style: array<f32>;
@group(0) @binding(2) var<storage, read> stabilityOverride: array<f32>;
@group(0) @binding(3) var<storage, read_write> inputs: array<f32>;

@compute @workgroup_size(64)
fn scoreInputKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&stabilityOverride)) { return; }
  let s = i * ${STYLE.WORDS}u;
  let nd = i * ${NS_STATE.WORDS}u;
  let b = i * ${SCORE_INPUT.WORDS}u;
  inputs[b + ${w('completion')}u] = nodes[nd + ${n('completion')}u];
  inputs[b + ${w('perfection')}u] = nodes[nd + ${n('perfection')}u];
  inputs[b + ${w('qi')}u] = nodes[nd + ${n('pool')}u];
  inputs[b + ${w('stability')}u] = stabilityOverride[i];
  inputs[b + ${w('step')}u] = nodes[nd + ${n('step')}u];
  inputs[b + ${w('controlBuffTurns')}u] = nodes[nd + ${n('controlBuffTurns')}u];
  inputs[b + ${w('controlBuffMultiplier')}u] = nodes[nd + ${n('controlBuffMultiplier')}u];
  inputs[b + ${w('intensityBuffTurns')}u] = nodes[nd + ${n('intensityBuffTurns')}u];
  inputs[b + ${w('intensityBuffMultiplier')}u] = nodes[nd + ${n('intensityBuffMultiplier')}u];
  inputs[b + ${w('toxicity')}u] = nodes[nd + ${n('toxicity')}u];
  inputs[b + ${w('maxToxicity')}u] = nodes[nd + ${n('maxToxicity')}u];
  inputs[b + ${w('harmony')}u] = nodes[nd + ${n('harmony')}u];
  inputs[b + ${w('conditionId')}u] = nodes[nd + ${n('conditionId')}u];
  let kind = style[s + ${STYLE.kind}u];
  if (kind > 0.5 && kind < 1.5) {
    inputs[b + ${w('harmonyKind')}u] = 1.0;
    let heat = style[s + ${STYLE.forgeHeat}u];
    inputs[b + ${w('forgeHeat')}u] = heat;
    inputs[b + ${w('forgeLastBuffedHeat')}u] = style[s + ${STYLE.forgeLastBuffedHeat}u];
  } else if (kind > 1.5 && kind < 2.5) {
    inputs[b + ${w('harmonyKind')}u] = 2.0;
    inputs[b + ${w('inscribedStacks')}u] = style[s + ${STYLE.inscriptionStacks}u];
    inputs[b + ${w('inscribedRemaining')}u] = style[s + ${STYLE.inscribedRemaining}u];
  } else if (kind > 2.5 && kind < 3.5) {
    inputs[b + ${w('harmonyKind')}u] = 3.0;
    inputs[b + ${w('alchemicalCharges')}u] = style[s + ${STYLE.alchemicalCharges}u];
    inputs[b + ${w('alchemicalModsFinite')}u] = style[s + ${STYLE.alchemicalModsFinite}u];
    inputs[b + ${w('alchemicalControlMult')}u] = style[s + ${STYLE.alchemicalControlMult}u];
    inputs[b + ${w('alchemicalIntensityMult')}u] = style[s + ${STYLE.alchemicalIntensityMult}u];
    inputs[b + ${w('alchemicalCritBonus')}u] = style[s + ${STYLE.alchemicalCritBonus}u];
    inputs[b + ${w('alchemicalSuccessBonus')}u] = style[s + ${STYLE.alchemicalSuccessBonus}u];
    inputs[b + ${w('alchemicalQiReduction')}u] = style[s + ${STYLE.alchemicalQiReduction}u];
    inputs[b + ${w('alchemicalStabReduction')}u] = style[s + ${STYLE.alchemicalStabReduction}u];
    let rc = u32(style[s + ${STYLE.recommendedCount}u]);
    for (var r: u32 = 0u; r < rc; r = r + 1u) {
      inputs[b + ${w('recommended')}u + r] = style[s + ${STYLE.recommended}u + r];
    }
  } else if (kind > 3.5 && kind < 4.5) {
    inputs[b + ${w('harmonyKind')}u] = 4.0;
    inputs[b + ${w('resonanceStrength')}u] = style[s + ${STYLE.resonanceStrength}u];
    inputs[b + ${w('resonanceType')}u] = style[s + ${STYLE.resonanceType}u];
    inputs[b + ${w('pendingResonanceType')}u] = style[s + ${STYLE.pendingResonanceType}u];
    inputs[b + ${w('pendingCount')}u] = style[s + ${STYLE.pendingCount}u];
  } else if (kind > 4.5 && kind < 5.5) {
    inputs[b + ${w('harmonyKind')}u] = 5.0;
    inputs[b + ${w('cadenceChain')}u] = style[s + ${STYLE.cadenceChain}u];
  }
}
`;

export async function createScoreInputPipeline(
  device: GpuDevice,
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: SCORE_INPUT_KERNEL_WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    throw new Error(`SCORE_INPUT WGSL: ${errors.map((e) => `${e.lineNum}:${e.message}`).join(' | ')}`);
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'scoreInputKernel' },
  });
}

/** One dispatch building all per-case SCORE_INPUT words from device nodes. */
export function createScoreInputBindGroup(
  device: GpuDevice,
  pipeline: GpuComputePipeline,
  nodes: GpuBuffer,
  style: GpuBuffer,
  stability: GpuBuffer,
  inputs: GpuBuffer,
): GpuBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: nodes } },
      { binding: 1, resource: { buffer: style } },
      { binding: 2, resource: { buffer: stability } },
      { binding: 3, resource: { buffer: inputs } },
    ],
  });
}
