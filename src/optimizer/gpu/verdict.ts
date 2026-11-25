/**
 * S4: device-side survival verdict. Every input is a device word from the
 * state projection (or a named host input that the engine itself supplies to
 * lookahead.ts:364-399). The two booleans are computed here, not by calling
 * safety.ts in the stand.
 *
 * Input words (VERDICT_IN):
 *  0 currentStability, 1 displayStability, 2 floorStability, 3 floorMaxStability,
 *  4 survivalProbability, 5 goalsMetAfterAction, 6 baseSuccessSecuredAfterAction,
 *  7 hasGuaranteedSafeStabilize, 8 minimumGuaranteedContinuationStability,
 *  9 nearDeathStability, 10 expectedCompletion, 11 expectedPerfection,
 *  12 successCompletion, 13 successPerfection,
 *  14 failureCompletion, 15 failurePerfection, 16 failureStability,
 *  17 isItem, 18 targetCompletion, 19 targetPerfection, 20 displayEndsCraft,
 *  21 initialMaxStability.
 * Output words (VERDICT_OUT): requiresProbabilisticSurvival,
 *  hasFatalTechniqueFailure, uncollapsedSurvivalRisk, blendProbability.
 */

export const VERDICT_IN = {
  currentStability: 0,
  displayStability: 1,
  floorStability: 2,
  floorMaxStability: 3,
  survivalProbability: 4,
  goalsMetAfterAction: 5,
  baseSuccessSecuredAfterAction: 6,
  hasGuaranteedSafeStabilize: 7,
  minimumGuaranteedContinuationStability: 8,
  nearDeathStability: 9,
  expectedCompletion: 10,
  expectedPerfection: 11,
  successCompletion: 12,
  successPerfection: 13,
  failureCompletion: 14,
  failurePerfection: 15,
  failureStability: 16,
  isItem: 17,
  targetCompletion: 18,
  targetPerfection: 19,
  displayEndsCraft: 20,
  initialMaxStability: 21,
  WORDS: 22,
} as const;

export const VERDICT_OUT = {
  requires: 0,
  fatal: 1,
  uncollapsedRisk: 2,
  blendProbability: 3,
  WORDS: 4,
} as const;

export const VERDICT_WGSL = `
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;

fn goalsMet(completion: f32, perfection: f32, targetC: f32, targetP: f32) -> bool {
  let hasC = targetC > 0.0;
  let hasP = targetP > 0.0;
  if (!hasC && !hasP) { return false; }
  return (!hasC || completion >= targetC) && (!hasP || perfection >= targetP);
}

@compute @workgroup_size(64)
fn verdictKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&outputs) / ${VERDICT_OUT.WORDS}u) { return; }
  let b = i * ${VERDICT_IN.WORDS}u;
  let o = i * ${VERDICT_OUT.WORDS}u;
  let current = inputs[b + ${VERDICT_IN.currentStability}u];
  let display = inputs[b + ${VERDICT_IN.displayStability}u];
  let floorStability = inputs[b + ${VERDICT_IN.floorStability}u];
  let floorMax = inputs[b + ${VERDICT_IN.floorMaxStability}u];
  let pSurvive = inputs[b + ${VERDICT_IN.survivalProbability}u];
  let goalsAfter = inputs[b + ${VERDICT_IN.goalsMetAfterAction}u] > 0.5;
  let baseSecured = inputs[b + ${VERDICT_IN.baseSuccessSecuredAfterAction}u] > 0.5;
  let hasSafe = inputs[b + ${VERDICT_IN.hasGuaranteedSafeStabilize}u] > 0.5;
  let minSafe = inputs[b + ${VERDICT_IN.minimumGuaranteedContinuationStability}u];
  let nearDeath = inputs[b + ${VERDICT_IN.nearDeathStability}u];
  let targetC = inputs[b + ${VERDICT_IN.targetCompletion}u];
  let targetP = inputs[b + ${VERDICT_IN.targetPerfection}u];
  let displayEnds = inputs[b + ${VERDICT_IN.displayEndsCraft}u] > 0.5;

  // safety.ts:196-244. Expected/success are per-skill gain projections from
  // calculateSkillGains (safety.ts:209-221), not post-clamp bars; failure is
  // the successChance:0 applySkill transition (post-clamp bars).
  var fatal = false;
  if (inputs[b + ${VERDICT_IN.isItem}u] < 0.5) {
    let expectedC = inputs[b + ${VERDICT_IN.expectedCompletion}u];
    let expectedP = inputs[b + ${VERDICT_IN.expectedPerfection}u];
    let successC = inputs[b + ${VERDICT_IN.successCompletion}u];
    let successP = inputs[b + ${VERDICT_IN.successPerfection}u];
    if (expectedC < successC || expectedP < successP) {
      let failStability = inputs[b + ${VERDICT_IN.failureStability}u];
      let failGoals = goalsMet(
        inputs[b + ${VERDICT_IN.failureCompletion}u],
        inputs[b + ${VERDICT_IN.failurePerfection}u], targetC, targetP);
      fatal = failStability <= 0.0 && !failGoals;
    }
  }

  // safety.ts:144-194, exact predicate order.
  var useFloor = false;
  let gap = floorStability < display;
  let threshold = max(nearDeath, ceil(minSafe));
  if (!goalsAfter && current <= threshold && display > 0.0) {
    if (floorStability <= 0.0) {
      useFloor = hasSafe || !baseSecured;
    } else if (hasSafe && gap && floorStability < minSafe) {
      useFloor = !baseSecured;
    } else {
      useFloor = !baseSecured && current <= 1.0 && gap;
    }
  }
  let needFloor = fatal || useFloor;
  let risk = !needFloor && !displayEnds && floorStability <= 0.0 &&
    pSurvive > 0.0 && pSurvive < 1.0;
  outputs[o + ${VERDICT_OUT.requires}u] = select(0.0, 1.0, needFloor);
  outputs[o + ${VERDICT_OUT.fatal}u] = select(0.0, 1.0, fatal);
  outputs[o + ${VERDICT_OUT.uncollapsedRisk}u] = select(0.0, 1.0, risk);
  outputs[o + ${VERDICT_OUT.blendProbability}u] = select(0.0, pSurvive, risk);
}
`;

export async function createVerdictPipeline(
  device: GpuDevice,
): Promise<GpuComputePipeline> {
  const module = device.createShaderModule({ code: VERDICT_WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(`VERDICT WGSL: ${errors.map((e) => `${e.lineNum}:${e.message}`).join(' | ')}`);
  }
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'verdictKernel' },
  });
}

/** One verdict dispatch; the input array is count * VERDICT_IN.WORDS f32. */
export async function runVerdictBatch(
  device: GpuDevice,
  pipeline: GpuComputePipeline,
  inputs: Float32Array,
): Promise<Float32Array> {
  const count = inputs.length / VERDICT_IN.WORDS;
  const inputBuf = device.createBuffer({
    size: Math.max(16, inputs.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, inputs);
  const outBuf = device.createBuffer({
    size: Math.max(16, count * VERDICT_OUT.WORDS * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: Math.max(16, count * VERDICT_OUT.WORDS * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readback, 0, count * VERDICT_OUT.WORDS * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    return new Float32Array(
      readback.getMappedRange().slice(0, count * VERDICT_OUT.WORDS * 4),
    );
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
    outBuf.destroy();
    inputBuf.destroy();
  }
}
