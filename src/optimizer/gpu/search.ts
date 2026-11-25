/**
 * WebGPU search backend (S1: no UI wiring).
 *
 * Ported from the device/buffer/dispatch harness in
 * tools/gpu-probe/stage7.html. One command buffer per search, one staging
 * map per search, no host round-trips between beam levels.
 */

import { probeCapability, type GpuCapability } from './capability';
import { gpuDebugLog } from './debug';
import { buildKernelSource } from './kernel';
import {
  GpuProfiler,
  describeProfile,
  type GpuProfile,
} from './profile';
import {
  NODE_ROOT,
  NS,
  OUT,
  buildGpuRecords,
  type GpuRecordSet,
} from './records';
import type { CraftingState } from '../state';
import type { OptimizerConfig, SkillDefinition } from '../skills';

const WORKGROUP = 64;

export interface GpuSkillDelta {
  readonly name: string;
  readonly dQi: number;
  readonly dStability: number;
  readonly dMaxStability: number;
  readonly dPerfection: number;
  readonly dCompletion: number;
}

export interface GpuTiming {
  readonly encodeMs: number;
  readonly gpuMs: number;
  readonly readbackMs: number;
  readonly totalMs: number;
}

export interface GpuRootRank {
  readonly name: string;
  readonly score: number;
  readonly nodes: number;
  readonly completion: number;
  readonly perfection: number;
}

export interface GpuBeamSummary {
  readonly nodes: number;
  readonly alive: number;
  readonly steps: number;
  readonly completion: number;
  readonly perfection: number;
  readonly roots: GpuRootRank[];
}

/** A replayable skill sequence extracted from one beam search. */
export interface GpuPath {
  readonly skills: string[];
  readonly score: number;
  readonly completion: number;
  readonly perfection: number;
  readonly stability: number;
}

/** Default number of replayable paths returned by one search. */
export const DEFAULT_PATH_LIMIT = 16;

export interface GpuSearchResult {
  readonly deltas: GpuSkillDelta[];
  readonly timing: GpuTiming;
  readonly profile: GpuProfile;
  readonly notes: Record<string, number>;
  readonly beam?: GpuBeamSummary;
  /** Empty when depth is 0 (no beam was run). */
  readonly paths: GpuPath[];
}

export interface GpuSearchOptions {
  readonly currentCondition?: string;
  /** Beam replication count. 1 = score-only S1 probe. */
  readonly beamWidth?: number;
  readonly topK?: number;
  /** Beam levels; 0 skips the beam kernels entirely. */
  readonly depth?: number;
  readonly timeoutMs?: number;
  /**
   * Whether a timeout disables the backend for the rest of the session.
   * Auxiliary calls (root ranking) pass false: a slow ranking must degrade
   * that one search, not kill the feature until a restart.
   */
  readonly disableOnTimeout?: boolean;
}

interface Pipelines {
  readonly score: GpuComputePipeline;
  readonly reset: GpuComputePipeline;
  readonly select: GpuComputePipeline;
  readonly top: GpuComputePipeline;
  readonly apply: GpuComputePipeline;
}

interface DeviceContext {
  readonly device: GpuDevice;
  readonly capability: GpuCapability;
  readonly pipelines: Record<string, Pipelines>;
}


let context: DeviceContext | null = null;
let pending: Promise<DeviceContext> | null = null;
let disabledReason: string | null = null;
let lastDeviceLoss: string | null = null;

const now = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();

function workgroups(count: number): number {
  return Math.max(1, Math.ceil(count / WORKGROUP));
}

/** Safety-net switch used by the timeout/error paths (and by the debug object). */
export function disableGpuBackend(reason: string): void {
  disabledReason = reason;
  context = null;
  pending = null;
}

export function resetGpuBackend(): void {
  disabledReason = null;
  lastDeviceLoss = null;
  context = null;
  pending = null;
}

export function gpuBackendStatus(): {
  active: boolean;
  disabledReason: string | null;
  lastDeviceLoss: string | null;
} {
  return {
    active: context !== null,
    disabledReason,
    lastDeviceLoss,
  };
}

function makeBuffer(device: GpuDevice, data: Float32Array, usage: number): GpuBuffer {
  const buffer = device.createBuffer({
    size: Math.max(16, data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

async function createPipelines(
  device: GpuDevice,
  dimensions: { beamWidth: number; skillCount: number; topK: number },
): Promise<Pipelines> {
  const module = device.createShaderModule({ code: buildKernelSource(dimensions) });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    throw new Error(
      `WGSL: ${errors.map((error) => `${error.lineNum}:${error.message}`).join(' | ')}`,
    );
  }
  const make = (entryPoint: string): Promise<GpuComputePipeline> =>
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint },
    });
  const [score, reset, select, top, apply] = await Promise.all([
    make('scoreKernel'),
    make('resetKernel'),
    make('selectKernel'),
    make('selectTopKernel'),
    make('applyKernel'),
  ]);
  return { score, reset, select, top, apply };
}

async function createContext(): Promise<DeviceContext> {
  const capability = await probeCapability();
  if (!capability.available) throw new Error(capability.reason);
  if (!navigator.gpu) {
    const reason = 'WebGPU недоступен: navigator.gpu исчез';
    gpuDebugLog(`error phase=context message=${reason}`);
    throw new Error(reason);
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    const reason = 'WebGPU недоступен: адаптер не найден';
    gpuDebugLog(`error phase=context message=${reason}`);
    throw new Error(reason);
  }
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (event: unknown) => {
    const raw = objEventMessage(event);
    gpuDebugLog(`error phase=device message=${raw}`);
    disableGpuBackend(`WebGPU-ошибка: ${raw}`);
  });
  const fresh: DeviceContext = { device, capability, pipelines: {} };
  device.lost.then((info) => {
    if (context === fresh) {
      context = null;
      lastDeviceLoss = `${info.reason}: ${info.message}`;
      gpuDebugLog(`device-loss reason=${info.reason} message=${info.message}`);
    }
  });
  context = fresh;
  return fresh;
}

function objEventMessage(event: unknown): string {
  if (!event || typeof event !== 'object') return String(event);
  const holder = event as { error?: { message?: unknown }; message?: unknown };
  return String(holder.error?.message ?? holder.message ?? event);
}

async function getContext(): Promise<DeviceContext> {
  if (disabledReason !== null) throw new Error(disabledReason);
  if (context !== null) return context;
  if (pending !== null) return pending;
  pending = createContext().then(
    (created) => {
      pending = null;
      return created;
    },
    (error: unknown) => {
      pending = null;
      throw error;
    },
  );
  return pending;
}

function bind(
  device: GpuDevice,
  pipeline: GpuComputePipeline,
  entries: Array<{ binding: number; buffer: GpuBuffer }>,
): GpuBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: entries.map(({ binding, buffer }) => ({ binding, resource: { buffer } })),
  });
}

function dispatch(
  encoder: GpuCommandEncoder,
  pipeline: GpuComputePipeline,
  group: GpuBindGroup,
  count: number,
): void {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(count);
  pass.end();
}

function decodeFloats(bytes: Uint8Array, offset: number, length: number): Float32Array {
  const copy = new Uint8Array(length);
  copy.set(bytes.subarray(offset, offset + length));
  return new Float32Array(copy.buffer);
}

function mapDeltas(values: Float32Array, skillCount: number, names: string[]): GpuSkillDelta[] {
  const out: GpuSkillDelta[] = [];
  for (let a = 0; a < skillCount; a++) {
    const base = a * OUT;
    out.push({
      name: names[a] ?? String(a),
      dQi: values[base],
      dStability: values[base + 1],
      dMaxStability: values[base + 2],
      dPerfection: values[base + 3],
      dCompletion: values[base + 4],
    });
  }
  return out;
}

function summarizeBeam(values: Float32Array, nodeCount: number): Omit<GpuBeamSummary, 'roots'> {
  let alive = 0;
  let steps = 0;
  let completion = 0;
  let perfection = 0;
  for (let n = 0; n < nodeCount; n++) {
    const base = n * NS;
    if (values[base + NODE_ALIVE] >= 0.5) alive++;
    steps += values[base + NODE_STEP];
    completion += values[base + 13];
    perfection += values[base + 14];
  }
  return {
    nodes: nodeCount,
    alive,
    steps: nodeCount > 0 ? steps / nodeCount : 0,
    completion: nodeCount > 0 ? completion / nodeCount : 0,
    perfection: nodeCount > 0 ? perfection / nodeCount : 0,
  };
}

/**
 * Group final beam nodes by the opening skill traced into slot 25.
 *
 * Contract: only alive nodes compete for a root's best entry; every node with
 * an in-range root id is counted, alive or dead. Out-of-range ids are dropped.
 * A root whose nodes are all dead is therefore absent (it has no winner).
 */
export function summarizeRoots(
  values: Float32Array,
  nodeCount: number,
  skills: readonly { name: string }[],
): GpuRootRank[] {
  const counted = new Map<number, number>();
  const best = new Map<number, { score: number; completion: number; perfection: number }>();
  for (let n = 0; n < nodeCount; n++) {
    const base = n * NS;
    const root = Math.round(values[base + NODE_ROOT]);
    if (!Number.isFinite(root) || root < 0 || root >= skills.length) continue;
    counted.set(root, (counted.get(root) ?? 0) + 1);
    if (values[base + NODE_ALIVE] < 0.5) continue;
    const completion = values[base + 13];
    const perfection = values[base + 14];
    // The beam's own heuristic (scoreOf in kernel.ts), NOT the engine's
    // scoreState. It only orders roots inside this GPU search.
    const score = completion * 3 + perfection + values[base + 1] * 0.05;
    const previous = best.get(root);
    if (!previous || score > previous.score) {
      best.set(root, { score, completion, perfection });
    }
  }
  const roots: GpuRootRank[] = [];
  // Map.forEach, not for..of: the build targets ES5 without downlevelIteration.
  best.forEach((value, root) => {
    roots.push({
      name: skills[root]?.name ?? String(root),
      score: value.score,
      nodes: counted.get(root) ?? 0,
      completion: value.completion,
      perfection: value.perfection,
    });
  });
  roots.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
  });
  return roots;
}

const NODE_SIZE = 38;
const NODE_STEP = NODE_SIZE;
const NODE_ALIVE = NODE_SIZE + 1;

/**
 * Turn the GPU replay trace into top skill sequences for the JS engine.
 *
 * Trace layout (floats): entry `(level * beamWidth + node) * 2` holds the
 * parent node index, `+1` the applied skill index. Entries are written by
 * applyKernel for every node of every level, including dead ones.
 *
 * Ranking uses the beam heuristic (scoreOf in kernel.ts), not scoreState.
 * A surviving final node has `step === depth`: applyKernel increments the step
 * only when it actually applies a skill, so any dead ancestor leaves it short.
 * That check, plus the alive flag, filters dead paths before the back-walk.
 */
export function reconstructPaths(
  trace: Float32Array,
  nodes: Float32Array,
  depth: number,
  beamWidth: number,
  skills: readonly { name: string }[],
  limit: number,
): GpuPath[] {
  if (
    !Number.isFinite(depth) || depth <= 0 ||
    !Number.isFinite(beamWidth) || beamWidth <= 0 ||
    !Number.isFinite(limit) || limit <= 0 ||
    skills.length === 0
  ) {
    return [];
  }
  const levels = Math.floor(depth);
  const width = Math.floor(beamWidth);
  const maxPaths = Math.floor(limit);

  const toIndex = (value: number, maxExclusive: number): number => {
    if (!Number.isFinite(value) || value < 0 || value >= maxExclusive) return -1;
    const rounded = Math.round(value);
    return rounded === value ? rounded : -1;
  };

  interface RankedNode {
    readonly node: number;
    readonly score: number;
    readonly completion: number;
    readonly perfection: number;
    readonly stability: number;
  }
  const ranked: RankedNode[] = [];
  for (let n = 0; n < width; n++) {
    const base = n * NS;
    if (base + NODE_ALIVE >= nodes.length) break;
    if (nodes[base + NODE_ALIVE] < 0.5) continue;
    if (Math.round(nodes[base + NODE_STEP]) !== levels) continue;
    const completion = nodes[base + 13];
    const perfection = nodes[base + 14];
    const stability = nodes[base + 1];
    ranked.push({
      node: n,
      score: completion * 3 + perfection + stability * 0.05,
      completion,
      perfection,
      stability,
    });
  }
  ranked.sort((a, b) => (a.score !== b.score ? b.score - a.score : a.node - b.node));

  const paths: GpuPath[] = [];
  const seen = new Set<string>();
  for (const candidate of ranked) {
    if (paths.length >= maxPaths) break;
    const seq = new Array<number>(levels);
    let node = candidate.node;
    let valid = true;
    // Walk from the final level back to level 0; seq is filled in reverse so
    // the returned order is the replay order.
    for (let level = levels - 1; level >= 0; level--) {
      const entry = (level * width + node) * 2;
      if (entry + 1 >= trace.length) {
        valid = false;
        break;
      }
      const parent = toIndex(trace[entry], width);
      const skill = toIndex(trace[entry + 1], skills.length);
      if (parent < 0 || skill < 0) {
        valid = false;
        break;
      }
      seq[level] = skill;
      node = parent;
    }
    if (!valid) continue;
    const key = seq.join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    paths.push({
      skills: seq.map((index) => skills[index].name),
      score: candidate.score,
      completion: candidate.completion,
      perfection: candidate.perfection,
      stability: candidate.stability,
    });
  }
  return paths;
}

async function run(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  options: GpuSearchOptions,
  profiler: GpuProfiler,
): Promise<GpuSearchResult> {
  const depth = Math.max(0, Math.floor(options.depth ?? 0));
  const beamWidth = Math.max(1, Math.floor(options.beamWidth ?? 1));
  const topK = Math.max(1, Math.min(8, Math.floor(options.topK ?? 1)));
  const records: GpuRecordSet = buildGpuRecords(state, config, skills, {
    currentCondition: options.currentCondition,
    nodeCount: beamWidth,
    beamWidth,
    topK,
  });
  profiler.mark('records');
  profiler.setShape({
    levels: depth,
    skillCount: records.skillCount,
    nodeCount: records.nodeCount,
  });
  // Two f32 per (level, node): parent node index and applied skill index.
  const traceBytes = depth > 0 ? depth * records.nodeCount * 2 * 4 : 0;
  const sizes = { ...records.sizes, traceBytes };
  gpuDebugLog(
    `records skillCount=${records.skillCount} nodeCount=${records.nodeCount} ` +
      `skillsBytes=${sizes.skillsBytes} nodesBytes=${sizes.nodesBytes} ` +
      `deltasBytes=${sizes.deltasBytes} candidatesBytes=${sizes.candidatesBytes} ` +
      `topBytes=${sizes.topBytes} traceBytes=${sizes.traceBytes}`,
  );
  const ctx = await getContext();
  const device = ctx.device;
  const key = `${records.nodeCount}/${records.skillCount}/${topK}`;
  const pipelines = ctx.pipelines[key] ?? (await createPipelines(device, {
    beamWidth: records.nodeCount,
    skillCount: records.skillCount,
    topK,
  }));
  // Cache only after compilation succeeds; a failed shader must not poison reuse.
  ctx.pipelines[key] = pipelines;
  profiler.mark('device');

  const skillBuf = makeBuffer(device, records.skills, GPUBufferUsage.STORAGE);
  const initBuf = makeBuffer(device, records.nodes, GPUBufferUsage.STORAGE);
  const deltaBytes = records.sizes.deltasBytes;
  const deltaBuf = device.createBuffer({
    size: deltaBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  // Readback: deltas, final nodes, then the whole replay trace.
  const nodesOffset = deltaBytes;
  const traceOffset = nodesOffset + (depth > 0 ? records.sizes.nodesBytes : 0);
  const stagingBytes = traceOffset + traceBytes;
  const staging = device.createBuffer({
    size: stagingBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const beamBytes =
    depth > 0
      ? 2 * records.sizes.nodesBytes +
        records.sizes.candidatesBytes +
        records.sizes.topBytes +
        traceBytes
      : 0;
  profiler.addBufferBytes(
    sizes.skillsBytes +
      sizes.nodesBytes +
      deltaBytes +
      stagingBytes +
      beamBytes,
  );
  const beamBuffers: GpuBuffer[] = [];
  try {
    let bufA: GpuBuffer | null = null;
    let bufB: GpuBuffer | null = null;
    let candBuf: GpuBuffer | null = null;
    let topBuf: GpuBuffer | null = null;
    let bgResetA: GpuBindGroup | null = null;
    let bgSelectA: GpuBindGroup | null = null;
    let bgSelectB: GpuBindGroup | null = null;
    let bgTop: GpuBindGroup | null = null;
    let traceBuf: GpuBuffer | null = null;
    const levelBuffers: GpuBuffer[] = [];
    const bgApply: GpuBindGroup[] = [];
    if (depth > 0) {
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
      bufA = device.createBuffer({ size: records.sizes.nodesBytes, usage });
      bufB = device.createBuffer({ size: records.sizes.nodesBytes, usage });
      candBuf = device.createBuffer({
        size: records.sizes.candidatesBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      topBuf = device.createBuffer({
        size: records.sizes.topBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      traceBuf = device.createBuffer({
        size: traceBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      beamBuffers.push(bufA, bufB, candBuf, topBuf, traceBuf);
      bgResetA = bind(device, pipelines.reset, [
        { binding: 1, buffer: initBuf },
        { binding: 2, buffer: bufA },
      ]);
      const mkSelect = (buffer: GpuBuffer): GpuBindGroup =>
        bind(device, pipelines.select, [
          { binding: 0, buffer: skillBuf },
          { binding: 2, buffer },
          { binding: 4, buffer: candBuf as GpuBuffer },
        ]);
      bgSelectA = mkSelect(bufA);
      bgSelectB = mkSelect(bufB);
      bgTop = bind(device, pipelines.top, [
        { binding: 4, buffer: candBuf },
        { binding: 5, buffer: topBuf },
      ]);
      // One 4-byte level buffer + one apply bind group per level. A uniform
      // write would collapse: all writes happen before the single submit, so
      // every dispatch would see the last level. Buffers alternate A->B->A...
      for (let level = 0; level < depth; level++) {
        const levelBuf = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE,
          mappedAtCreation: true,
        });
        new Float32Array(levelBuf.getMappedRange())[0] = level;
        levelBuf.unmap();
        levelBuffers.push(levelBuf);
        const even = level % 2 === 0;
        bgApply.push(bind(device, pipelines.apply, [
          { binding: 2, buffer: (even ? bufA : bufB) as GpuBuffer },
          { binding: 3, buffer: (even ? bufB : bufA) as GpuBuffer },
          { binding: 5, buffer: topBuf as GpuBuffer },
          { binding: 7, buffer: levelBuf },
          { binding: 8, buffer: traceBuf as GpuBuffer },
        ]));
      }
      beamBuffers.push(...levelBuffers);
    }

    profiler.mark('upload');
    const encodeStart = now();
    const encoder = device.createCommandEncoder();
    const bgScore = bind(device, pipelines.score, [
      { binding: 0, buffer: skillBuf },
      { binding: 2, buffer: initBuf },
      { binding: 6, buffer: deltaBuf },
    ]);
    dispatch(
      encoder,
      pipelines.score,
      bgScore,
      workgroups(records.nodeCount * records.skillCount),
    );
    profiler.addDispatches(1);
    gpuDebugLog(
      `dispatch level=score dispatches=1 groups=${workgroups(records.nodeCount * records.skillCount)}`,
    );

    let finalBuf: GpuBuffer = initBuf;
    if (
      depth > 0 &&
      bufA && bufB && bgResetA && bgSelectA && bgSelectB && bgTop &&
      traceBuf && bgApply.length === depth
    ) {
      dispatch(encoder, pipelines.reset, bgResetA, workgroups(records.nodeCount));
      let inputIsA = true;
      for (let level = 0; level < depth; level++) {
        dispatch(
          encoder,
          pipelines.select,
          inputIsA ? bgSelectA : bgSelectB,
          records.nodeCount,
        );
        dispatch(
          encoder,
          pipelines.top,
          bgTop,
          workgroups(records.nodeCount * topK),
        );
        dispatch(
          encoder,
          pipelines.apply,
          bgApply[level],
          workgroups(records.nodeCount),
        );
        profiler.addDispatches(3);
        gpuDebugLog(`dispatch level=${level + 1}/${depth} dispatches=3`);
        inputIsA = !inputIsA;
      }
      finalBuf = inputIsA ? bufA : bufB;
    }
    encoder.copyBufferToBuffer(deltaBuf, 0, staging, 0, deltaBytes);
    if (depth > 0) {
      encoder.copyBufferToBuffer(finalBuf, 0, staging, nodesOffset, records.sizes.nodesBytes);
      if (traceBuf) {
        encoder.copyBufferToBuffer(traceBuf, 0, staging, traceOffset, traceBytes);
      }
    }
    device.queue.submit([encoder.finish()]);
    const encodeMs = now() - encodeStart;
    const submitAt = now();
    profiler.mark('dispatch');

    await staging.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(staging.getMappedRange());
    const deltaValues = decodeFloats(bytes, 0, deltaBytes);
    const beamValues =
      depth > 0 ? decodeFloats(bytes, nodesOffset, records.sizes.nodesBytes) : null;
    const traceValues =
      depth > 0 ? decodeFloats(bytes, traceOffset, traceBytes) : null;
    staging.unmap();
    const readbackMs = now() - submitAt;
    profiler.mark('readback');
    const profile = profiler.finish();
    gpuDebugLog(`profile ${describeProfile(profile)}`);

    return {
      deltas: mapDeltas(deltaValues, records.skillCount, records.skillNames),
      timing: {
        encodeMs,
        gpuMs: readbackMs,
        readbackMs,
        totalMs: profile.totalMs,
      },
      profile,
      notes: records.notes,
      paths:
        traceValues && beamValues
          ? reconstructPaths(
              traceValues,
              beamValues,
              depth,
              beamWidth,
              skills,
              DEFAULT_PATH_LIMIT,
            )
          : [],
      ...(beamValues
        ? {
            beam: {
              ...summarizeBeam(beamValues, records.nodeCount),
              roots: summarizeRoots(beamValues, records.nodeCount, skills),
            },
          }
        : {}),
    };
  } catch (error) {
    gpuDebugLog(
      `error phase=search elapsedMs=${profiler.elapsedSince('start').toFixed(1)} message=${objEventMessage(error)}`,
    );
    disableGpuBackend(`WebGPU-поиск прерван: ${String(error)}`);
    throw error;
  } finally {
    skillBuf.destroy();
    initBuf.destroy();
    deltaBuf.destroy();
    staging.destroy();
    for (const buffer of beamBuffers) buffer.destroy();
  }
}

/**
 * Run one search. A hard timeout bounds the caller's wait; on expiry the
 * backend disables itself for the rest of the session (the in-flight GPU
 * command buffer is not cancelled by WebGPU).
 */
export async function search(
  state: CraftingState,
  config: OptimizerConfig,
  skills: SkillDefinition[],
  options: GpuSearchOptions = {},
): Promise<GpuSearchResult> {
  const profiler = new GpuProfiler(now);
  const timeoutMs = Math.max(1, Math.min(60_000, Math.floor(options.timeoutMs ?? 2000)));
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const elapsedMs = profiler.elapsedSince('start');
      const message = `WebGPU-таймаут ${timeoutMs} мс (прошло ${elapsedMs.toFixed(0)} мс)`;
      gpuDebugLog(`timeout limit=${timeoutMs} elapsedMs=${elapsedMs.toFixed(1)}`);
      if (options.disableOnTimeout !== false) disableGpuBackend(message);
      reject(new Error(message));
    }, timeoutMs);
  });
  try {
    return await Promise.race([
      run(state, config, skills, options, profiler),
      timeout,
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
