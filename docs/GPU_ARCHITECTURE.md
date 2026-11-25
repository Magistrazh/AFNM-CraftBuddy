# GPU search architecture

## Entry point

`tryGpuRecommendation` in `src/modContent/statefulGpuRecommendation.ts` selects
the search backend by `lookaheadDepth`:

- `lookaheadDepth <= 2` goes to `tryStatefulRecommendation`, a host-orchestrated
  depth-2 pipeline.
- `lookaheadDepth > 2` goes to `tryResidentRecommendation`, which calls
  `runResidentSearch` in `src/optimizer/gpu/residentSearch.ts`.

Both facades are non-throwing. Any failure returns an outcome with `ok: false`
and a refusal reason; the caller then falls back to the normal search ladder.

## Resident search

`runResidentSearch` performs four steps:

1. Build constant tables from the current state and config: skill records,
   catalog, condition effects, gains, availability parameters, score parameters,
   style. Every search setting is read from the function input; there are no
   parallel constants.
2. Run `runResidentDepth` from `src/optimizer/gpu/residentDepthLoop.ts`.
3. Rank the roots with the aggregation callback supplied by `residentSearch.ts`.
4. Map the winning root to a skill key and return it with the node counts and
   timing.

## Depth loop

`runResidentDepth` iterates `level = 0 .. depth-1`, clamped to `DEPTH_MAX = 96`.

- Before each level after the first, `recompute` callbacks rebuild gains,
  availability and style tables from the previous level's device readback. Root
  tables are never reused for deeper levels.
- `maxNodes` is checked between submits. There is no wall-clock check in the
  loop.
- Each level is one `runOneResidentLevel` call from
  `src/optimizer/gpu/residentBeam.ts`. Node and metadata buffers alternate
  between two device buffers (ping-pong), so states never leave the device
  between levels.
- `observe` is called after each level's readback; `rankRoots` produces the
  winning move from the collected levels.

`runResidentDepth` rejects any `beamWidth` other than `BEAM = 20`, because the
WGSL constants are compiled with that value. The UI pins the setting to 20 while
a GPU engine is selected (`src/ui/gpuBeamLock.ts`).

## Kernels

`src/optimizer/gpu/residentBeam.ts` compiles the WGSL kernels for state
transition, condition distribution, availability, verdict, score input, score,
survival blend and beam selection. Per level, one command encoder submits all
dispatches; the host reads back only node counts, metadata, values and a bounded
trace.

The beam keeps a fixed number of slots. `beamApply` ranks parents by their best
available continuation and keeps one child per parent, so a single level-1 node
cannot occupy all slots.

## Root aggregation

The aggregation in `residentSearch.ts` uses the full continuation table from the
device, not just the retained beam slots:

- for each level-1 slot, take the best available skill from the full `values`
  table using host availability;
- fold deeper levels into their parents through recorded lineage and branch
  probability;
- re-seed level-1 slots that were dropped by the beam, rebuilding their tables
  from the slot's own post-action state;
- apply the auto-finish short-circuit and the survival correction to the root
  expectations;
- pick the root with the highest corrected expectation.

This aggregation is the part that determines the chosen move. The depth loop
only produces the continuation data it consumes.

## Refusal paths

`tryResidentRecommendation` refuses with an explicit reason when WebGPU is
absent, the adapter is software, a timeout fires, the device is lost, or the
result records are incomplete. The refusal is reported through a diagnostic
event; the normal search ladder remains unchanged.
