# AFNM CraftBuddy (GPU)

A crafting assistant for **Ascend From Nine Mountains** with an accelerated
search backend that runs on the graphics processor. It shows the next action,
its expected gains and costs, a multi-step plan, and alternative actions.

## Features

- **WebGPU search backend.** Candidate construction and state evaluation are
  moved to the GPU. A device-resident beam keeps states in device buffers
  between levels and supports depth up to 96.
- **The same settings as the normal search.** Depth, time budget, node limit and
  beam width are read from the panel unchanged.
- **Reliable fallback.** Any failure in the GPU path returns a refusal and falls
  back to the normal search without throwing into the UI.
- **Diagnostics.** Engine selection and plan recalculation are reported through
  a local debug sink.
- **Russian client.** Localized crafting labels are recognized.

The search architecture is documented in
[docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md).

## Installation

1. Build the archive:

   ```bash
   bun install
   bun run build
   ```

2. Copy `builds/afnm-craftbuddy.zip` into the `mods` folder next to the game
   executable, **without unzipping**.
3. Launch the game.

## Usage

Open the panel with `Ctrl+Shift+C`. In **Advanced search**, select the `GPU`
engine, then press **Update plan** if the current plan is marked out of date.

Selecting a GPU engine pins **Branches to keep** at 20 and greys the control out,
because the compiled kernel requires that width. The field becomes editable
again for the normal search.

## Measured result

On the hardest scenario from the game files (`hardest-54`, 54 techniques):

| Source | Score | Time |
|---|---:|---:|
| Normal search, 60 s | −133 976 983.31 | 60 010 ms |
| GPU mode, depth 96 | −21 072 985.93 | 550 ms |

A higher score, meaning closer to zero, is better.

## Development

```bash
bun run typecheck
bun run test
bun run build
```

## License

MIT. Portions of the project are based on work by **lemon07r**; the original
copyright notice is retained as required by the MIT license.
