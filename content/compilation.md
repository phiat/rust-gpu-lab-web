---
title: Compilation
summary: Launch-time JIT, what creates a new specialization, caches, and debugging knobs.
order: 5
section: Concepts
---

`cargo build` doesn't compile your kernels for the GPU. `#[cutile::module]`
captures each kernel's **Rust AST** into the host binary. The GPU code is
produced **at first launch**:

```text
Rust AST  →  Tile IR bytecode  →  cubin (GPU binary)
```

```rust
let op = add((&mut z).partition([16, 16]), &x, &y); // builds a DeviceOp
let _ = op.sync_on(&stream)?;                       // may compile, then launches
```

The cubin is cached **in process memory**, so later launches of the same variant
skip compilation. Restarting the process clears that cache.

## What makes a new specialization

It works a lot like Rust monomorphization. A **specialization** is one compiled
variant for one entry function, one target GPU, and one set of compile-time
inputs.

| Causes a recompile                                                         | Doesn't                               |
| -------------------------------------------------------------------------- | ------------------------------------- |
| element types (`f32` vs `f16`)                                             | tensor **contents**                   |
| const generic values, e.g. the tile shape from `.partition([BM, BN])`      | **dynamic** (`-1`) dimensions         |
| static dims bound from a tensor's shape                                    | floating-point **scalar** values      |
| compile options (`max_divisibility`, …)                                    | runtime grid set with `.grid(...)`    |
| target GPU architecture (`sm_89`, `sm_90`, …)                              |                                       |
| power-of-two divisibility of integer scalars and dynamic dims (100 vs 128) | values in the same bucket (64 vs 128) |

So in a Mandelbrot kernel, changing **tile size** recompiles, but panning or
zooming by passing new `f32` scalars shouldn't.

One subtlety: the cache key records the largest power-of-two divisor, capped at
16, of every integer scalar argument and of every tensor's shape, strides and
base pointer. Values such as 16, 32, 64 and 1024 share a bucket, while 100
(divisible by 4) and 1000 (divisible by 8) land in others. `light2d`'s
jump-flooding kernel compiles into 10 variants this way, because its passes
differ in block offsets and view shifts (16, 8, 4, 2, 1).

The same rule caught `sand`: passing the Margolus pass index (0, 1, 2, 3) as an
`i32` argument compiled its 2.4 s `step` kernel three times, for pass 0, the odd
passes, and 2. The fix is to keep per-launch integers in a small tensor and
slice a view per launch, which the JIT doesn't inspect, and which is also what a
CUDA graph needs. See
[sand](./sand.md#passing-per-launch-integers-without-jit-variants).

Shapes count too. `cloth`'s constraint kernel takes two shifted views whose
offsets differ per batch (−1 row, +2 columns, a diagonal), and views of
different shapes would have meant several variants. It slices every view to one
shape and loads them all at the same block, so one kernel serves every
direction. See [cloth](./cloth.md#one-kernel-whatever-the-direction).

## Disk cache (opt-in)

Off by default, and no environment variable turns it on:

```rust
cutile::jit_cache::enable_default()?; // ~/.cache/cutile/kernels, 2 GiB LRU
```

See `cutile-examples/examples/jit_disk_cache.rs`. Run it twice and the second
run loads from disk. Every tileworld demo turns it on through one helper in
[tilekit](./tilekit.md), which passes the store explicitly:

```rust
pub fn enable_jit_cache() -> Result<(), Box<dyn std::error::Error>> {
    cutile::jit_cache::enable(Arc::new(
        cutile::jit_cache::FileSystemJitStore::default_location()?,
    ));
    Ok(())
}
```

| Demo                            | First start        | Later runs   |
| ------------------------------- | ------------------ | ------------ |
| `mandelbrot`, `life`, `filters` | about 0.75–0.9 s   | about 0.3 s  |
| `raymarch`                      | about 30 s         | about 1.5 s  |
| `light2d`, 15 kernels           | 15 compiles        | about 4 s    |
| `sand`, 4 kernels               | `step` alone 1.7 s | about 2.2 s  |
| `cloth`, 4 kernels              |                    | about 0.75 s |

A hit skips `tileiras` but not the IR build, which costs 40–300 ms per kernel
variant on every start. That's why `light2d` still takes seconds, and why `sand`
barely gains: its one big `step` kernel spends 1.7 s in the stage before
`tileiras`.

`raymarch`'s kernel takes about 30 s to compile the first time and about 1.5 s
to start after that. Any edit to the kernel changes the cache key. Its step
counts are integer scalars, so only their divisibility matters: the presets (64,
128 and 256 steps) are all multiples of 16 and share one cached kernel, but a
100-step preset would compile another.

## Compile time grows with inlining

Device functions are inlined at every call site. In `raymarch`, computing a
surface normal with 4 separate calls to the scene SDF took 48 s in `tileiras`.
The same 4 samples inside a `for` loop took 26 s, because the loop body inlines
the scene once. Building the IR also costs about 1.4 s on every start, even on a
cache hit.

That IR cost is per variant. `light2d` compiles 15 kernels, 10 of them
divisibility variants of one jump flooding kernel, and each costs about 270–320
ms of IR building at startup. With every compiled kernel on disk, it still takes
about 4 s to reach the first frame.

Tile size matters too. `mandelbrot`'s first launch took about 1 s to compile at
128 px tiles and 12 s at 256 px.

Op count matters most, and it's the same count that decides the frame time.
`sand`'s `step` kernel went from 2.4 s and 0.5 ms per frame to 1.7 s and 0.21 ms
by writing its rules smaller: bit-mask tables instead of
`kind == X || kind
== Y` chains, the density table packed into one constant, and
`kind()` hoisted out of the rules. See
[sand](./sand.md#compile-time-is-run-time).

## Seeing what the compiler did

```rust
#[cutile::entry(print_ir = true)]
#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
```

| Env var               | Effect                                              |
| --------------------- | --------------------------------------------------- |
| `CUTILE_DUMP=ir`      | print Tile IR per compiled module (`bc`, `all` too) |
| `CUTILE_DUMP_FILTER`  | limit dumps to `module::function` names             |
| `CUTILE_JIT_TIMING=1` | per-kernel timing, incl. whether it hit disk        |

## When JIT fails

| Symptom                         | Likely cause                    | Check                             |
| ------------------------------- | ------------------------------- | --------------------------------- |
| `invalid GPU architecture`      | toolkit doesn't support your SM | CUDA Toolkit version              |
| failed to load generated kernel | driver/toolkit mismatch         | `nvidia-smi`, `nvcc --version`    |
| segfault in CUDA / Tile IR libs | broken toolkit path             | `CUDA_TOOLKIT_PATH`, lib path     |
| OOM on first launch             | host RAM exhausted compiling    | memory, number of specializations |

Most other errors are ordinary Rust type or shape errors caught before anything
runs. These JIT-time errors came up while building tileworld's crates:

| Error text                                            | Cause                                                        | Fix                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| "tile shape dimensions must have power of two length" | a tile such as `[B, B, 3]`                                   | pad to `[B, B, 4]`                                                     |
| "input does not correspond to Tile IR bytecode"       | usually a `tileiras` error printed earlier                   | read all of stderr, set `CUTILE_DUMP=ir,bytecode`                      |
| ``binary `Add` requires operands of the same type``   | a generic `{[B, B]}` tile mixed with a concrete `{[32, 32]}` | write shapes as literals, or keep a kernel all generic or all concrete |
| "Return type required"                                | an untyped nested call, like `gt_tile(x, constant(…))`       | bind the inner value to an annotated `let`                             |
| "missing attribute 'overflow'"                        | `trunci(x, overflow::None)`                                  | use `overflow::NoWrap` (or another real mode)                          |
| "unrecognized macro `unreachable`"                    | scalar `.broadcast(shape)` on a literal or local `let`       | broadcast an entry parameter, or use `constant(v, shape)`              |
| "duplicate functions are not supported"               | a device function named like a DSL op, such as `unpack`      | rename the helper                                                      |
| "Failed to get type parameters for convert_tile"      | `convert_tile(…)` as a function's tail expression            | bind the result to an annotated `let`, then return it                  |

The [filters gotchas](./filters.md#gotchas-at-d92c160) explain the tile-type
mismatch in more detail.

Next: [Stencils](./stencils.md).
