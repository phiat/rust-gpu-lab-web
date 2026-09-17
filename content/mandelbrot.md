---
title: "Project: mandelbrot"
summary: The first tileworld crate. Two cuTile kernels against a rayon baseline, and what they taught.
order: 10
section: Projects
---

A Mandelbrot renderer with a CPU reference (rayon) and two GPU tile kernels,
plus a benchmark that checks they agree. The [Project page](/project) shows the
current source and any renders in the workspace. This note covers the design.

## Running it

```bash
# GPU render (early-exit kernel, 1920×1080, 1000 iters, 64×64 tiles)
cargo run --release -p mandelbrot -- render --out mandelbrot.png

# same view on the CPU
cargo run --release -p mandelbrot -- render --cpu

# pick the view and kernel
cargo run --release -p mandelbrot -- render \
  --center -0.75,0.1 --span 0.05 --iters 4000 --tile 32 --kernel fixed

# time serial CPU, rayon, and both kernels, and compare pixels
cargo run --release -p mandelbrot -- bench --runs 10 --skip-serial
```

`--center` accepts a leading minus. `--check-every` sets how often the
early-exit kernel checks whether its tile is finished.

## Layout

| File         | Role                                                          |
| ------------ | ------------------------------------------------------------- |
| `main.rs`    | clap CLI (`render`, `bench`), `View`, timing, CPU/GPU compare |
| `cpu.rs`     | `escape()` per pixel; `render_serial` and `render_rayon`      |
| `gpu.rs`     | `#[cutile::module] kernels` plus the host-side `Gpu` wrapper  |
| `palette.rs` | smooth count → RGB (cosine gradient), done on the CPU         |

## Shared math

Both sides run the same `f32` recurrence so their results can be compared:

- **Pixel mapping**: square pixels, sampled at pixel centers.
  `c = (x0 + col·step, y0 − row·step)`, and row 0 is the top.
- **Bailout** at `|z|² > 256`, larger than the minimal 4 so the smooth count is
  stable.
- **Output** is `f32` per pixel: `n + 1 − log2(ln|z|)` if it escaped, `−1` if it
  didn't. Color is applied afterwards on the host.

## Kernel 1: `mandelbrot_fixed`

A tile program has no per-pixel thread index, but it knows its block id. It
builds the coordinates for its whole `BH × BW` block from two 1-D vectors:

```rust
let pid: (i32, i32, i32) = get_tile_block_id();
let row0: i32 = pid.0 * BH;

let rows: Tile<i32, { [BH] }> = iota(shape![BH]);
let rows: Tile<i32, { [BH] }> = rows + row0.broadcast(shape![BH]);
let rows: Tile<f32, { [BH] }> = convert_tile(rows);
// ... ci from rows, cr from columns the same way ...

let cr: Tile<f32, { [BH, BW] }> = cr.reshape(shape![1, BW]).broadcast(s);
let ci: Tile<f32, { [BH, BW] }> = ci.reshape(shape![BH, 1]).broadcast(s);
```

Then it runs **every** step on the whole tile. Escaped pixels are frozen with
`select`, not branched away:

```rust
for _step in 0i32..max_iter {
    let zr2: Tile<f32, { [BH, BW] }> = zr * zr;
    let zi2: Tile<f32, { [BH, BW] }> = zi * zi;
    let alive: Tile<bool, { [BH, BW] }> = le_tile(zr2 + zi2, limit);
    let zi_next: Tile<f32, { [BH, BW] }> = (zr + zr) * zi + ci;
    let zr_next: Tile<f32, { [BH, BW] }> = zr2 - zi2 + cr;
    zr = select(alive, zr_next, zr);
    zi = select(alive, zi_next, zi);
    n = select(alive, n + one, n);
}
out.store(smooth_count(s, zr, zi, n));
```

`max_iter` is an ordinary `i32` scalar, so a runtime loop bound works.

## Kernel 2: `mandelbrot_early_exit`

The math is the same, but a **tile** stops once every one of its pixels has
escaped. Tiles entirely outside the set finish in a few steps. Tiles that touch
the set still run to `max_iter`.

```rust
while done < max_iter {
    let chunk: i32 = min(check_every, max_iter - done);
    for _step in 0i32..chunk {
        // same masked step as the fixed kernel
    }
    done = done + chunk;

    // Reduce the 2-D mask to one scalar: is anything still alive?
    let mag2: Tile<f32, { [BH, BW] }> = zr * zr + zi * zi;
    let alive: Tile<bool, { [BH, BW] }> = le_tile(mag2, limit);
    let alive_f: Tile<f32, { [BH, BW] }> = select(alive, one, zero);
    let row_any: Tile<f32, { [BH] }> = reduce_max(alive_f, 1i32);
    let any: Tile<f32, { [] }> = reduce_max(row_any, 0i32);
    let any: f32 = tile_to_scalar(any);
    if any < 0.5f32 {
        break;
    }
}
```

Loop shape matters. According to the comment in `gpu.rs`, one `while` with the
check inside it ran about 2.7× slower than the fixed kernel, even when the check
never fired. So the hot loop stays a counted `for`, and only the outer `while`
can `break`, which `for` doesn't allow anyway.

> The [Playground](/playground) can overlay how many steps each tile would run
> under this kernel.

## Host side

The whole pipeline is built lazily and synced once:

```rust
let out = api::zeros::<f32>(&[view.height, view.width]).partition([tile, tile]);
let pixels = kernels::mandelbrot_fixed(out, x0, y0, step, iters)
    .first()          // the launch returns (out, x0, y0, step, iters)
    .unpartition()
    .to_host_vec()
    .sync_on(&self.stream)?;
```

`bench` reuses one device buffer (`render_on_device`) so it can time GPU work
without the download. It times the **first launch** separately because that
launch includes JIT compilation.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, 1920×1080, 64×64 tiles. GPU
times are warm medians. Each new process first spends about 350–550 ms compiling
kernels.

Default view, 1000 iterations:

| Renderer       | Time    |
| -------------- | ------- |
| CPU, 1 thread  | 980 ms  |
| CPU, rayon     | 48.7 ms |
| GPU fixed      | 1.06 ms |
| GPU early-exit | 0.69 ms |

10,000 iterations, by view:

| View                            | Rayon   | GPU fixed | GPU early-exit |
| ------------------------------- | ------- | --------- | -------------- |
| default (mixed)                 | 435 ms  | 10.1 ms   | 6.1 ms         |
| entirely outside (`2.5,2.5`)    | 4.7 ms  | 10.2 ms   | 0.10 ms        |
| entirely inside (`-0.1,0`, 0.4) | 1650 ms | 10.6 ms   | 12.1 ms        |

Early exit wins by 100× where every tile escapes quickly, and costs a little
where no tile can stop early, because the checks still run.

Tile size (fixed kernel, 10k iterations, default view): 16 px 9.7 ms, 32 px 10.7
ms, 64 px 10.4 ms, **128 px 85 ms, 256 px 1577 ms**. The first launch grows too:
1 s to compile at 128 px, 12 s at 256 px.

## Lessons recorded in the code

- **Launchers return every argument as a tuple, and `.first()` only exists up to
  6 elements.** Keep entry-point scalar lists short. That's why there is a
  single `step` and why pixels are square.
- **Tile size:** 16–64 perform about the same, while 128 and up is far slower
  both to run and to compile (see the results above).
- **Image sizes don't have to divide evenly.** 1080 ÷ 64 isn't whole. The
  partition rounds the grid up and masks stores in the edge tiles, and `bench`
  has agreed with the CPU on every pixel so far.
- **A device function broke a comparison.** A one-line
  `fn bailout<const BH, const BW>(s: Shape<..>) -> Tile<..>` failed with
  `` binary `Le` requires operands of the same type ... `{[BH, BW]}` and
  `{[32, 32]}` ``.
  Inlining the constant fixed it. `filters` later found the cause:
  device-function results get concrete tile types, while constants and nested
  arithmetic keep generic ones (see its
  [gotchas](./filters.md#gotchas-at-d92c160)). `raymarch` avoids it by writing
  every tile shape as a literal.
- **Nested calls need types.** `gt_tile(x, constant(256.0f32, s))` fails with
  "Return type required". Bind the constant to an annotated `let` first.
- **GPU and CPU can differ slightly** on chaotic boundary pixels: the compiler
  may fuse or reorder float ops. `compare()` reports inside/outside
  disagreements separately from escape counts that differ by more than 0.01.

## Still open

- **Recompiles from `--iters`.** Float scalars don't create specializations, but
  the book says scalar hints are bucketed by power-of-two divisibility. Check
  whether 1000 → 1024 iterations recompiles, using `CUTILE_JIT_TIMING=1`.
  `raymarch` found its integer step counts aren't part of the cache key, but its
  presets (64, 128, 256) are all divisible by 16, so this case is still
  untested.
- **Precision.** `f32` limits zoom depth. What does an `f64` path cost?
- **Disk cache.** Not turned on here yet. In `raymarch` it cut a 30 s first
  start to about 1.5 s on later runs (see [Compilation](./compilation.md)).
- **Palette on the GPU.** Output colors directly as `[B, B, 4]` RGBA. RGB won't
  work, because tile dimensions must be powers of two.
- **A zoom animation** rendered into one preallocated buffer, as a warm-up for
  [CUDA graphs](./cuda-graphs.md).
