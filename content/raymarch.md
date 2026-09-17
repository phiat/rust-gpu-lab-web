---
title: "Project: raymarch"
summary: A real-time SDF ray marcher in one 32×32 tile kernel, with its inputs in a device buffer so a CUDA graph can replay every frame.
order: 13
section: Projects
---

A signed distance field (SDF) ray marcher written as a single cuTile kernel.
Each 32×32 tile program sphere-traces its pixels, shades them with soft shadows,
ambient occlusion, specular highlights and fog, and packs the result into
`0x00RRGGBB` for a `minifb` window. A rayon version of the same math checks the
output, and the largest per-channel difference is 1 level out of 255.

It reuses ideas from earlier crates: per-tile early exit from
[mandelbrot](./mandelbrot.md), graph replay from [life](./life.md) and
[filters](./filters.md), and the tile-type gotcha from `filters`, which it
avoids entirely.

## Running it

```bash
cargo run --release -p raymarch -- run                  # window, 1280×720
cargo run --release -p raymarch -- run --width 1920 --height 1080 --quality 3
cargo run --release -p raymarch -- render --yaw 200 --time 2 --out shot.png
cargo run --release -p raymarch -- render --view tiles  # per-tile work heatmap
cargo run --release -p raymarch -- bench --width 1920 --height 1080 --quality 3
```

In the window, drag or use the arrow keys to orbit, and scroll or press `+`/`-`
to zoom. Space pauses the animation, `O` toggles auto-orbit, and `V` cycles the
views: shaded, march steps per pixel, and steps each tile ran. `1`, `2` and `3`
set quality, `P` saves a screenshot, and Esc quits.

**The first run takes about 30 s** while `tileiras` compiles the kernel. `main`
turns on cuTile's disk cache (`tilekit::enable_jit_cache()`), so later runs
start in about 1.5 s. Any change to the kernel changes its cache key and pays
the full compile again.

## Layout

| File       | Role                                                  |
| ---------- | ----------------------------------------------------- |
| `gpu.rs`   | the kernel (SDFs, march, shading) and `Renderer`      |
| `cpu.rs`   | the same scene and shading, one pixel at a time       |
| `scene.rs` | orbit camera, animation, the 32-slot parameter block  |
| `main.rs`  | `run` (window), `render` (PNG plus CPU diff), `bench` |

## One tile shape, spelled as a literal

The kernel has no `const B` generic. Every shape is the literal `32`, behind
type aliases:

```rust
type F = Tile<f32, { [32, 32] }>;
type Mask = Tile<bool, { [32, 32] }>;
type Params = Tile<f32, { [32] }>;

fn fill(v: f32) -> F {
    broadcast_scalar(v, const_shape![32, 32])
}
```

With a generic entry point, device-function results come back typed `{[32, 32]}`
while locals stay `{[B, B]}`, and `+`, `-` and `*` reject the mix (the
[filters gotcha](./filters.md#gotchas-at-d92c160)). With only one spelling,
helper functions compose like ordinary Rust. The cost: the host must partition
with exactly 32×32 tiles, and a 16×16 version would need a second copy of the
kernel.

## Inputs in a device buffer, not scalars

Scalar kernel arguments are baked into a captured CUDA graph. So the camera,
animation and view mode go in a 32-slot `f32` tensor that the kernel loads as a
`[32]` tile and reads with `extract`:

```rust
/// `params[k]`, broadcast over the tile.
fn param(p: Params, k: i32) -> F {
    let idx: Tile<i32, { [] }> = scalar_to_tile(k);
    let v: Tile<f32, { [1] }> = extract(p, [idx]);
    let v: Tile<f32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
    v.broadcast(const_shape![32, 32])
}
```

Each frame copies new values into the same buffer, then replays the graph. The
copy goes through a pinned host buffer ([tilekit](./tilekit.md#pinned-buffers)),
so it allocates nothing:

```rust
pub fn set_params(&mut self, params: &[f32; PARAMS]) -> Result<(), Error> {
    self.params_host.as_mut_slice().copy_from_slice(params);
    self.params_host.upload(&mut self.params, &self.stream)
}
```

The step budgets (`max_steps`, `check_every`, `shadow_steps`) are loop bounds,
so they stay scalars. Pressing `1`, `2` or `3` recaptures the graph. The slots
are listed in `scene::slot`, and trig for the animation runs once on the host
instead of per pixel per step.

## Sphere tracing with per-tile early exit

Each ray advances by the scene distance until it's within `eps · t` of a surface
or passes `tmax`. Finished rays are frozen with `select`, and every 16 steps the
tile checks whether any ray is still marching:

```rust
while done < max_steps {
    let chunk: i32 = min(check_every, max_steps - done);
    for _i in 0i32..chunk {
        let d: F = scene(ox + dx * t, oy + dy * t, oz + dz * t, rs, rc, bob);
        let not_hit: Mask = ge_tile(d, eps * t);
        let in_range: Mask = le_tile(t, tmax);
        active = select(not_hit, select(in_range, active, in_range), not_hit);
        t = select(active, t + d, t);
        steps = select(active, steps + one, steps);
        tile_steps = tile_steps + 1.0f32;
    }
    done = done + chunk;
    // reduce `active` to one scalar, and break if nothing is marching
}
```

This is Mandelbrot's loop shape again: a counted `for` inside, the `break` in
the outer `while`. `--view tiles` shows how it plays out. Tiles in front of the
camera stop after 16 steps and sky tiles after about 32, while tiles along the
horizon, where rays skim the ground, use the whole budget. The larger the
budget, the more early exit saves: 1.2× at 64 steps, 2.3× at 256.

## Skipping whole tiles with a scalar `if`

Within a tile, every pixel runs every op, so `select` computes both sides. The
only way to skip work is a scalar `if` over the whole tile. Tiles that are all
sky skip the normal, shadow and occlusion code, which is most of the per-pixel
cost:

```rust
let any_hit: f32 = tile_to_scalar(any_hit);
if any_hit > 0.5f32 {
    // normals, material, soft shadow, occlusion, lighting, fog
}
```

## SDF cost is per sample

On a CPU or in a shader you'd skip a costly shape when the sample is far away
from it. Here that would be a `select`, and both branches run anyway. The first
pillar rings used `atan2`, `cos` and `sin` for polar repetition, and the 720p
frame time went from about 1.6 ms to 5.8 ms. Rewriting the rings as mirror folds
brought it back to 1.6 ms:

```rust
let ax: F = absf(x);
let az: F = absf(z);
let (x45, z45) = (max_tile(ax, az), min_tile(ax, az));
// 22.5 degree wedge: 16 columns at radius 6.5, centered at 11.25 degrees.
let (x22, z22) = fold(x45, z45, 0.92387953f32, 0.38268343f32);
```

`abs` folds space into one quadrant, the min/max swap into 45°, and each fold
halves the wedge again, so one column stands in for all of them.

## Compile time

Device functions are inlined, so the scene SDF is copied once per call site.
With the normal written as 4 separate `scene` calls, `tileiras` took 48 s.
Folding the 4 samples into a `for` loop cut that to 26 s, and the finished
kernel takes about 31 s. Building the IR also costs about 1.4 s on every start,
even when the compiled kernel comes from the disk cache.

The disk cache is opt-in: `main` calls `tilekit::enable_jit_cache()`, which
wraps `cutile::jit_cache::enable(...)` (see
[Compilation](./compilation.md#disk-cache-opt-in)).

`generics=` is empty, but the cache key still records each integer argument's
largest power-of-two divisor, capped at 16 (and the same for tensor shapes,
strides and pointers). Every preset's step counts are multiples of 16, so
switching quality doesn't recompile, but a 100-step preset would. `light2d` got
10 variants of one kernel this way. `CUTILE_JIT_TIMING=1` shows each stage and
whether the kernel came from disk. See
[what makes a new specialization](./compilation.md#what-makes-a-new-specialization).

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, 1920×1080, default camera, per
frame:

| Quality (march / shadow steps) | CPU rayon | GPU eager | GPU graph | GPU, no early exit |
| ------------------------------ | --------- | --------- | --------- | ------------------ |
| low (64 / 16)                  | 129 ms    | 2.34 ms   | 2.25 ms   | 2.77 ms            |
| medium (128 / 32)              | 175 ms    | 3.19 ms   | 3.17 ms   | 5.07 ms            |
| high (256 / 64)                | 256 ms    | 4.14 ms   | 4.10 ms   | 9.58 ms            |

That's 55–62× faster than rayon. The 1280×720 window at medium quality runs at
about 335 fps: 1.65 ms rendering, 0.23 ms downloading and 0.05 ms uploading
parameters, through pinned host buffers. With pageable transfers it was 310 fps,
with a 0.4 ms download. The remaining millisecond per frame is the window:
presenting the image and polling input.

A frame is a single kernel launch, so the graph barely beats eager (3.17 vs 3.19
ms). `life`, which launches a kernel per generation, gained up to 5×.

## Matching the CPU

`cpu.rs` keeps the kernel's operation order and lands within 1/255 everywhere.
It isn't bit-exact, because fused multiply-add and `exp` can differ in the last
bits, but sphere tracing didn't amplify that into different hits.

## Gotchas at `d92c160`

- **An underestimating SDF made wavy shadows.** The first pillar layout cut a
  courtyard out of a repeated grid with `max(column, 5 - length(xz))`. That's a
  valid lower bound, but near the cut it badly underestimates, and the soft
  shadow term `k · h / t` turns that into false penumbras. The fold-based rings
  are exact.
- **`atan2(y, x)`** matches Rust's `y.atan2(x)`.
- **The first warm frames can be slow.** A frame measured right after the 30 s
  compile can take 30–90 ms, probably while the GPU leaves its idle power state.
  `bench` renders 50 frames before timing anything.

## Ideas to try next

- Reflections: a second march from the hit point for the ring and the ball.
- Anti-aliasing with 4 rays per pixel (the checkerboard shimmers in the
  distance), or temporal accumulation while the camera is still.
- A 16×16 kernel for finer early exit, which needs a second copy of the kernel.
- Launch frame N+1 before presenting frame N, so the GPU works while the window
  (about 1 ms per frame) is busy.
