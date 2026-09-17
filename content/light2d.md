---
title: "Project: light2d"
summary: Paint walls and lights, and rays marched through a jump-flooded distance field light the scene. Sixteen kernel launches per frame, replayed as a CUDA graph.
order: 14
section: Projects
---

Paint walls and lights into a 2D world, and a global illumination pass lights it
with soft shadows, live in a window. Each frame is a CUDA graph of cuTile
kernels:

1. `seed_init`: every occupied pixel becomes its own seed.
2. `jfa_step` × 11: **jump flooding**. Each pass looks at offsets of 512, 256, …
   1 pixels (plus one more pass at 1) and keeps the nearest seed.
3. `distance` and `nearest`: the distance field, and the scene value of each
   pixel's nearest surface.
4. `radiance`: every pixel casts rays that sphere-trace the distance field and
   add up the lights they land on, blended into a running average.
5. `compose`: shading, tone mapping and the debug views, packed into
   `0x00RRGGBB`.

The workspace README's screenshot shows the demo scene lit by three lights, then
the same frame's distance field, nearest surface (a Voronoi diagram) and raw
radiance:

![light2d: lit view, distance field, Voronoi and radiance](/project/renders/docs/images/light2d.png)

It draws on almost every earlier crate: the offset split from [life](./life.md),
now at any distance; the literal 32×32 shapes, parameter buffer and early exit
from [raymarch](./raymarch.md); and the `Submit` trait from
[filters](./filters.md), so one pipeline runs eagerly or records a graph.

## Running it

```bash
cargo run --release -p light2d -- run                  # 640×352 world, 2× window
cargo run --release -p light2d -- run --width 1280 --height 704 --scale 1
cargo run --release -p light2d -- render --view distance --out dist.png
cargo run --release -p light2d -- check                # JFA vs CPU vs exact EDT
cargo run --release -p light2d -- bench --quality 3
```

In the window, drag with the left button to paint light (Shift+drag erases) and
with the right button to paint walls. Scroll sets the brush size, `1`–`6` pick
the light color, `V` cycles the views (lit, radiance, distance, Voronoi), `Q`
cycles quality, `O` toggles the orbiting light and Space pauses it. `C` clears,
`R` resets the scene, `P` saves a screenshot, and Esc quits.

The first run compiles 15 kernels, 10 of them variants of `jfa_step` (see the
[gotchas](#gotchas-at-d92c160)). Later runs load them from the disk cache, but
still spend about 4 s building IR before the first frame.

## Layout

| File       | Role                                                                  |
| ---------- | --------------------------------------------------------------------- |
| `gpu.rs`   | the kernels, and `Pipeline` (buffers, ping-pong, even and odd graphs) |
| `cpu.rs`   | bit-exact CPU jump flood, exact distance transform, CPU radiance      |
| `scene.rs` | packed scene pixels, brush strokes, demo and random scenes            |
| `main.rs`  | `run` (window), `render` (PNG), `bench`, `check`                      |

## The scene and the buffers

Each scene pixel packs `kind << 24 | r << 16 | g << 8 | b`. Kind 0 is empty
space, 1 is a wall (blocks light, emits nothing) and 2 is a light (blocks light
and emits its color). The world rounds up to whole tiles and gains one ring of
ghost tiles, on the leading edges only, so a 640×352 world lives in a
`[384, 672]` buffer. The host scene uses that same layout and uploads as is.

Jumps start at half the next power of two above the buffer's longer side (512
for 672), halve down to 1, and then repeat 1 once. That last pass is "JFA+1",
which fixes most of the pixels plain jump flooding gets wrong.

## Jump flooding

A **seed** is the position of the nearest occupied pixel found so far. At first
every occupied pixel is its own seed and every other pixel has none. In a pass
at jump `k`, each pixel looks at the seeds held by the 9 pixels at `−k`, 0 and
`+k` on each axis and keeps the closest. Starting with big jumps spreads seeds
across the whole world in about log₂(size) passes, instead of one pixel per
pass.

The seed encoding makes "no seed" free:

```rust
/// A seed is the position of the nearest occupied pixel, packed as
/// `(row + 8192) * 32768 + (col + 8192)`.
///
/// Zero decodes to (-8192, -8192), farther from every pixel than any
/// real seed can be. So 0 means "no seed" without a special case, and
/// the zero padding cuTile uses for partial tiles reads as "no seed" too.
fn encode(row: I, col: I) -> I {
    let bias: I = fill_i(8192i32);
    (row + bias) * fill_i(32768i32) + col + bias
}
```

Each of the 9 candidates then goes through the same branch-free comparison:

```rust
/// Keep `candidate` where it is strictly closer than `best`.
fn closer(best: I, best_d: I, candidate: I, row: I, col: I) -> (I, I) {
    let d: I = dist2(candidate, row, col);
    let wins: Mask = lt_tile(d, best_d);
    (select(wins, candidate, best), select(wins, d, best_d))
}
```

## Long-range stencils: Life's split at any distance

A read at offset `d` splits into `d = q·32 + s` with `0 ≤ s < 32`: a host view
shifted by `s`, loaded at block index `+ q` (see
[Stencils](./stencils.md#strategy-2-split-the-offset-life)). Life only needed
`d = ±1`. Jump flooding needs every power of two:

| Jump `k` | Behind (`−k`)               | Ahead (`+k`)           |
| -------- | --------------------------- | ---------------------- |
| `k ≥ 32` | shift 0, block `−k/32`      | shift 0, block `+k/32` |
| `k < 32` | shift `32 − k`, block `− 1` | shift `k`, same block  |

For `k ≥ 32` all nine views are the same unshifted buffer, and only the block
offsets differ. The host builds the views, and one kernel handles every pass
with the two block offsets as scalars:

```rust
let (behind, back, front, ahead) = if k < TILE {
    (TILE - k, -1, k, 0)
} else {
    (0, -((k / TILE) as i32), 0, (k / TILE) as i32)
};
let shifts = [behind, 0, front];
let v = |r: usize, c: usize| src.slice(&[shifts[r]..rows, shifts[c]..cols]);
```

## Edges from padding rules, not branches

Life's world wraps, so its ghost ring recomputes the far side. Here reads past
the world must mean "no seed", and two cuTile rules decide how that works:

- **Partial tiles zero-pad on load.** A view shifted by `s` is `s` pixels
  shorter, so its last tile is partial and reads zeros past the end, which
  decode as "no seed".
- **An out-of-range block index is a runtime assertion.** So the kernel clamps
  every block index before loading, then uses a scalar `if` to swap in zeros
  where the real block is outside the buffer.

```rust
let up: i32 = pid.0 + back;
let up_ok: bool = up >= 0i32;
// Clamped so every load is in range; the *_ok flags decide whether
// the loaded tile is used.
let up: i32 = max(up, 0i32);
let t_um: I = um.partition(const_shape![32, 32]).load([up, j]);
let t_um: I = if up_ok { t_um } else { zero };
```

The leading ghost ring makes block `i − 1` exist for the first real tile, and
ghost tiles always store "no seed". No trailing ring is needed, because reads
past the far edge land in zero-padded partial tiles.

## Gathers: the unsafe escape hatch

A ray marching through the distance field samples wherever it has got to. No
fixed view offset expresses that, so `radiance` builds a tile of pointers from
`tensor.as_ptr()`, offsets it by `row * cols + col`, and loads through it inside
`unsafe`:

```rust
/// Read `tensor[idx]` for a whole tile of flat indices.
///
/// # Safety
/// Every index must be inside `tensor`, which must be contiguous.
unsafe fn gather_i32(tensor: &Tensor<i32, { [-1, -1] }>, idx: I) -> I {
    let base: PointerTile<*const i32, { [] }> = pointer_to_tile(tensor.as_ptr());
    let base: PointerTile<*const i32, { [1, 1] }> = base.reshape(const_shape![1, 1]);
    let ptrs: PointerTile<*const i32, { [32, 32] }> = base.broadcast(const_shape![32, 32]);
    let ptrs: PointerTile<*const i32, { [32, 32] }> = ptrs.offset_tile(idx);
    let (values, _token): (I, Token) = load_ptr_tko(
        ptrs,
        ordering::Weak,
        None::<scope::TileBlock>,
        None,
        None,
        None,
        Latency::<0>,
    );
    values
}
```

The kernel meets the safety condition by clamping every row and column into the
buffer before it builds the index. `nearest` uses the same helper to look up the
scene pixel each seed points at. This is the book's tutorial 9, pointer
addition, put to work.

## Marching rays in 2D

Every pixel casts rays at evenly spaced angles, rotated by a per-pixel jitter:
interleaved gradient noise, scrolled by the golden ratio each frame. A ray reads
the distance at its current pixel and steps that far less one pixel, but at
least one pixel. It stops when it lands on an occupied pixel (distance below
0.5) or leaves the world. A ray that lands on a light adds the light's color,
and one that escapes adds a faint ambient sky.

The march has `raymarch`'s loop shape: chunks of `check_every` steps, then a
tile-wide check that breaks once no ray is marching. That makes the whole frame
1.5–1.9× faster.

| Quality    | Rays per pixel | Steps | Check every |
| ---------- | -------------- | ----- | ----------- |
| 1 (low)    | 8              | 32    | 8           |
| 2 (medium) | 16             | 48    | 8           |
| 3 (high)   | 32             | 64    | 16          |

`run` defaults to high, and `render` and `bench` to medium.

## Two graphs for temporal accumulation

A few rays per pixel are noisy, so `radiance` blends each frame into a running
average that it reads from last frame's buffer. A captured graph bakes in which
buffer is read and which is written, so the pipeline captures two graphs and
alternates between them:

```rust
/// One whole frame per graph, for even and odd frames.
pub fn capture(&mut self) -> Result<[CudaGraph<()>; 2], Error> {
    let stream = self.stream.clone();
    let even = CudaGraph::scope(&stream, |s| Ok(self.run(s, 0)?))?;
    let odd = CudaGraph::scope(&stream, |s| Ok(self.run(s, 1)?))?;
    Ok([even, odd])
}
```

`run` is written against the `Submit` trait, so the same code also runs eagerly.
A frame is 16 kernel launches, so graphs matter again: at 640×352 with 16 rays,
replay takes 0.65 ms against 1.14 ms eager. The 11 flood passes alone take 0.11
ms.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, demo scene. GPU frame times
are for the whole pipeline replayed as one CUDA graph.

| World, rays × steps           | CPU flood | CPU radiance | GPU flood | GPU frame | GPU frame, eager | GPU frame, no early exit |
| ----------------------------- | --------- | ------------ | --------- | --------- | ---------------- | ------------------------ |
| 640×352, 16 × 48              | 27 ms     | 21 ms        | 0.11 ms   | 0.65 ms   | 1.14 ms          | 0.97 ms                  |
| 640×352, 32 × 64              | 28 ms     | 36 ms        | 0.12 ms   | 1.25 ms   | 1.78 ms          | 2.21 ms                  |
| 1280×704, 32 × 64 (12 passes) | 37 ms     | 161 ms       | 0.27 ms   | 3.40 ms   | 3.94 ms          | 6.50 ms                  |

The window at its defaults (640×352 world, 32 rays) runs at about 355 fps, or
460 fps at 16 rays. Each window frame also uploads the scene (0.13 ms) and
downloads the image (0.22 ms).

**Everything matches the CPU exactly.** Seeds, distances and nearest-surface
colors are integer or correctly rounded math, so they match bit for bit, both
eager and replayed from a graph on a new scene. The float radiance matched too,
with zero difference on every pixel of the demo scene, `cos` and `sin` included.
`raymarch` only got within 1/255.

`check` measures jump flooding against an exact Euclidean distance transform, on
random discs plus 3000 isolated dots:

| World     | Plain JFA                     | JFA+1                        |
| --------- | ----------------------------- | ---------------------------- |
| 640×352   | 51 px wrong, worst by 0.92 px | 1 px wrong, worst by 0.27 px |
| 1920×1088 | 51 px wrong, worst by 0.94 px | 1 px wrong, worst by 0.12 px |

## Gotchas at `d92c160`

- **Kernels are specialized on divisibility, not just generics.** The JIT cache
  key records the largest power-of-two divisor, capped at 16, of every integer
  scalar and of every tensor's shape, strides and base pointer. Passes differ in
  block offsets (±16 or more, 8, 4, 2, 1) or view shifts (16, 24/8, 28/4, 30/2,
  31/1), so `jfa_step` compiles into 10 variants. Passes in the same class share
  one, and any world size gets the same 10 keys, because buffers are always
  whole tiles. Each quality preset has its own `radiance` key too (8, 16 and 32
  rays), so pressing `Q` compiles a new one the first time. Every variant costs
  about 270–320 ms of IR building at each startup, even when the kernel comes
  from the disk cache. See
  [Compilation](./compilation.md#what-makes-a-new-specialization).
- **Helper names can collide with DSL ops.** A device function called `unpack`
  failed with "duplicate functions are not supported", because Tile IR already
  has `pack` and `unpack` ops.
- **`convert_tile` needs an annotated `let`.** As a function's tail expression
  it failed with "Failed to get type parameters for convert_tile".
- **Pass a `&mut Tensor` as a read-only input with `&*src`.** Otherwise the
  launcher looks for `DeviceOp` on `&mut Tensor`.
- **Out-of-range partition loads assert at runtime; partial tiles zero-pad.**
  Hence the clamp plus scalar `if` in `jfa_step`.

## Ideas to try next

- Radiance cascades for noise-free 2D GI, instead of more rays.
- A bounce: let walls reflect the radiance arriving at them.
- Point lights with SDF soft shadows (one march per light instead of per ray).
- Half-resolution radiance with an edge-aware upsample for 4K worlds.
