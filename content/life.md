---
title: "Project: life"
summary: Conway's Game of Life on a torus. One stencil kernel with a ghost ring, replayed through a CUDA graph in a 60 fps window.
order: 11
section: Projects
---

Game of Life on a wrapping world. One cuTile stencil kernel computes a
generation, a captured CUDA graph replays many generations per launch, and a
`minifb` window draws the result. `bench` runs the same seed on the CPU, eagerly
on the GPU, and through the graph, then checks all three worlds match cell for
cell.

The ideas behind it have their own notes: [Stencils](./stencils.md) and
[CUDA graphs](./cuda-graphs.md). [light2d](./light2d.md) later reuses the same
offset split for jumps of up to 512 pixels.

## Running it

```bash
cargo run --release -p life -- run                      # 1280×736 world, 60 fps window
cargo run --release -p life -- run --width 640 --height 360 --tile 16 --scale 2
cargo run --release -p life -- run --gens-per-frame 20  # faster evolution
cargo run --release -p life -- bench                    # 4096²: CPU vs GPU eager vs GPU graph
cargo run --release -p life -- bench --size 256 --tile 32 --gens 100
```

In the window, Space pauses, R reseeds and Esc quits. The title bar shows fps,
generations per second and the time per graph launch.

## Layout

| File      | Role                                                                 |
| --------- | -------------------------------------------------------------------- |
| `main.rs` | clap CLI (`run`, `bench`), `Layout` with ghost-tile padding, drawing |
| `gpu.rs`  | `life_step` kernel, `World` with ping-pong buffers, graph capture    |
| `cpu.rs`  | rayon reference step and a seeded random soup                        |

## The world buffer

`Layout` rounds the world up to a multiple of the tile size (720 rows become 736
at tile 32), then adds one tile of padding on every side:

```rust
pub fn buf_rows(&self) -> usize {
    self.rows + 2 * self.tile
}
```

Logical cell `(r, c)` lives at buffer `[tile + r, tile + c]`. `pad` fills the
ghost ring by wrapping around, and only runs when the host loads a new world.
After that the kernel keeps the ring correct.

## The kernel

`life_step` takes the output plus nine views of the current generation, named by
row offset (up, middle, down) and column offset (left, middle, right). Each view
is shifted by `B − 1`, 0 or 1:

```rust
/// The nine offset views of `src`, in `life_step` argument order.
fn neighborhood<'a>(src: &'a Tensor<u8>, lay: &Layout) -> Result<[TensorView<'a, u8>; 9], Error> {
    let (rows, cols) = (lay.buf_rows(), lay.buf_cols());
    // Offset -1 is (B-1 at block i-1), 0 is (0 at i), +1 is (1 at i).
    let offsets = [lay.tile - 1, 0, 1];
    let view = |r: usize, c: usize| src.slice(&[offsets[r]..rows, offsets[c]..cols]);
    // ...
}
```

Inside, a ghost tile program swaps its block index for the interior tile it
mirrors, then loads each view at the right block:

```rust
let t_ul: Tile<u8, { [B, B] }> = p_ul.load([up, left]);
let t_um: Tile<u8, { [B, B] }> = p_um.load([up, j]);
let t_ur: Tile<u8, { [B, B] }> = p_ur.load([up, j]);
let t_ml: Tile<u8, { [B, B] }> = p_ml.load([i, left]);
let t_mm: Tile<u8, { [B, B] }> = p_mm.load([i, j]);
```

[Stencils](./stencils.md#strategy-2-split-the-offset-life) explains why `ur`
loads at block `j` while `ul` loads at `left`.

## The rule without branches

Sum the full 3×3 block, the cell included. A cell is alive next generation if
the sum is 3 (born, or survives with 2 neighbors), or if the sum is 4 and it's
alive now (survives with 3):

```rust
let sum: Tile<u8, { [B, B] }> =
    t_ul + t_um + t_ur + t_ml + t_mm + t_mr + t_dl + t_dm + t_dr;
// ...
let is3: Tile<bool, { [B, B] }> = eq_tile(sum, three);
let is4: Tile<bool, { [B, B] }> = eq_tile(sum, four);
let survives: Tile<u8, { [B, B] }> = select(is4, t_mm, zero);
out.store(select(is3, one, survives));
```

The rayon reference writes the same rule directly:
`*px = (sum == 3 || (sum == 4 && mid[c] == 1)) as u8;`

## Eager steps and graph replays

`World` holds two padded buffers. The eager step writes `back`, then swaps:

```rust
pub fn step_eager(&mut self) -> Result<(), Error> {
    {
        let n = neighborhood(&self.front, &self.layout)?;
        step_op(&mut self.back, &n, self.layout.tile).sync_on(&self.stream)?;
    }
    std::mem::swap(&mut self.front, &mut self.back);
    Ok(())
}
```

A graph can't swap, so `capture` records pairs of steps (front → back → front)
and rounds the generation count up to even. Reseeding copies into `front` with
`api::memcpy` instead of allocating, so the captured graph stays valid.

The window keeps a per-cell "heat" value on the host. A live cell is 255, and a
dead one fades by 24 per frame, which leaves a short cyan trail behind moving
patterns.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, per generation:

| World | Tile | CPU rayon | GPU eager | GPU graph | Graph vs eager |
| ----- | ---- | --------- | --------- | --------- | -------------- |
| 256²  | 32   | 1.633 ms  | 0.035 ms  | 0.007 ms  | 5.0×           |
| 1024² | 32   | 0.877 ms  | 0.046 ms  | 0.021 ms  | 2.2×           |
| 4096² | 64   | 4.117 ms  | 0.352 ms  | 0.332 ms  | 1.06×          |

At 1024² and larger the GPU tops out near 50 billion cells per second. Each cell
reads 9 bytes and writes 1, which is about 500 GB/s against the card's 672 GB/s
memory bandwidth. Memory traffic, not compute, looks like the limit.

## Lessons recorded in the code

- **A 10-argument kernel is fine.** It works with `s.record(...)` and `sync_on`.
  Only the `.first()` helper stops at 6-element tuples, and `life` never calls
  it.
- **Warm up outside the graph.** `run` takes one eager step to trigger the JIT,
  reloads the seed, then captures.
- **Recording doesn't advance the world.** The benchmark confirms the state
  after replays matches the CPU after exactly `launches × gens` generations.
- **The window is X11 only.** `minifb` is built with just the `x11` feature,
  since WSLg provides an X server and Wayland would need extra dev packages.

## Ideas to try next

- Paint cells with the mouse by copying a small host patch into `front`.
- Load RLE patterns, such as a Gosper glider gun wrapping around the torus.
- Pack 8 cells per byte to cut the memory traffic that limits large worlds.
- Try a particle simulation to exercise float kernels instead of `u8` stencils.
