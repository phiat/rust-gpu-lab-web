---
title: "Project: cloth"
summary: Position based dynamics on a grid of particles. Constraints colored into twelve batches so both ends of a link can move, one kernel for every direction, and 102 launches a frame that a CUDA graph replays 3.3× faster.
order: 17
section: Projects
---

A grid of particles joined by distance constraints hangs from pins, blows in a
gusty wind and drapes over a sphere you move with the mouse. The solver is
**position based dynamics**: a Verlet step, then passes that each move every
particle a little toward satisfying its links. `cloth` is the seventh demo and
the second in the [roadmap](./getting-started.md#whats-next). The CPU reference
runs the same operations in the same order.

![cloth: a curtain of 256×160 particles after 240 frames of wind, draped over the sphere](/images/cloth.png)

It's still a stencil problem, like [life](./life.md) and [sand](./sand.md):
every particle reads its neighbors from shifted views of one tensor. Two things
are new. The grid holds **objects** with several fields (a position and an
inverse mass) instead of one number, and the solver is **iterative**: 102 kernel
launches per frame, most of them cheap.

## Running it

```bash
cargo run --release -p cloth -- run                       # 256×160 particles, 960×640 window
cargo run --release -p cloth -- run --cols 512 --rows 320
cargo run --release -p cloth -- render --frames 240        # demo → cloth.png
cargo run --release -p cloth -- bench                      # CPU vs GPU eager vs GPU graph
cargo run --release -p cloth -- check --frames 30          # GPU vs CPU, particle for particle
```

In the window the sphere follows the mouse, right drag orbits the camera and the
wheel changes the wind. `1`–`4` pin the top row, a few points of it, the two
corners, or nothing (each resets the cloth). `W` toggles the wind, `Space`
pauses, `R` resets, `P` saves a screenshot and `Esc` quits.

`--substeps` (default 4) is the number of Verlet steps per frame and
`--iterations` (default 2) the solver iterations per substep. Each iteration is
12 passes, one per constraint batch.

## Layout

```
cloth/src/world.rs   layout, particle encoding, pins, constraint batches, anchors, physics, camera
cloth/src/gpu.rs     integrate / pair / normals / shade kernels and the Pipeline
cloth/src/cpu.rs     the same math on rayon
cloth/src/raster.rs  CPU triangle rasterizer for the projected, lit particle grid
cloth/src/main.rs    clap CLI: run (window) / render (PNG) / bench / check
```

## A grid of objects, not pixels

A particle is four `f32`s: position and inverse mass `w`. The buffer is a
`[rows, cols, 4]` tensor, loaded as `[32, 32, 4]` tiles, with `extract` picking
channels apart and `cat` putting them back:

```rust
/// A tile of particles: x, y, z, inverse mass.
type V = Tile<f32, { [32, 32, 4] }>;

fn channel(v: V, c: i32) -> F {
    let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
    let idx: Tile<i32, { [] }> = scalar_to_tile(c);
    let t: Tile<f32, { [32, 32, 1] }> = extract(v, [zero, zero, idx]);
    t.reshape(const_shape![32, 32])
}

fn pack4(x: F, y: F, z: F, w: F) -> V {
    let x: Tile<f32, { [32, 32, 1] }> = x.reshape(const_shape![32, 32, 1]);
    // ... same for y, z, w ...
    let xy: Tile<f32, { [32, 32, 2] }> = cat(x, y, 2i32);
    let zw: Tile<f32, { [32, 32, 2] }> = cat(z, w, 2i32);
    cat(xy, zw, 2i32)
}
```

Pinned particles have inverse mass 0, and so does the ghost ring around the
grid. "Never moves" then falls out of the arithmetic instead of needing a
special case: when a link is projected, each end moves by its share of the mass,
`w / (w + w_neighbor)`, and a share of zero is no move.

Everything the kernels need per frame (time step, gravity, wind, the sphere, the
stiffnesses, the camera) goes through one 64-slot `f32` parameter buffer, as in
[raymarch](./raymarch.md). No kernel takes an integer argument, so the JIT
compiles each kernel once.

## Red/black, done right

The roadmap said "an iterative solver with red/black passes", and the first
version did exactly that: color the particles like a checkerboard, and in each
pass move one color toward satisfying all twelve of its links (four structural,
four shear, four bend) while the other color holds still.

It exploded, even at 1% gravity. Only the structural links join particles of
_different_ colors. Shear and bend links join particles of the _same_ color, so
both ends moved at once, each against the other's stale position, and the error
compounded pass after pass. Pure Jacobi (everybody moves every pass, by half)
didn't explode, but converged too slowly to hold a curtain up.

The fix is to color the **constraints**, not the particles. Split the links into
batches such that no particle has two links in a batch:

```rust
/// One batch of constraints, all solved in the same pass: every particle
/// is in at most one link of a batch, so both ends of a link can move by
/// their share without stepping on another link's result. This is
/// red/black for links: `kind` picks the link direction (0 vertical, 1
/// horizontal, 2 the `\` diagonal, 3 the `/` diagonal), `lg` the stride
/// (`1 << lg`: 1 for structural and shear, 2 for bend), and `b` which of
/// the two interleaved sets of links along that direction.
pub struct Batch {
    pub b: i32,
    pub kind: i32,
    pub lg: i32,
    /// Stiffness slot: 0 structural, 1 shear, 2 bend.
    pub k: usize,
}
```

Vertical links with an even top row, then an odd top row, then horizontal, then
each diagonal, then the two-apart bend links: 12 batches. One pass projects one
batch, every particle has at most one partner, so both ends move by their share
and the pair ends up exactly at rest length. That's Gauss-Seidel over batches,
which is what red/black is for a one-link stencil.

The same `pair` kernel serves all 12 batches. The batch's direction, stride and
parity come from a `[4]` view of a small table, the trick from
[sand](./sand.md#passing-per-launch-integers-without-jit-variants), and the two
shifted views passed in are what make one batch's partner "up or down" and
another's "left or right":

```rust
// Which way is my partner? (`Batch::partner`)
let b: I = pass_entry(pass_buf, 0i32);
let kind: I = pass_entry(pass_buf, 1i32);
let lg: I = pass_entry(pass_buf, 2i32);
let horizontal: Mask = eq_tile(kind, ione);
let coord: I = select(horizontal, col, row);
let group: I = shri(coord, lg);
let plus_side: Mask = eq_tile(andi(group + b, ione), izero);
// ...
let qp: V = load4(plus, up, left);
let qm: V = load4(minus, up, left);
let qx: F = select(plus_side, channel(qp, 0i32), channel(qm, 0i32));
```

Passes are in-place in spirit but ping-pong in practice: a pass reads one buffer
and writes the other, so neighbors always see a consistent snapshot.

## One kernel, whatever the direction

A view shifted by −1 row and one shifted by +2 columns have different shapes,
and cuTile specializes a kernel on the divisibility of every argument's shape
(see [Compilation](./compilation.md#what-makes-a-new-specialization)). A kernel
that takes "minus" and "plus" views for six different directions would compile
in several variants.

`views` therefore cuts every view to the **same shape**, the buffer minus 31
rows and columns, starting inside the ghost ring, and every kernel loads all of
them at the same block `[i − 1, j − 1]`:

```rust
/// Views of `src` shifted by each `(dr, dc)` offset, so that block
/// `[i - 1, j - 1]` of every view holds the `(dr, dc)` neighbors of block
/// `[i, j]` of `src`: view (dr, dc) starts at row `TILE + dr`, column
/// `TILE + dc`, which the ghost ring makes possible for offsets of either
/// sign. Every view has the same shape, so a kernel's argument pattern is
/// the same whichever offsets a launch uses, and the JIT (which
/// specializes on shape divisibility) builds it once. The length leaves
/// the last ghost block a partial tile, which loads zero-padded rather
/// than asserting.
fn views<'a>(src: &'a Tensor<f32>, offsets: &[(i32, i32)]) -> Result<Vec<TensorView<'a, f32>>, Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let (lr, lc) = (rows - 2 * TILE + 1, cols - 2 * TILE + 1);
    offsets
        .iter()
        .map(|&(dr, dc)| {
            let (r, c) = ((TILE as i32 + dr) as usize, (TILE as i32 + dc) as usize);
            src.slice(&[r..r + lr, c..c + lc, 0..4])
        })
        .collect()
}
```

This is a third way to build a stencil view, after `life`'s offset split and
`filters`' valid convolution (see [Stencils](./stencils.md)). The ghost ring is
what lets a view start at "row 32 − 2" without going negative, and the odd
length leaves the last ghost block a [partial tile](./glossary.md), which loads
zero-padded rather than asserting. One `pair` kernel, one compile: 0.75 s to the
first frame with the cache warm.

## Long range attachments

Gauss-Seidel moves information one link per pass, so a 160-row curtain under
gravity sagged by tens of percent no matter how many iterations were affordable:
the top row is pinned, and the bottom rows fall until the correction reaches
them.

The cheap, standard answer (Kim, Chentanez and Müller, 2012) is to give every
particle its nearest pin as an **anchor** and clamp it to within the flat-cloth
distance of that anchor, every pass:

```rust
// Long range attachment: never farther from the anchor than the
// flat cloth allows.
let free: Mask = gt_tile(w, zero);
let a: V = load4(anchor, i, j);
let reach: F = channel(a, 3i32);
let vx: F = x - channel(a, 0i32);
// ... vy, vz ...
let dist: F = sqrt_f(vx * vx + vy * vy + vz * vz);
let pull: F = reach / max_tile(dist, fill(1e-6f32));
let far: Mask = and(free, and(gt_tile(reach, zero), gt_tile(dist, reach)));
let x: F = select(far, ax + vx * pull, x);
```

The anchors are a tensor computed on the host when the pins change. With them,
stretch stays under 1% for the pinned cases at 2 iterations per substep, and the
free cloth (no pins, no anchors) simply falls onto the floor and the sphere.

## Where the tile model stops: rasterizing on the CPU

Projecting and lighting every particle is a per-element job and runs on the GPU:
`shade` writes screen x, y, depth and a packed color for each particle. Turning
the quads between particles into pixels is a **scatter**: each triangle touches
an unpredictable set of pixels, and the tile model has no primitive for that. So
the last step is a small rasterizer on rayon, two Gouraud-shaded triangles per
grid cell with a depth buffer, and triangles bucketed into horizontal bands so
every pixel is owned by one thread. Item 8 on the roadmap (boids) will hit the
same wall from the other side.

## Results

RTX 4070 Ti SUPER, i9-14900KF (8 P-cores + 16 E-cores), WSL2. Per frame: 4
substeps × (1 integrate + 24 constraint passes), plus normals and shading, 102
launches.

| Particles | CPU rayon         | GPU eager | GPU graph | Speedup |
| --------- | ----------------- | --------- | --------- | ------- |
| 256×160   | 18.6 ms (8 thr.)  | 3.01 ms   | 0.91 ms   | 20×     |
| 512×320   | 59.9 ms (28 thr.) | 3.60 ms   | 1.34 ms   | 45×     |

This is the biggest win for [CUDA graphs](./cuda-graphs.md) in the workspace so
far: 102 launches of small kernels per frame, and the graph replays them 3.3×
faster than eager launches. The remaining per-frame costs are the CPU rasterizer
(2.8 ms at 960×640 for 81k triangles) and the parameter upload plus screen
download through `tilekit::Pinned` (0.07 ms). The window runs at about 180 fps,
rasterizer bound.

The CPU column has a twist. At 256×160 each pass is about 1 ms of
single-threaded work, and rayon's default pool of 28 threads makes it _slower_
(166 ms per frame) than 8 threads (18.6 ms): every pass ends with a join that
waits for the slowest thread, and on a hybrid chip that's an E-core. At 512×320
the passes are long enough for all 28 threads to pay off. `bench` times both and
reports the better.

`check` runs the GPU and CPU side by side, alternating eager and graph frames,
and compares every particle after each frame. Frame 0 matches to 3×10⁻⁸
spacings, that is, to float rounding; by frame 30 the difference has grown to
about 0.5 spacings. That isn't a bug: a cloth in gusty wind against a sphere is
chaotic, and the rounding differences (the GPU fuses multiplies and adds where
the CPU doesn't) roughly double every frame. Both stay at 0.3–3% mean stretch
throughout, and the pictures are indistinguishable. Compare
[sand](./sand.md#randomness-as-a-hash), where integer math made the match exact.

## Gotchas at `d92c160`

- **Helper names collide with builtins.** A module function called `pack`,
  `load` or `dot` fails with "duplicate functions are not supported" because
  `cutile::core` already has those names. Renamed to `pack4`, `load4`, `dot3`.
- **`select` needs a mask of the tile's own shape.** Selecting between two
  `[32, 32, 4]` tiles with a `[32, 32]` mask is a type error; select each
  channel instead.
- **Block indices are scalars.** A kernel can't pick which block of a view to
  load from a tile value, so a batch can't decide its own view offsets; hence
  the same-shape views loaded at the same block for every direction.
- **Two `&mut Tensor` outputs are fine**: `integrate` writes the new positions
  and the new "previous" positions in one kernel.

## Ideas to try next

- Self collisions: a spatial hash on the GPU is a scatter, but a coarse grid
  density (splat particles into cells, gather in the next pass) would give a
  repulsion term.
- Tearing: drop a link when it stretches past a limit; needs a per-link flag
  tensor, and a mesh that can have holes.
- Bending as a dihedral constraint instead of a distance two apart.
- XPBD compliance, so stiffness stops depending on the iteration count.
- Soft bodies: the same solver on a 3D grid of particles with volume
  constraints.
