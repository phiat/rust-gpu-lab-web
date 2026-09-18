---
title: Glossary
summary: Short definitions of the terms that show up everywhere.
order: 30
section: Reference
---

**Tile**: an immutable, statically shaped array fragment that lives in registers
during a kernel. `Tile<E, S>`. All arithmetic happens on tiles.

**Tensor**: a multi-dimensional array in GPU global memory. Kernels take
`&Tensor<E, S>` (read-only) or `&mut Tensor<E, S>` (writable). You can load from
and store to it, but you can't do arithmetic on it directly.

**Partition**: a tiled view of a tensor. On the host it splits a writable output
into disjoint pieces and sets the launch grid. On the device it lets a kernel
load arbitrary tiles of a read-only input.

**Tile program / tile block / tile thread**: one logical unit of concurrent
execution. It runs the entry function once for one cell of the grid.

**Launch grid**: the `(x, y, z)` extents of tile programs. Usually inferred as
`ceil(tensor_dim / partition_dim)` per axis; tensor dim 0 maps to grid x.

**`program_id(axis)` / `num_programs(axis)`**: this program's coordinate, and
the grid extent, along one axis.

**Entry point**: a `#[cutile::entry()]` function the host can launch.

**Device function**: an unmarked function in a `#[cutile::module]`, inlined into
entry points.

**Launcher**: the host-side function the macro generates for an entry point.
Calling it returns a `DeviceOp`.

**DeviceOp**: a lazy description of GPU work. It runs when synced (`.sync()`,
`.sync_on(&stream)`), awaited, or captured into a CUDA graph.

**Stream**: a CUDA work queue. Ops on one stream run in order; ops on different
streams may overlap.

**Static / dynamic dimension**: a compile-time constant dimension, or `-1`
resolved at runtime. A static dim is part of the specialization; a dynamic one
only counts through its power-of-two divisibility.

**Specialization**: one compiled kernel variant for one entry function, one GPU
architecture, and one set of compile-time inputs.

**Tile IR**: the CUDA Tile intermediate representation the Rust AST is lowered
into.

**cubin**: the compiled GPU binary for one specialization and one architecture.

**SM (`sm_89`, `sm_90`, …)**: the GPU architecture target a cubin is built for.

**`load_like(out)`**: load the tile of an input tensor or view that lines up
with the block of `out` this program writes.

**Generic / concrete tile type**: how the JIT names a tile's shape, either with
const generics (`{[B, B]}`) or with numbers (`{[32, 32]}`). `+`, `-` and `*`
require both operands to use the same form.

**`exti` / `trunci`**: widen or narrow an integer tile (`u8` → `i32` → `u8`).
`convert_tile` only converts between integers and floats.

**Stencil**: a kernel whose output element depends on a neighborhood of input
elements. See [Stencils](./stencils.md).

**Valid convolution**: a stencil that only reads non-negative offsets, so its
output is smaller than its input by twice the radius.

**Ghost tile**: a padding tile around the world whose program computes the
interior tile on the opposite edge, which keeps a wrapping world correct. In
`sand` the ring is a wall instead, and is never updated.

**CUDA graph**: a recorded sequence of GPU work replayed with one driver call.
See [CUDA graphs](./cuda-graphs.md).

**Ping-pong buffers**: two preallocated buffers that alternate as input and
output, so repeated steps never allocate.

**Signed distance function (SDF)**: a function that returns, for any point, a
lower bound on the distance to a shape's surface. `raymarch` builds its whole
scene from them.

**Sphere tracing**: marching a ray forward by the SDF value at each step, which
can't overshoot a surface when the SDF is a true lower bound.

**Partial tile**: an edge tile that hangs past the end of a tensor whose size
isn't a multiple of the tile size. Stores there are masked and loads read 0.

**Jump flooding (JFA)**: building a nearest-seed map in about log₂(size) passes.
Each pass at jump `k` checks the seeds held `k` pixels away in 9 directions,
then `k` halves. **JFA+1** adds one more pass at 1 to fix most errors. See
[Project: light2d](./light2d.md).

**Distance field**: each pixel's distance to the nearest occupied pixel. The map
of which surface is nearest is a **Voronoi diagram**.

**Gather**: reading input at computed, per-element indices instead of fixed
offsets. In cuTile it takes a tile of raw pointers and `load_ptr_tko` inside
`unsafe`.

**Temporal accumulation**: blending each frame into a running average so noise
from a few random samples settles over time.

**Pinned (page-locked) memory**: host memory the OS can't swap out or move, so
the GPU can copy to and from it directly. Transfers from ordinary **pageable**
memory are staged through a pinned buffer by the driver first. See
[tilekit](./tilekit.md).

**Parameter buffer**: a small device tensor holding per-frame inputs, so a
captured CUDA graph can see new values without recapturing.

**Margolus neighborhood**: a block cellular automaton scheme. The grid is split
into 2×2 blocks, each block updates on its own, and the partition shifts by one
cell on alternate passes. Cells can move without two of them ever being written
to the same place. See [Project: sand](./sand.md).

**Hashed randomness**: a per-cell or per-block random word computed as an
integer hash of coordinates and a per-pass salt, since kernels have no RNG. A
CPU running the same hash gets the same bits, so the results can be compared
exactly. `sand` uses lowbias32.

**Position based dynamics (PBD)**: a simulation method that integrates positions
(Verlet), then repeatedly projects constraints by moving particles directly,
instead of computing forces. See [Project: cloth](./cloth.md).

**Verlet integration**: advancing a position from its current and previous
positions, `x + (x − x_prev) + a·dt²`, so velocity is implicit.

**Jacobi / Gauss-Seidel**: two ways to iterate a solver. Jacobi updates every
element from the previous pass's values; Gauss-Seidel uses updated values as
soon as they exist, which converges faster. **Red/black** ordering makes
Gauss-Seidel parallel for a one-link stencil; `cloth` generalizes it by coloring
constraints into batches.

**Constraint batch**: a set of links in which no particle appears twice, so all
of them can be projected in one pass with both ends moving.

**Long range attachment (LRA)**: clamping each particle to within its flat-cloth
distance of its nearest pin, every pass, so a curtain doesn't sag while
corrections propagate one link per pass.

**Scatter**: writing to computed, per-element destinations, the mirror of a
gather. The tile model has no primitive for it; `cloth` rasterizes on the CPU
for that reason.
