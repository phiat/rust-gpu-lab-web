---
title: "Project: sand"
summary: A falling sand game on the GPU. Margolus 2×2 blocks move cells without write conflicts, a hash stands in for a random number generator, and the CPU matches cell for cell.
order: 16
section: Projects
---

Sand piles up, water levels out and drains off shelves, oil floats on water,
fire rises, wood catches and burns down to smoke, and water puts embers out.
Paint any of it with the mouse. `sand` is the sixth demo and the first of the
game-dev series in the [roadmap](./getting-started.md#whats-next).

It's a cellular automaton, like [life](./life.md), but its cells **move**, and
that changes the shape of the kernel. Every cell rule is integer math, so the
GPU is checked against the CPU cell for cell, over 60 frames of random blobs.

![sand: the demo scene after 300 frames, with the hut on fire](/images/sand.png)

The demo scene after 300 frames: sand and water taps running, a pool draining
off a shelf, oil on the floor, and the wooden hut on fire.

## Running it

```bash
cargo run --release -p sand -- run                  # 640×352 window at 2×
cargo run --release -p sand -- run --width 960 --height 512 --scale 1
cargo run --release -p sand -- render --frames 300  # demo scene → sand.png
cargo run --release -p sand -- bench                # CPU vs GPU eager vs GPU graph
cargo run --release -p sand -- check --frames 60    # exact CPU/GPU comparison
```

In the window, left drag paints, right drag erases and the wheel sets the brush
size. `1`–`8` pick sand, water, oil, wood, wall, fire, smoke and ember. `Space`
pauses, `E` toggles the sand and water taps, `C` clears, `R` resets the demo
scene, `P` saves a screenshot and `Esc` quits.

`--passes` sets the Margolus passes per frame (default 4, must be even). A pass
moves a cell at most one step, so more passes means faster falling at the same
frame rate.

## Layout

```
sand/src/world.rs  elements, the cell encoding, buffer layout, host-side painting
sand/src/gpu.rs    the step kernel (one Margolus pass), paint and render kernels, Pipeline
sand/src/cpu.rs    the same rules in plain Rust, on rayon
sand/src/main.rs   clap CLI: run (window) / render (PNG) / bench / check
```

## The cell

A cell is one `i32`. The element sits in the low byte, a 4-bit shade in bits
8–11 so a grain keeps its tint as it moves, and liquids carry a direction bit:

```rust
pub const EMPTY: i32 = 0;
pub const SAND: i32 = 1;
pub const WATER: i32 = 2;
pub const OIL: i32 = 3;
pub const WOOD: i32 = 4;
pub const WALL: i32 = 5;
pub const FIRE: i32 = 6;
pub const SMOKE: i32 = 7;
/// Wood that is on fire: stays put, throws off flames, burns out.
pub const EMBER: i32 = 8;

/// Liquids flow in one direction until blocked, then turn around.
pub const DIR: i32 = 1 << 12;
```

The world is 32×32 tiles with a ring of ghost tiles around it, as in `life`, but
the ring is filled with `WALL` and never updated. No rule ever has to ask "is
this the edge?": the edge is a wall, and walls don't move.

## Moving cells without write conflicts

In Life, every cell's next state depends only on the current generation, so each
tile program reads its neighborhood and writes its own tile. A cell that
**falls** is different: it has to agree with the cell below about who ends up
where, and two grains above one hole must not both fall into it. On a GPU nobody
can lock anything.

The classic answer is the **Margolus neighborhood**: partition the grid into 2×2
blocks, update each block on its own (its four cells only trade places among
themselves), and shift the partition by one cell on alternate passes so that
blocks overlap over time. Every cell is written by exactly one block, so there
are no conflicts and no atomics.

In a tile kernel that looks like this. Each cell reads its 3×3 neighborhood from
the nine shifted views of [life](./life.md#the-kernel), works out which corner
of its block it is from the pass parity, and picks out its three block mates
with `select`:

```rust
// Which corner of its block is this cell?
let (row, col) = coords();
let parity: I = andi(pass, fill(1i32));
let pr: I = andi(row + parity, fill(1i32));
let pc: I = andi(col + parity, fill(1i32));
let top: Mask = eq_tile(pr, fill(0i32));
let lft: Mask = eq_tile(pc, fill(0i32));
// Rows of the block: for a top cell they are (this row, below);
// for a bottom cell (above, this row). Same for columns.
let a_l: I = select(top, t_ml, t_ul);
let a_m: I = select(top, t_mm, t_um);
let a_r: I = select(top, t_mr, t_ur);
let c_l: I = select(top, t_dl, t_ml);
let c_m: I = select(top, t_dm, t_mm);
let c_r: I = select(top, t_dr, t_mr);
let a: I = select(lft, a_m, a_l);
let b: I = select(lft, a_r, a_m);
let c: I = select(lft, c_m, c_l);
let d: I = select(lft, c_r, c_m);
```

Then it runs the block rule on `a b / c d` and keeps the result for its own
corner:

```rust
let (na, nb, nc, nd) = update(a, b, c, d, r);
let mine: I = select(top, select(lft, na, nb), select(lft, nc, nd));
```

All four cells of a block compute the same thing, redundantly, and that costs
nothing next to the reads. Nothing has to be shared between them.

## Randomness as a hash

There's no random number generator on the device, so every block gets one 32-bit
word from a hash of its block coordinates and a salt that changes every pass:

```rust
let bx: I = shri(col - pc, fill(1i32));
let by: I = shri(row - pr, fill(1i32));
let salt: I = param(params, 0i32) * fill(64i32) + pass;
let r: I = hash(xori(bx, hash(xori(by, hash(salt)))));
```

`hash` is lowbias32, a five-line integer mixer. Its four bytes drive the
reactions (one byte per cell of the block), and a second hash of the word
supplies the bits that pick slide order and gas wandering. The CPU reference
does the same hashes, in the same order, and that's what makes the comparison
exact: `check` runs 60 frames at 640×352 with random blobs of every element and
the taps running, alternating eager and graph frames, and every cell matches.

Two things bit on the way. The kernel's `*` operator emits `muli` without an
overflow flag, so it wraps like `wrapping_mul`, which is what a hash wants
(`muli(.., overflow::None)` fails to serialize, as `trunci` did in `filters`).
And `0x846c_a68b` doesn't fit an `i32`: typing its two's complement by hand went
wrong twice, and both times the GPU was right and the CPU reference had the
typo. Compute the constant (`0x846c_a68bu32 as i32`) instead of writing it.

## Rules that look right

The block rule is short, and each line fixes something that looked wrong:

- **Gravity** swaps a cell with the one below when it's denser and neither is
  solid. Gases rise because empty space (density 2) is denser than smoke (0) and
  fire (1).
- **Diagonal slides** go left or right first at random. Without that, piles
  lean.
- **A cell that moved vertically in a pass may not also move sideways.** Without
  this, water "trades" diffusively and piles up in 45° cones instead of finding
  a level.
- **Liquids carry a direction bit** and keep flowing that way until blocked,
  then turn around. That's what makes a pool level out and drain off a shelf in
  a stream instead of a random walk.
- **Fire on its own rises and dies in a few passes**, so it never had time to
  light wood. `EMBER` is burning wood: it stays put, throws fire into empty
  neighbors, burns out to smoke, and turns back to wood when water touches it.

The GPU spells these rules out on tiles with `select`, in the same order as
`cpu.rs`. Here's gravity and the first diagonal, with the masks remembering who
moved:

```rust
// Gravity. The masks remember who moved vertically this pass.
let m_ac: Mask = sinks(a, c);
let (a, c) = swap(m_ac, a, c);
let m_bd: Mask = sinks(b, d);
let (b, d) = swap(m_bd, b, d);
// Diagonals, in a random order so piles don't lean.
let left_first: Mask = not(bit(r2, 0i32));
let m1: Mask = and(left_first, slides_down(a, d));
let (a, d) = swap(m1, a, d);
```

## Passing per-launch integers without JIT variants

cuTile specializes a kernel on the power-of-two divisibility of every integer
scalar argument (see
[Compilation](./compilation.md#what-makes-a-new-specialization)). Passing the
pass index as a scalar compiled three variants of the 2.4 s `step` kernel: for
pass 0, for odd passes, and for passes divisible by 2 but not 4.

The pass index now lives in a small tensor, one number every four elements (16
bytes) so every pass's view has the same alignment and shape, and each launch
slices out its own `[4]` view:

```rust
/// Pass numbers, one every 4 elements (16 bytes) so every pass's view
/// has the same pointer alignment and shape, hence the same kernel.
pass_table: Tensor<i32>,
```

```rust
let range = 4 * pass..4 * pass + 4;
let pass_view = self.pass_table.slice(std::slice::from_ref(&range))?;
```

One variant. And the slice is also what a CUDA graph needs: a graph replays the
launch with the same view, so the pass number comes along for free.

## Compile time is run time

The first `step` kernel took 2.4 s to compile and 0.5 ms per frame. Both numbers
scale with the count of tile ops, and the kernel had a lot of them. Device
functions are inlined at every call site, and every `kind == X || kind == Y`
predicate was a chain of compares and selects, repeated for every neighbor.

The fix was to write the rules smaller. Per-element properties became bit-mask
tables indexed by kind, so a test is one shift and one mask:

```rust
/// Per-element tables as bit masks indexed by kind, so a test is a
/// shift and a mask instead of a chain of compares.
fn has(k: I, table: i32) -> Mask {
    eq_tile(andi(shri(fill(table), k), fill(1i32)), fill(1i32))
}

fn solid(k: I) -> Mask {
    has(k, 0x130i32) // wood, wall, ember
}

fn falls(k: I) -> Mask {
    has(k, 0x0ei32) // sand, water, oil
}
```

The density table packs into one constant, four bits per kind:

```rust
/// Densities of kinds 0-7, 4 bits each: empty 2, sand 5, water 4,
/// oil 3, wood 9, wall 9, fire 1, smoke 0. Ember (8) is 9.
fn density(k: I) -> I {
    let d: I = andi(shri(fill(0x01993452i32), k * fill(4i32)), fill(15i32));
    select(is(k, 8i32), fill(9i32), d)
}
```

With that, and `kind()` of each cell hoisted out of the rules, compile dropped
to 1.7 s and the frame to 0.21 ms. Same rules, 2.4× faster.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2. Per frame, 4 passes:

| World     | CPU rayon | GPU eager | GPU graph | Speedup |
| --------- | --------- | --------- | --------- | ------- |
| 640×352   | 6.7 ms    | 0.356 ms  | 0.231 ms  | 29×     |
| 1920×1088 | 12.1 ms   | 1.28 ms   | 1.10 ms   | 11×     |

The graph is worth a third at the small size: a frame is one paint kernel, four
step kernels and one render kernel, so launch overhead is a large share of 0.23
ms. Like `light2d`, it captures two graphs, one per buffer parity, because the
paint step and the even number of passes together leave each frame's result in
the other world buffer (see [CUDA graphs](./cuda-graphs.md)).

Transfers per frame go through `tilekit::Pinned`: paint upload 0.08 ms and frame
download 0.13 ms at 640×352, 0.46 ms and 1.0 ms at 1080p. The window runs at
about 600 fps at 640×352; the CPU-side painting and blit cost more than the
simulation.

The first frame costs 2.2 s even with the JIT cache warm. The `step` kernel is
big (it spells out every block rule on nine views of the source), and its first
compile stage alone takes 1.7 s.

## Gotchas at `d92c160`

- **`muli(.., overflow::None)` fails to serialize** ("missing attribute
  'overflow' on op MulI"), and so does `shli`. The `*` operator wraps, which is
  what a hash wants. Left shifts became multiplications by a power of two.
- **Signed constants**: write `0x846c_a68bu32 as i32`, don't type the two's
  complement.
- **Every kernel name becomes a type.** A kernel called `hash_map` generated a
  `HashMap` type that clashed with the `std` one the macro's own expansion uses.
  Renamed to `hash_grid`.
- **Block indices can't go negative.** Views shifted by −1 assert when the block
  index is out of range, so the "up" and "left" views clamp with
  `max(i - 1, 0)`, and the ghost ring makes that harmless: the ring is `WALL`,
  is never updated, so what those loads return never matters.
- **`Stream::synchronize` is unsafe** and needs the device context bound to the
  calling thread first: `stream.device().bind_to_thread()?`.

## Ideas to try next

- More elements: acid, steam (water near fire rises, condenses at the ceiling),
  lava, plants that grow toward water, gunpowder.
- Temperature as a second field instead of "near fire" checks, so heat spreads
  through metal and water boils.
- Rigid bodies on top of the grid (Noita's trick: rasterize a body into the
  cells, simulate, read it back).
- Pressure for liquids so water rises in a U-bend.
- A persistent world larger than the window, streamed in tiles.
