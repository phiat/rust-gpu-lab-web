---
title: Partitioning and the grid
summary: How a partitioned output tensor decides how many tile programs launch and what each one writes.
order: 3
section: Concepts
---

A kernel runs as **many tile programs at once**, one per cell of a launch grid.
Partitioning is how each program gets its own slice of the data.

> Try it: the [Playground](/playground) has a grid explorer and a Mandelbrot
> renderer that draws tile by tile.

## Host side: partition the output

Mutable outputs **must** be partitioned on the host before launch:

```rust
let x = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
let z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?.partition([4, 4]);
let (z, _x, _y) = add(z, &x, &y).sync_on(&stream)?;
```

This is Rust's ownership rule carried over to the GPU. Every tile program writes
a **disjoint** region, so no two of them can race on the same memory. The
partition also sets the launch grid: 32÷4 = 8 per axis, so **8×8 = 64 tile
programs**.

## Device side: partition inputs however you like

Read-only inputs aren't partitioned on the host. Many programs may read the same
(or overlapping) memory safely. Inside the kernel you can partition them in
whatever shape the algorithm needs, even several ways at once:

```rust
let pid_m = program_id(0);
let part_x = x.partition(shape![BM, BK]);
let tile_x = part_x.load([pid_m, k_tile]);
```

Matrix multiply does exactly this: the left and right inputs use different tile
shapes.

## Grid geometry

```text
Tensor shape:    [128, 256]
Partition shape: [ 32,  64]
Grid:            (ceil(128/32), ceil(256/64), 1) = (4, 4, 1)
```

- Tensor **dim 0 → grid x**, dim 1 → grid y, dim 2 → grid z.
- Lower-rank tensors get trailing grid dimensions of 1: `[1024]` split into
  `[128]` gives `(8, 1, 1)`.
- If a kernel has several mutable outputs, their inferred grids must match.
- You can also set the grid explicitly with `.grid((16, 16, 1))`.

For an image stored row-major as `[height, width]`, that means `program_id(0)`
is the **tile row** and `program_id(1)` is the **tile column**.

## Partial tiles at the edges

When a dimension isn't a multiple of the tile size, the grid rounds up and the
last tiles hang off the end. A 720-row image with 32-row tiles gets 23 tile
rows, and the last one covers only 16 real rows. Kernel code still sees full
32×32 tiles, and cuTile handles the overhang:

- **Stores are masked.** Elements past the end are never written.
- **Loads zero-pad.** Elements past the end read as 0.
- **Out-of-range block indices assert.** Loading block `[i, j]` from a
  device-side partition where the block doesn't exist is a runtime error, not a
  zero tile.

`light2d` builds its edge handling on these rules: it encodes "no seed" as 0 so
padding reads correctly, and clamps block indices before loading (see
[Project: light2d](./light2d.md#edges-from-padding-rules-not-branches)). The
overhanging pixels still cost work: `raymarch` marches all 32×32 rays in every
edge tile.

## Program ids

Every program runs the same code with different coordinates:

```rust
let pid0 = program_id(0);   // 0 <= pid0 < num_programs(0)
let n0   = num_programs(0);
```

`get_tile_block_id()` and `get_num_tile_blocks()` return all three axes at once.
_Tile program_, _tile block_ and _tile thread_ all mean the same thing.

## Order is unspecified

All tile programs are **concurrent**: some run in parallel, and their relative
order isn't defined. A kernel must never depend on block (0,0) finishing before
block (0,1). The Playground's Mandelbrot demo shuffles the launch order to make
this visible.

## Exercises (from tutorial 02)

Try each one before opening the answer.

1. `zeros(&[32, 32]).partition([8, 8])`: how many programs, and how many
   elements each?

   <details><summary>Show answer</summary>

   32 ÷ 8 = 4 per axis, so a `(4, 4, 1)` launch grid: 16 tile programs, each
   writing 8×8 = 64 elements.

   </details>

2. Change the kernel to add three tensors: `z = x + y + w`.

   <details><summary>Show hint</summary>

   Add a third read-only parameter and load its tile the same way as the others.
   Nothing new is partitioned on the host, and the launch grid still comes from
   `z`.

   ```rust
   let tile_w = w.load_like(z);
   z.store(tile_x + tile_y + tile_w);
   ```

   </details>

3. Try `partition([4, 8])`, a rectangular tile. Does it still work?

   <details><summary>Show answer</summary>

   Yes. 32 ÷ 4 = 8 along dim 0 and 32 ÷ 8 = 4 along dim 1, so the launch grid is
   `(8, 4, 1)`: 32 tile programs, each writing 4×8 = 32 elements. To check, load
   Tutorial 02 in the [grid explorer](/playground) and change the partition to 4
   and 8.

   </details>
