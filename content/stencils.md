---
title: Stencils
summary: Reading neighboring elements when a tile program only sees its own block, and the two ways tileworld does it.
order: 6
section: Concepts
---

A **stencil** computes each output element from a small neighborhood of input
elements. A blur, an edge detector and a Game of Life step are all stencils.

Element-wise kernels such as vector addition read one input element per output
element, so `x.load_like(z)` is all they need. A stencil also reads the elements
around it, and near a tile's edge those neighbors belong to the next block over.

## The trick both crates share: shifted views

Make several host views of the same input, one per neighbor offset. Load each
view at the output's block. Element `(r, c)` of the view shifted by `(dr, dc)`
is input element `(r + dr, c + dc)`, so the kernel receives one aligned tile per
neighbor and combines them with ordinary tile arithmetic. Nothing indexes
individual elements.

From `filters`, where each view has exactly the output's shape:

```rust
/// The eight non-center 3x3 views of `src`, each `(rows - 2) x (cols - 2)`.
fn sobel_views(src: &Tensor<u8>) -> Result<[TensorView<'_, u8>; 8], Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let v = |dr: usize, dc: usize| src.slice(&[dr..dr + rows - 2, dc..dc + cols - 2]);
    Ok([
        v(0, 0)?,
        v(0, 1)?,
        // ...
    ])
}
```

There's a catch: **a view can't start at a negative offset.** Host `slice` views
start at offset 0 or later, and an output partition must cover a whole tensor,
so you can't write into the interior of a padded buffer either. A centered 3×3
stencil needs offset −1. `filters` and `life` get around that in different ways.

## Strategy 1: valid convolution (`filters`)

Only use non-negative offsets. Output `(r, c)` reads input `(r + dr, c + dc)`
with `0 ≤ dr, dc ≤ 2R`, where `R` is the stencil radius. Each output is then
`2R` smaller than its input, and each neighbor view lines up with
`load_like(out)` without any block arithmetic:

```rust
let t_tl: Tile<u8, { [B, B] }> = tl.load_like(out);
let t_tm: Tile<u8, { [B, B] }> = tm.load_like(out);
```

To end at the original size, the host pads the source **once** by the whole
chain's radius, repeating edge pixels. That's 2 per 5×5 blur pass plus 1 for
Sobel. With the defaults (3840×2160, two blur passes, radius 5) the buffers
shrink like this, as `[rows, cols]`:

| Stage                  | Buffer         |
| ---------------------- | -------------- |
| padded RGBA, grayscale | `[2170, 3850]` |
| blur 1, horizontal     | `[2170, 3846]` |
| blur 1, vertical       | `[2166, 3846]` |
| blur 2, horizontal     | `[2166, 3842]` |
| blur 2, vertical       | `[2162, 3842]` |
| Sobel edges            | `[2160, 3840]` |

This fits a pipeline that runs start to finish and handles the image border
once, on the host.

## Strategy 2: split the offset (`life`)

Life can't shrink: every generation feeds the next at the same size, and the
world wraps around. So `life` keeps offset −1 and splits it into a non-negative
view shift plus a block-index shift:

> offset `d = q·B + s` with `0 ≤ s < B` reads as "view shifted by `s`, loaded at
> block `i + q`".

| Offset | View shifted by | Loaded at block |
| ------ | --------------- | --------------- |
| −1     | `B − 1`         | `i − 1`         |
| 0      | 0               | `i`             |
| +1     | 1               | `i`             |

A check with `B = 4`: output block `j` writes buffer columns `4j … 4j+3`. The
view shifted by 3, loaded at block `j − 1`, covers view columns `4j−4 … 4j−1`,
which are buffer columns `4j−1 … 4j+2`. That's the output columns shifted by −1.

Block `i − 1` doesn't exist when `i = 0`, so the buffer carries a **ring of
ghost tiles**: logical cell `(r, c)` lives at `[B + r, B + c]`. Ghost tile
programs run too. Each computes the interior tile it mirrors on the far side of
the torus, so the ring is correct after every step without a copy pass:

```rust
// Ghost tiles alias the interior tile on the opposite edge.
let i: i32 = if pid.0 == 0i32 { tile_rows } else { pid.0 };
let i: i32 = if pid.0 == tile_rows + 1i32 { 1i32 } else { i };
let j: i32 = if pid.1 == 0i32 { tile_cols } else { pid.1 };
let j: i32 = if pid.1 == tile_cols + 1i32 { 1i32 } else { j };
let up: i32 = i - 1i32;
let left: i32 = j - 1i32;
```

Because the block index is computed, `life` partitions its views inside the
kernel and loads by index instead of using `load_like`:

```rust
let p_ul = ul.partition(shape![B, B]);
let t_ul: Tile<u8, { [B, B] }> = p_ul.load([up, left]);
```

## Choosing between them

|                  | Valid convolution         | Offset split and ghost ring           |
| ---------------- | ------------------------- | ------------------------------------- |
| Output size      | shrinks by `2R` per stage | same every step                       |
| Border           | host pads once            | ghost tiles, recomputed each step     |
| Loading          | `view.load_like(out)`     | device-side `partition` + `load([…])` |
| Block arithmetic | none                      | alias ghost indices, subtract 1       |
| In tileworld     | `filters`                 | `life`                                |

The [Playground](/playground) runs the `life` kernel in the browser. Hover a
tile program to see the blocks its views load.

Next: [CUDA graphs](./cuda-graphs.md).
