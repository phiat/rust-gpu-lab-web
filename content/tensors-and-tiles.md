---
title: Tensors and tiles
summary: The two kinds of data in a kernel, and the load → compute → store idiom.
order: 2
section: Concepts
---

A cuTile kernel moves data **from tensors into tiles**, computes on the tiles,
and **stores the result back** into a tensor. Nearly every kernel has this
shape.

## Two kinds of data

| Property   | Tensor                    | Tile                                       |
| ---------- | ------------------------- | ------------------------------------------ |
| Lives in   | GPU global memory         | registers                                  |
| Mutability | mutable or read-only      | immutable                                  |
| Shape      | static, dynamic, or mixed | static                                     |
| You can…   | load and store            | do arithmetic, reductions, matmul, reshape |
| Lifetime   | persists across kernels   | exists only inside a kernel                |

A tensor is where data _is_; a tile is where data is _worked on_. Tensors don't
support arithmetic. You have to load them into tiles first.

## Load → compute → store

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,            // output: static tile shape S
    x: &Tensor<f32, { [-1, -1] }>,     // input: dynamic full shape
    y: &Tensor<f32, { [-1, -1] }>,
) {
    let tile_x = x.load_like(z);   // 1. LOAD   tensor → tile
    let tile_y = y.load_like(z);
    z.store(tile_x + tile_y);      // 2. COMPUTE, 3. STORE  tile → tensor
}
```

`x.load_like(z)` loads the region of `x` that lines up with the piece of `z`
this tile program owns. Use it for element-wise work. The input must have the
shape of the _whole_ output tensor, not of the per-program slab.

Because tiles are immutable, every operation makes a new tile:

```rust
let tile = x.load_like(z);
let shifted = tile + 1.0f32;
let scaled = shifted * 2.0f32;
z.store(scaled);
```

If you need to read the output's current value before writing it, use
`load_tile_mut(z)`.

## Static vs dynamic shapes

- **Static** dimensions are compile-time constants. They let the compiler check
  shapes and optimize layout.
- **Dynamic** dimensions are written `-1` and resolved from the runtime tensor.

The usual pattern is a **static output tile shape** with **dynamic input
shapes**. Changing a static (const generic) value can produce a new compiled
variant; changing a dynamic dimension doesn't. See
[Compilation](./compilation.md).

## Shape errors are type errors

```rust
let a: Tile<f32, { [16, 8] }> = ...;
let b: Tile<f32, { [16, 32] }> = ...;
let c = mma(a, b, acc); // compile error: inner dimensions don't match
```

Mixed element types need an explicit `convert_tile`.

## Operation families

| Category              | Examples                                                 |
| --------------------- | -------------------------------------------------------- |
| Load / store          | `load_like`, `load_tile_mut`, `Partition::load`, `store` |
| Arithmetic            | `+ - * /`, `fma`, `true_div`                             |
| Math                  | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`      |
| Reduction / scan      | `reduce_sum`, `reduce_max`, `scan`                       |
| Matrix multiply       | `mma`, `mmaf_scaled`                                     |
| Shape                 | `reshape`, `broadcast`, `transpose`, `shape!`            |
| Comparison            | `lt_tile`, `gt_tile`, `eq_tile`, `select`                |
| Creation / conversion | `constant`, `iota`, `broadcast_scalar`, `convert_tile`   |

Next: [Partitioning and the grid](./partitioning-and-the-grid.md).
