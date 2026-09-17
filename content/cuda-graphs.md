---
title: CUDA graphs
summary: Record a fixed sequence of kernel launches once and replay it with one driver call, and what that asks of your buffers.
order: 7
section: Concepts
---

Every launch costs CPU time: choosing a stream, packing arguments, calling the
driver. When kernels are fast, that overhead can be most of the work. A **CUDA
graph** records a sequence of GPU work once and replays it with a single driver
call.

Tutorial 10 in the book covers this. Both `life` and `filters` use it.

## Capture and replay

cuTile offers two ways to capture. `.graph_on(stream)` captures a lazy DeviceOp
built from combinators. `CudaGraph::scope` records ops one at a time using
ordinary `&mut` borrows, and it's the one tileworld uses:

```rust
let graph = CudaGraph::scope(&self.stream, |s| {
    for _ in 0..pairs {
        {
            let n = neighborhood(front, &lay)?;
            s.record(step_op(back, &n, lay.tile))?;
        }
        {
            let n = neighborhood(back, &lay)?;
            s.record(step_op(front, &n, lay.tile))?;
        }
    }
    Ok(())
})?;

graph.launch().sync_on(&stream)?; // replay everything
```

`s.record(op)` only accepts graph nodes: kernel launches and `api::memcpy`.
Allocating ops such as `api::zeros` or `dup` are rejected at compile time,
because their addresses could change on replay.

## What a graph bakes in

| Baked in            | So you have to                                       |
| ------------------- | ---------------------------------------------------- |
| buffer addresses    | preallocate every buffer and copy new input into it  |
| tensor shapes       | capture one graph per size                           |
| the sequence of ops | keep host-side control flow out of the replayed part |

New input goes into the fixed input buffer, not a new tensor. From `filters`:

```rust
/// Copy a padded RGBA frame into the input buffer (no reallocation, so a
/// captured graph stays valid).
pub fn upload(&mut self, padded_rgba: Vec<u8>) -> Result<(), Error> {
    let shape = [self.dims.padded_rows(), self.dims.padded_cols(), 4];
    let src = api::copy_host_vec_to_device(&Arc::new(padded_rgba))
        .sync_on(&self.stream)?
        .reshape(&shape)?;
    api::memcpy(&mut self.rgba, &src).sync_on(&self.stream)?;
    Ok(())
}
```

## What that meant in tileworld

- **No swapping buffers.** Eager Life swaps `front` and `back` after every step.
  A recorded step always reads and writes the same buffers, so the graph records
  pairs: front → back → front. Generations per launch are even, and the result
  always ends in `front`.
- **Compile before capturing.** Both crates run the chain eagerly once first, so
  the JIT compile happens outside the capture.
- **Scalars are baked in too.** `raymarch` keeps its camera and animation in a
  32-slot device buffer instead of kernel scalars, and copies new values in
  before each replay. Its step budgets are loop bounds, so they stay scalars,
  and changing quality recaptures the graph.
- **Recording didn't advance the state.** In `life`'s benchmark, the world after
  `launches × gens` replays matched the CPU after exactly that many generations.
- **Reading a buffer the graph keeps writing** uses `.dup().to_host_vec()`,
  because `to_host_vec()` consumes the tensor.
- **One pipeline, both ways.** `filters` writes its chain once against a small
  trait. `Eager` syncs each op, and the graph `Scope` records it:

```rust
pub trait Submit {
    fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(&self, op: N) -> Result<(), Error>;
}
```

## When it pays off

Graphs remove launch overhead, not kernel time. Measured on an RTX 4070 Ti
SUPER, per generation or frame:

| Workload                         | Eager    | Graph    | Speedup |
| -------------------------------- | -------- | -------- | ------- |
| `life` 256², tile 32             | 0.035 ms | 0.007 ms | 5.0×    |
| `life` 1024², tile 32            | 0.046 ms | 0.021 ms | 2.2×    |
| `life` 4096², tile 64            | 0.352 ms | 0.332 ms | 1.06×   |
| `filters` 4K, 6 kernels, tile 32 | 0.76 ms  | 0.64 ms  | 1.2×    |
| `raymarch` 1080p, 1 kernel       | 3.19 ms  | 3.17 ms  | 1.01×   |

The fewer and heavier the kernels, the less there is to save: a `raymarch` frame
is a single launch. At 4096² `life`'s kernel itself dominates. `life`'s README
estimates about 500 GB/s of memory traffic against the card's 672 GB/s
bandwidth, so launch overhead barely matters there.

The book's rule of thumb: use a graph when the same ops repeat many times with
the same shapes. Skip it when shapes or control flow change per iteration, or
when you want to profile kernels one at a time, because a replay shows up as a
single event.

Next: [Project: mandelbrot](./mandelbrot.md).
