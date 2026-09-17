---
title: Reading list
summary: The cuTile Rust book and examples, in a sensible order, at the pinned rev.
order: 20
section: Reference
---

The book is versioned with the code. Since tileworld pins `cutile-rs` to
`d92c160`, read that copy. Cargo has already fetched it:

```bash
ls ~/.cargo/git/checkouts/cutile-rs-*/d92c160/cutile-book/
```

The published book is at <https://nvlabs.github.io/cutile-rs/>, but it tracks
`main`.

## Tutorials

1. **Hello world**: program ids and the launch grid
2. **Vector addition**: partitioning, `load_like`, load/store
3. **SAXPY**: `a·x + y`, a scalar as a kernel parameter
4. **Matrix multiplication**: device-side partitions, loops, `mma`
5. **Fused softmax**: reductions and broadcasting
6. **Fused multihead attention**
7. **Intro to async execution**: DeviceOps and streams (tokio)
8. **Data-parallel MLP**
9. **Pointer addition**: raw device pointers when abstractions are not enough
10. **CUDA graphs**
11. **Inference with NVFP4/MXFP8**: low-precision formats

1–5 cover what the Mandelbrot project needs. 10 is the background for `life` and
`filters`, which both capture their kernels as a graph.

## Guides

- `guide/host-vs-device.md`: see [Host and device](./host-and-device.md)
- `guide/tensors-and-tiles.md`: see [Tensors and tiles](./tensors-and-tiles.md)
- `guide/useful-mental-models.md`: grid geometry, memory hierarchy, coming from
  CUDA/Triton
- `guide/jit-compilation.md`: see [Compilation](./compilation.md)
- `guide/device-operations.md`: composing DeviceOps, streams, CUDA graphs
- `guide/performance.md`, `guide/autotuning.md`,
  `guide/bounds-check-placement.md`
- `guide/debugging-and-profiling.md`
- `reference/dsl-api.md`: every device-side op with its signature

## Examples worth running

Run these from a `cutile-rs` clone with
`cargo run -p cutile-examples --example <name>`:

| Example          | Why                                   |
| ---------------- | ------------------------------------- |
| `hello_world`    | toolchain smoke test                  |
| `saxpy`          | the smallest real kernel              |
| `softmax`        | reductions and broadcasting           |
| `gemm`           | device-side partitioning and loops    |
| `autotune`       | picking tile sizes                    |
| `jit_disk_cache` | run twice to see cold vs warm start   |
| `cuda_graphs`    | graph capture with `CudaGraph::scope` |
