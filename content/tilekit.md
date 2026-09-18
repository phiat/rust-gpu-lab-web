---
title: "Shared helpers: tilekit"
summary: The host-side code the demos share. Pinned transfer buffers that made filters 45% faster end to end, the eager-or-graph Submit trait, and the JIT disk cache switch.
order: 15
section: Projects
---

`tilekit` isn't a demo. It's a small library crate holding the host-side code
that kept getting copied between demos: about 150 lines, no kernels. It came out
of one measurement. In [filters](./filters.md), the GPU work for a 4K frame took
0.64 ms, and getting the frame to and from the GPU took more than 5 ms.

| Piece                | What it does                                                          | Used by                                           |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| `Pinned<T>`          | page-locked host buffer, allocated once, with `upload` and `download` | `filters`, `raymarch`, `light2d`, `sand`, `cloth` |
| `Submit`, `Eager`    | run a pipeline eagerly or record it into a CUDA graph                 | `filters`, `light2d`, `sand`, `cloth`             |
| `enable_jit_cache()` | turn on cuTile's disk cache at the default location                   | every demo                                        |
| `as_u32()`           | view packed `i32` pixels as the `u32` a window wants, without a copy  | `raymarch`, `light2d`, `sand`                     |

## Why transfers were slow

The demos first uploaded with `api::copy_host_vec_to_device`. Per frame, that
path does four things:

1. clones the frame into a new `Vec` (33 MB for padded 4K RGBA),
2. allocates a new device tensor,
3. copies from **pageable** memory, which the driver first has to stage through
   a pinned buffer of its own,
4. copies device to device into the buffer the captured graph reads, because a
   graph bakes in its buffer addresses (see [CUDA graphs](./cuda-graphs.md)).

Downloads did the mirror image: `dup()` on the device, because `to_host_vec()`
consumes its tensor, then a copy into a new `Vec`.

## Pinned buffers

**Pinned** (page-locked) host memory can't be swapped out or moved by the OS, so
the GPU can read and write it directly (DMA). `Pinned<T>` allocates it once with
`cuMemAllocHost`, and each transfer is then a single copy between it and the
tensor the kernels already use:

```rust
/// Copy this buffer into `dst` and wait for it.
pub fn upload(&self, dst: &mut Tensor<T>, stream: &Arc<Stream>) -> Result<(), Error> {
    assert_eq!(dst.size(), self.len);
    assert!(dst.is_contiguous());
    stream.device().bind_to_thread()?;
    // SAFETY: both sides hold `len` elements; `&mut dst` means no kernel
    // is borrowing it; the context is current; and we synchronize before
    // returning, so the borrow outlives the copy.
    unsafe {
        memcpy_htod_async(
            dst.device_pointer().cu_deviceptr(),
            self.ptr,
            self.len,
            stream,
        )?;
        stream.synchronize()?;
    }
    Ok(())
}
```

cuTile has no safe wrapper for pinned memory yet, so this goes through
`cuda-core` with the tensor's raw device pointer. It's the workspace's second
use of `unsafe`, after
[light2d's gathers](./light2d.md#gathers-the-unsafe-escape-hatch), and the
safety argument is spelled out the same way: sizes are asserted equal, the
`&mut` borrow proves no kernel holds the tensor, and the call synchronizes
before it returns, so nothing outlives the borrow.

Two details make it fit the rest of the workspace:

- **The copy is queued on the kernels' stream.** A download therefore sees their
  finished output without a separate sync.
- **The device buffer never changes.** Uploading into the existing tensor is
  exactly what a captured graph needs, so the device-to-device `memcpy` step
  disappears.

Using it looks like this, from `raymarch`:

```rust
pub fn set_params(&mut self, params: &[f32; PARAMS]) -> Result<(), Error> {
    self.params_host.as_mut_slice().copy_from_slice(params);
    self.params_host.upload(&mut self.params, &self.stream)
}

/// Copy the frame to the host and borrow it as 0x00RRGGBB pixels.
pub fn download(&mut self) -> Result<&[u32], Error> {
    self.frame_host.download(&self.frame, &self.stream)?;
    Ok(tilekit::as_u32(self.frame_host.as_slice()))
}
```

`download` now returns a borrowed slice instead of a new `Vec`, so a frame
allocates nothing. `as_u32` reinterprets the `i32` pixels the kernels store as
the `u32` pixels `minifb` wants, which replaces a per-pixel `map` and `collect`.

## What it bought

RTX 4070 Ti SUPER, per frame:

| Demo                      | Transfer        | Pageable | Pinned  | End to end          |
| ------------------------- | --------------- | -------- | ------- | ------------------- |
| `filters`, 4K frame       | upload          | 4.6 ms   | 3.0 ms  | about 170 → 245 fps |
|                           | download        | 0.8 ms   | 0.4 ms  |                     |
| `raymarch`, 720p window   | download        | 0.39 ms  | 0.22 ms | 310 → 335 fps       |
|                           | params upload   | 0.12 ms  | 0.05 ms |                     |
| `light2d`, 640×352 window | frame download  | 0.22 ms  | 0.12 ms | 355 → 360 fps       |
| `sand`, 640×352 window    | paint upload    | —        | 0.08 ms | about 600 fps       |
|                           | frame download  | —        | 0.13 ms |                     |
| `cloth`, 256×160          | params + screen | —        | 0.07 ms | about 180 fps       |

Of `filters`' 3.0 ms upload, 1.3 ms is the benchmark copying its frame into the
pinned buffer, and the transfer itself runs at about 20 GB/s. A decoder or
camera writing straight into `Pipeline::frame_in()` would skip that copy. `sand`
and `cloth` have no pageable column because they were written after `tilekit`.

The gain tracks how much of a frame is transfer. `filters` moves 33 MB for 0.64
ms of compute and gains 45%. `raymarch` gains 8%, and its remaining millisecond
per frame is the window presenting the image and polling input. `light2d` moves
little and barely changes.

[sand](./sand.md), written after `tilekit`, used `Pinned` from the start, for
its paint layer and frame. `mandelbrot` doesn't use `Pinned` yet. Its 8 MB
download takes 3 ms, four times longer than the render, and is listed as its
next thing to try.

## `Submit`: one pipeline, eager or graph

`filters` and `light2d` each carried an identical copy of this trait, and it now
lives here (`sand` and `cloth` use it too):

```rust
/// Runs a device op either eagerly or as part of a graph capture, so a
/// pipeline is written once for both.
pub trait Submit {
    fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(&self, op: N) -> Result<(), Error>;
}
```

`Eager` syncs each op as it's submitted, and the graph `Scope` records it. A
pipeline written against `&impl Submit` runs both ways, which is how `bench`
compares eager and graph times over the same code. See
[CUDA graphs](./cuda-graphs.md#what-that-meant-in-tileworld).

## The JIT cache switch

```rust
pub fn enable_jit_cache() -> Result<(), Box<dyn std::error::Error>> {
    cutile::jit_cache::enable(Arc::new(
        cutile::jit_cache::FileSystemJitStore::default_location()?,
    ));
    Ok(())
}
```

Every demo's `main` calls it. `raymarch` and `light2d` already had the cache on;
`mandelbrot`, `life` and `filters` gained it, and their startup dropped from
about 0.75–0.9 s to 0.3 s; `sand` and `cloth` were written with it. A hit skips
`tileiras` (0.3–30 s per kernel in this workspace) but not the IR build, which
still costs 40–300 ms per kernel variant on every start.
[Compilation](./compilation.md#disk-cache-opt-in) has the details.

## Clippy, and code that mirrors a kernel

The same commit made `cargo clippy` clean. The CPU references in `raymarch` and
`light2d` keep three lints off on purpose:

```rust
// The math below mirrors the GPU kernel expression for expression (same
// operation order, same `min(max(..))` clamps, same literals), so that the
// two can be compared. Clippy's tidier spellings would hide that.
#![allow(
    clippy::assign_op_pattern,
    clippy::manual_clamp,
    clippy::excessive_precision
)]
```

`t = t + d` instead of `t += d` looks odd in Rust, but it's the only spelling
the kernel has, and a reference that reads line for line like the kernel is
easier to check. This site's [raymarch port](/playground) keeps the same order
for the same reason.

## Ideas it opens up

- **Overlap transfers with compute** (`filters`): upload frame N+1 on a second
  stream while frame N is filtered.
- **Upload 3 bytes per pixel instead of 4** (`filters`): the alpha channel is
  padding.
- **Launch frame N+1 before presenting frame N** (`raymarch`), so the GPU works
  while the window is busy.
- **A pinned download for `mandelbrot`.**
