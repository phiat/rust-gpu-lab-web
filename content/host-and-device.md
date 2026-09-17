---
title: Host and device
summary: Modules, entry points, launchers, and lazy DeviceOps.
order: 4
section: Concepts
---

A cuTile program has two halves. **Host code** runs on the CPU. It allocates
tensors, picks streams, launches kernels and reads results back. **Device code**
runs on the GPU and describes the work of a single tile program.

```rust
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
    ) {
        let tile_x = x.load_like(z);
        z.store(tile_x * alpha);
    }
}

fn main() -> Result<(), cuda_async::error::DeviceError> {
    let device = cuda_core::Device::new(0)?;   // connect to GPU 0
    let stream = device.new_stream()?;         // a work queue

    let x = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
    let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;

    let _ = kernels::scale((&mut z).partition([4, 4]), &x, 2.0f32)
        .sync_on(&stream)?;
    Ok(())
}
```

## Modules and entry points

- `#[cutile::module]` marks a module whose functions can compile for the GPU.
- `#[cutile::entry()]` marks a **kernel entry point**, which the host can
  launch.
- Unmarked functions in the module are **device functions**. They're inlined
  into entry points but can't be launched on their own.

Entry point rules:

1. They live inside a `#[cutile::module]`.
2. Writable tensor params use **static** tile shapes (`Tensor<f32, S>`).
3. Read-only tensor params may use **dynamic** dims (`{ [-1, -1] }`).
4. Kernels **write into tensor params**. They don't return values.

## Launchers

The macro generates a host-side launcher with a matching signature:

| Kernel parameter    | Host passes                                           |
| ------------------- | ----------------------------------------------------- |
| `&Tensor<T, S>`     | `&Tensor<T>`, `Arc<Tensor<T>>`, or `Tensor<T>`        |
| `&mut Tensor<T, S>` | `Partition<&mut Tensor<T>>` or `Partition<Tensor<T>>` |
| scalar (`f32`, …)   | the same scalar                                       |

A launch **returns all runtime arguments in parameter order**, so you get your
tensors back: `let (z, _x, _y) = add(z, x, y).sync()?;`.

## DeviceOps are lazy

Tensor constructors, launchers and readbacks all return a `DeviceOp`, a
_description_ of GPU work. Nothing happens until you synchronize, await, or
capture it into a CUDA graph:

```rust
let z = api::zeros::<f32>(&[32, 32]); // nothing allocated yet
let z = z.sync_on(&stream)?;          // runs here
```

Reading a result back to the CPU is also a DeviceOp:

```rust
let host: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream)?;
```

Independent ops can overlap on different streams. Ops that depend on each other
must be chained or placed on the same stream.

Next: [Compilation](./compilation.md).
