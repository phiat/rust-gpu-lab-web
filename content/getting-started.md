---
title: Getting started
summary: What tileworld is, how the workspace is set up, and how to run things.
order: 1
section: Start here
---

**tileworld** is a Rust workspace for learning
[cuTile Rust](https://github.com/NVlabs/cutile-rs) (`cutile-rs`), NVIDIA Labs'
tile-based way of writing GPU kernels in ordinary Rust. It's on GitHub as
[rust-gpu-lab](https://github.com/phiat/rust-gpu-lab). This site is its
notebook: concept notes, interactive demos, and a live view of the workspace.

## The workspace

| Piece                 | Value                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Crates                | six demos (`mandelbrot`, `life`, `filters`, `raymarch`, `light2d`, `sand`) and `tilekit`, their shared host helpers |
| Rust                  | edition 2021, `rust-version = "1.89"` (stable, no nightly)                                                          |
| `cutile`, `cuda-core` | git, pinned to rev `d92c160`                                                                                        |
| Other deps            | `clap`, `rayon`, `image` (PNG and JPEG), `minifb` (X11 only, for the windows)                                       |
| CUDA                  | `.cargo/config.toml` sets `CUDA_TOOLKIT_PATH` to CUDA 13.3                                                          |
| History               | one commit per crate, then the README, a license, `tilekit` and `sand`                                              |

Each demo has a CPU reference in `cpu.rs`, the cuTile version in `gpu.rs`, and a
`bench` command that checks the two agree before timing them. From `life` on,
`bench` also times the GPU replaying a CUDA graph. The crate READMEs call the
workspace by its working name, tileworld, and so does this site.

The pin exists for a reason spelled out in `Cargo.toml`: crates.io only has
`0.3.1`, while `main` (`0.4.0`) is what the book documents, and the project is
pre-1.0 and breaks its API often. **Read the book at the pinned rev**, not the
latest one (see the [reading list](./reading-list.md)).

The [Project page](/project) reads these files from disk on every request, so it
stays current as the workspace grows.

## Requirements

From the workspace README:

- **An NVIDIA GPU** with compute capability 8.0 or newer (Ampere, Ada, Hopper,
  Blackwell).
- **CUDA Toolkit 13.2 or newer** (13.3 recommended), which provides the
  `tileiras` compiler cuTile uses.
- **Rust 1.89 or newer**, stable.
- **Linux.** Tested on WSL2 with Ubuntu 24.04.
- **An X11 display** for the windowed demos (`life`, `raymarch`, `light2d`,
  `sand`). On WSL2, WSLg provides one.

`.cargo/config.toml` points `CUDA_TOOLKIT_PATH` at `/usr/local/cuda-13.3`. If
your toolkit is elsewhere, edit that file or export the variable yourself: Cargo
doesn't override a variable that's already set.

The toolkit minimums per architecture, from the upstream README at `d92c160`:

| GPU compute capability | Minimum CUDA Toolkit |
| ---------------------- | -------------------- |
| `sm_8x` (Ampere / Ada) | 13.2                 |
| `sm_90` (Hopper)       | 13.3                 |
| `sm_100+` (Blackwell)  | 13.2                 |

CUDA 13.3 is recommended; GPUs below `sm_80` are unsupported. The Project page
probes `nvidia-smi` so you can check your compute capability against this table.

> Upstream tests on Ubuntu 24.04. tileworld is developed under WSL2, where the
> NVIDIA driver lives on the Windows side, a setup upstream doesn't claim to
> test.

## Running things

In the tileworld workspace:

```bash
cargo run --release -p mandelbrot -- render            # GPU → mandelbrot.png
cargo run --release -p mandelbrot -- bench --skip-serial
cargo run --release -p life -- run                     # 60 fps window (X11)
cargo run --release -p life -- bench
cargo run --release -p filters -- run                  # → filters-out/*.png
cargo run --release -p filters -- bench
cargo run --release -p raymarch -- run                 # window; first start compiles for ~30 s
cargo run --release -p raymarch -- bench
cargo run --release -p light2d -- run                  # paint lights and walls in a window
cargo run --release -p light2d -- check                # jump flood vs exact distance transform
cargo run --release -p sand -- run                     # paint sand, water, oil and fire in a window
cargo run --release -p sand -- check --frames 60       # GPU vs CPU, cell for cell
cargo run --release -p <crate> -- --help               # every command and option
```

The first run of each demo JIT compiles its kernels. That takes under a second
for `mandelbrot`, about 2 s for `sand` and about 30 s for `raymarch`. Every demo
saves compiled kernels to `~/.cache/cutile/kernels`, so later runs start in
0.3–4 s (see [Compilation](./compilation.md)).

More flags are in each project note: [mandelbrot](./mandelbrot.md),
[life](./life.md), [filters](./filters.md), [raymarch](./raymarch.md),
[light2d](./light2d.md), [sand](./sand.md) and [tilekit](./tilekit.md).

## Where each idea is taught

The workspace README maps topics to crates. Here is the same map, with the notes
that explain each topic:

| Topic                                                                       | Crates                          | Notes                                                                         |
| --------------------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------- |
| Thinking in tile programs instead of pixel threads; masking with `select`   | `mandelbrot`                    | [Partitioning](./partitioning-and-the-grid.md), [mandelbrot](./mandelbrot.md) |
| Per-tile early exit, and why the loop shape decides whether it pays off     | `mandelbrot`, `raymarch`        | [mandelbrot](./mandelbrot.md), [raymarch](./raymarch.md)                      |
| Stencils that read neighbors: view shift plus block offset, ghost tiles     | `life`, `light2d`               | [Stencils](./stencils.md), [life](./life.md)                                  |
| "Valid" convolutions that need no block arithmetic at all                   | `filters`                       | [Stencils](./stencils.md), [filters](./filters.md)                            |
| CUDA graphs: ping-pong buffers, no allocation, when they pay off            | `life`, `filters`, `light2d`    | [CUDA graphs](./cuda-graphs.md)                                               |
| One `Submit` trait so the same pipeline runs eagerly or records a graph     | `filters`, `light2d`, `tilekit` | [CUDA graphs](./cuda-graphs.md), [tilekit](./tilekit.md)                      |
| A whole shading pipeline in one kernel, with parameters in a device buffer  | `raymarch`                      | [raymarch](./raymarch.md)                                                     |
| Data-dependent reads through `unsafe` pointer gathers                       | `light2d`                       | [light2d](./light2d.md)                                                       |
| JIT costs: compile time, the disk cache, specialization on divisibility     | `raymarch`, `light2d`           | [Compilation](./compilation.md)                                               |
| Pinned host buffers: transfers without per-frame allocation                 | `filters`, `tilekit`            | [tilekit](./tilekit.md), [Host and device](./host-and-device.md)              |
| The generic tile shape type bug, and writing literal shapes to avoid it     | `filters`, `raymarch`           | [filters](./filters.md), [raymarch](./raymarch.md)                            |
| Matching float kernels to the CPU (within 1/255, then exactly)              | `raymarch`, `light2d`           | [raymarch](./raymarch.md), [light2d](./light2d.md)                            |
| Moving cells without conflicts: Margolus blocks, hashed randomness          | `sand`                          | [sand](./sand.md)                                                             |
| Per-launch integers through a tensor view, so one kernel variant, not three | `sand`                          | [sand](./sand.md), [Compilation](./compilation.md)                            |
| Fewer tile ops means faster compile _and_ faster frames                     | `sand`                          | [sand](./sand.md), [Compilation](./compilation.md)                            |

To check that the toolchain works end to end, run the upstream hello world from
a clone of `cutile-rs` checked out at the same rev:

```bash
git clone https://github.com/NVlabs/cutile-rs && cd cutile-rs
git checkout d92c160
cargo run -p cutile-examples --example hello_world
```

## What's next

The workspace README keeps a roadmap. Everything on it is grid-shaped, so the
stencil, gather and CUDA graph lessons carry over. In order:

1. ~~**Falling sand**~~: done, see [sand](./sand.md).
2. **Cloth and soft bodies**: a grid of particles joined by constraints, with
   wind and a collider. An iterative solver with red/black passes, and a grid
   that holds objects instead of pixels.
3. **Smoke and fluid** (stable fluids): advection as an interpolated gather, a
   20–40 pass pressure solve per frame, and `light2d`-style painted obstacles.
4. **Flow-field pathfinding with crowds**: a distance-to-goal field that
   respects walls, its gradient as a direction field, and thousands of agents
   that sample it.

Later: terrain with hydraulic erosion, a voxel world ray tracer, a
post-processing stack for `raymarch` (bloom, tone mapping, depth of field,
FXAA), and boids through a density grid, since the tile model has no scatter or
sort. Also on the list: Gray-Scott reaction-diffusion, Lattice Boltzmann, an
async job server that queues GPU work, an MNIST MLP, embedding search, and a
kernel shootout against CUDA C++ and cuda-oxide.

## This site

```bash
deno task dev                          # dev server with hot reload
TILEWORLD_DIR=/path/to/tileworld deno task dev   # point at another checkout
```

Notes are plain markdown in `content/`. Add a file with a `title`, `summary` and
`order` in its front matter, then refresh. Link between notes with relative
links such as `[the grid](./partitioning-and-the-grid.md)`.
