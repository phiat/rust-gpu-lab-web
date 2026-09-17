---
title: Getting started
summary: What tileworld is, how the workspace is set up, and how to run things.
order: 1
section: Start here
---

**tileworld** is a Rust workspace for learning
[cuTile Rust](https://github.com/NVlabs/cutile-rs) (`cutile-rs`), NVIDIA Labs'
tile-based way of writing GPU kernels in ordinary Rust. This site is its
notebook: concept notes, interactive demos, and a live view of the workspace.

## The workspace

| Piece                 | Value                                                                         |
| --------------------- | ----------------------------------------------------------------------------- |
| Crates                | `mandelbrot`, `life`, `filters`, `raymarch`                                   |
| Rust                  | edition 2021, `rust-version = "1.89"` (stable, no nightly)                    |
| `cutile`, `cuda-core` | git, pinned to rev `d92c160`                                                  |
| Other deps            | `clap`, `rayon`, `image` (PNG and JPEG), `minifb` (X11 only, for the windows) |
| CUDA                  | `.cargo/config.toml` sets `CUDA_TOOLKIT_PATH` to CUDA 13.3                    |
| History               | a git repo, one commit per crate                                              |

Each crate has a CPU reference in `cpu.rs`, the cuTile version in `gpu.rs`, and
a `bench` command that checks the two agree before timing them.

The pin exists for a reason spelled out in `Cargo.toml`: crates.io only has
`0.3.1`, while `main` (`0.4.0`) is what the book documents, and the project is
pre-1.0 and breaks its API often. **Read the book at the pinned rev**, not the
latest one (see the [reading list](./reading-list.md)).

The [Project page](/project) reads these files from disk on every request, so it
stays current as the workspace grows.

## Requirements

From the upstream README at `d92c160`:

| GPU compute capability | Minimum CUDA Toolkit |
| ---------------------- | -------------------- |
| `sm_8x` (Ampere / Ada) | 13.2                 |
| `sm_90` (Hopper)       | 13.3                 |
| `sm_100+` (Blackwell)  | 13.2                 |

CUDA 13.3 is recommended; GPUs below `sm_80` are unsupported. The Project page
probes `nvidia-smi` so you can check your compute capability against this table.

> Upstream tests on Ubuntu 24.04. If you run tileworld under WSL2, the NVIDIA
> driver lives on the Windows side; that setup isn't something upstream claims
> to test.

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
```

More flags are in each project note: [mandelbrot](./mandelbrot.md),
[life](./life.md), [filters](./filters.md) and [raymarch](./raymarch.md).

To check that the toolchain works end to end, run the upstream hello world from
a clone of `cutile-rs` checked out at the same rev:

```bash
git clone https://github.com/NVlabs/cutile-rs && cd cutile-rs
git checkout d92c160
cargo run -p cutile-examples --example hello_world
```

## This site

```bash
deno task dev                          # dev server with hot reload
TILEWORLD_DIR=/path/to/tileworld deno task dev   # point at another checkout
```

Notes are plain markdown in `content/`. Add a file with a `title`, `summary` and
`order` in its front matter, then refresh. Link between notes with relative
links such as `[the grid](./partitioning-and-the-grid.md)`.
