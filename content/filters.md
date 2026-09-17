---
title: "Project: filters"
summary: Grayscale, Gaussian blur and Sobel edges as a chain of integer stencil kernels, checked bit for bit and replayed as one CUDA graph.
order: 12
section: Projects
---

An image filter chain: grayscale, then N passes of a separable 5×5 Gaussian
blur, then Sobel edge detection. Every stage uses integer math, so the GPU
result is compared bit for bit with a rayon version. The chain is written once
and runs either eagerly or as a single CUDA graph.

It builds on [Stencils](./stencils.md) (valid convolution, the approach `life`
couldn't use) and [CUDA graphs](./cuda-graphs.md).

## Running it

```bash
cargo run --release -p filters -- run                      # 4K synthetic image → filters-out/*.png
cargo run --release -p filters -- run --input photo.jpg    # any PNG or JPEG
cargo run --release -p filters -- run --blur-passes 0 --out-dir filters-out-noblur
cargo run --release -p filters -- bench                    # 30 frames: CPU vs GPU eager vs GPU graph
```

`run` writes `gray.png`, `blurred.png` and `edges.png`, plus `input.png` when
the source is synthetic. The synthetic image has per-pixel noise on purpose.
With `--blur-passes 0`, Sobel lights up the noise everywhere. Two passes remove
it and leave the real edges. If you've run both commands, the
[Project page](/project#renders) shows the two output folders side by side.

## Layout

| File      | Role                                                                      |
| --------- | ------------------------------------------------------------------------- |
| `main.rs` | clap CLI (`run`, `bench`), `Dims` and padding, synthetic test image       |
| `gpu.rs`  | four kernels, `Pipeline` with preallocated stage and pinned frame buffers |
| `cpu.rs`  | rayon reference with identical integer math                               |

## Padding

Each stencil stage shrinks its output by twice its radius, so the host pads the
source once, repeating edge pixels:

```rust
/// 2 per 5x5 blur pass, plus 1 for Sobel.
pub fn radius(&self) -> usize {
    2 * self.blur_passes + 1
}
```

[Stencils](./stencils.md#strategy-1-valid-convolution-filters) lists the buffer
shape after every stage.

## Grayscale

BT.601 weights in integers: `(77 R + 150 G + 29 B + 128) >> 8`. The input is
RGBA, not RGB, because every tile dimension must be a power of two. The weights
become a `[4]` tile, broadcast over the pixels, and summed over the channel
axis:

```rust
let px: Tile<u8, { [B, B, 4] }> = part.load([pid.0, pid.1, 0i32]);
let px: Tile<i32, { [B, B, 4] }> = exti(px);
// ... w = [77, 150, 29, 0], reshaped to [1, 1, 4] and broadcast
let weighted: Tile<i32, { [B, B, 4] }> = px * w;
let luma: Tile<i32, { [B, B] }> = reduce_sum(weighted, 2i32);

let half: Tile<i32, { [B, B] }> = fill(128i32, out.shape());
let eight: Tile<i32, { [B, B] }> = fill(8i32, out.shape());
let luma: Tile<i32, { [B, B] }> = shri(luma + half, eight);
let luma: Tile<u8, { [B, B] }> = trunci(luma, overflow::NoWrap);
```

`fill` is a one-line device function. It exists only to give `half` and `eight`
the same tile type as `luma` (see the gotchas below).

## Blur

The binomial `[1 4 6 4 1]` sums to 16, so a horizontal pass followed by a
vertical one sums to 256. The horizontal pass writes unnormalized `u16` values,
at most 16 × 255. The vertical pass adds 128 and shifts right by 8 to get back
to `u8`. That's 5 views per pass instead of 25 for a full 5×5 kernel:

```rust
let four: Tile<i32, { [B, B] }> = constant(4i32, shape![B, B]);
let six: Tile<i32, { [B, B] }> = constant(6i32, shape![B, B]);
let sum: Tile<i32, { [B, B] }> = x0 + x4 + (x1 + x3) * four + x2 * six;
let sum: Tile<u16, { [B, B] }> = trunci(sum, overflow::NoWrap);
out.store(sum);
```

## Sobel

The center pixel has weight 0 in both Sobel kernels, so only 8 views are passed.
Magnitude is the integer L1 approximation `min(|gx| + |gy|, 255)`:

```rust
let gx: Tile<i32, { [B, B] }> = (x_tr + x_mr * two + x_br) - (x_tl + x_ml * two + x_bl);
let gy: Tile<i32, { [B, B] }> = (x_bl + x_bm * two + x_br) - (x_tl + x_tm * two + x_tr);
let mag: Tile<i32, { [B, B] }> = absi(gx) + absi(gy);
let mag: Tile<i32, { [B, B] }> = min_tile(mag, max);
```

## Chaining the stages

`.then(|out| next_op)` hands the previous output to a closure by value, but a
stencil stage needs borrowed views of that output, and views made inside the
closure can't outlive it. So `Pipeline` preallocates every stage buffer and
submits each kernel through the `Submit` trait (now in [tilekit](./tilekit.md),
shared with `light2d`):

```rust
let mut src: &Tensor<u8> = &self.gray;
for (mid, dst) in self.blur.iter_mut() {
    {
        let v = col_views(src)?;
        sub.submit(kernels::blur_h(mid.partition([t, t]), &v[0], &v[1], &v[2], &v[3], &v[4]))?;
    }
    {
        let v = row_views(mid)?;
        sub.submit(kernels::blur_v(dst.partition([t, t]), &v[0], &v[1], &v[2], &v[3], &v[4]))?;
    }
    src = dst;
}
```

Capturing the whole chain is then one line:

```rust
Ok(CudaGraph::scope(&stream, |s| Ok(self.run(s)?))?)
```

## Results

3840×2160, 2 blur passes (6 kernels per frame). RTX 4070 Ti SUPER, i9-14900KF
(28 threads), WSL2, per frame:

| Path      | Tile 16 | Tile 32 | Tile 64 |
| --------- | ------- | ------- | ------- |
| CPU rayon | 11.7 ms | 11.7 ms | 11.7 ms |
| GPU eager | 0.84 ms | 0.76 ms | 0.94 ms |
| GPU graph | 0.66 ms | 0.64 ms | 0.77 ms |

Filtering is the cheap part: moving the frame costs several times more. Per
frame, with the graph:

| Transfers                         | Upload | Download | End to end    |
| --------------------------------- | ------ | -------- | ------------- |
| pageable `Vec`, new device tensor | 4.6 ms | 0.8 ms   | about 170 fps |
| pinned buffers, reused            | 3.0 ms | 0.4 ms   | about 245 fps |

The first version cloned the 33 MB RGBA frame into a new `Vec`, allocated a
device tensor, copied from pageable memory, then copied device to device into
the buffer the graph reads. `tilekit::Pinned` replaces that with one copy from
page-locked memory into the existing tensor. Of the 3.0 ms that remains, 1.3 ms
is the benchmark copying its frame into the pinned buffer, so a decoder writing
straight into `Pipeline::frame_in()` would skip it. See
[tilekit](./tilekit.md#why-transfers-were-slow).

## Gotchas at `d92c160`

- **Tile dimensions must be powers of two.** A `[B, B, 3]` tile fails in
  `tileiras` with "tile shape dimensions must have power of two length". The
  only visible error was "input does not correspond to Tile IR bytecode", and
  the real message was further up in stderr.
- **Integer widths change with `exti` and `trunci`.** `convert_tile` only
  handles int ↔ float and float ↔ float. `exti` takes signedness from the source
  type, so `u8` zero-extends.
- **`trunci(x, overflow::None)` fails** with "missing attribute 'overflow'" when
  the bytecode is written. `overflow::NoWrap` works, and is correct here because
  every value is already in range.
- **`+`, `-` and `*` compare tile types token for token.** Nested arithmetic,
  `constant(...)` and `exti(...)` results have the generic type `{[B, B]}`.
  Reductions and device-function results have a concrete one, such as
  `{[32, 32]}`. Mixing them fails with
  ``binary `Add` requires operands of the
  same type``. Keep each kernel on one
  side: grayscale is all concrete, the stencils are all generic. `shri`,
  `select` and `eq_tile` don't check. A cleaner fix turned up later in
  [raymarch](./raymarch.md#one-tile-shape-spelled-as-a-literal): drop the
  `const B` generic and write shapes as literals (`{[32, 32]}`), so there's only
  one spelling.
- **Scalar `.broadcast(shape)` only resolves on entry parameters.** On a literal
  or a local `let`, the JIT fails with "unrecognized macro `unreachable`", and
  `Tile::shape()` fails the same way.
- **Debug with `CUTILE_DUMP=ir,bytecode`** and read all of stderr. The useful
  `tileiras` message comes before the generic one.

## Ideas to try next

- Overlap transfers with compute: upload frame N+1 on a second stream while
  frame N is filtered.
- Upload 3 bytes per pixel instead of 4, since the alpha channel is padding.
- Compare an `f32` Sobel using `sqrt(gx² + gy²)` with the integer L1 version.
- Push a live video stream or webcam frames through the captured graph.
