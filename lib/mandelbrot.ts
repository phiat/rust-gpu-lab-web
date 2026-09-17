/**
 * Browser-side mirror of tileworld/mandelbrot: the same pixel mapping,
 * bailout, smooth count and palette, so the demos look like real renders.
 * Keep in sync with mandelbrot/src/{main,cpu,palette}.rs.
 */

/** BAILOUT_SQ in main.rs. */
export const BAILOUT_SQ = 256;

export interface View {
  cx: number;
  cy: number;
  /** Width of the view in the complex plane (`--span`). */
  span: number;
}

export interface Location extends View {
  name: string;
  iters: number;
}

export const HOME: View = { cx: -0.65, cy: 0, span: 3.2 };

export const LOCATIONS: Location[] = [
  { name: "Whole set", ...HOME, iters: 250 },
  { name: "Seahorse valley", cx: -0.7453, cy: 0.1127, span: 0.012, iters: 500 },
  { name: "Elephant valley", cx: 0.285, cy: 0.012, span: 0.012, iters: 500 },
  {
    name: "Spiral valley",
    cx: -0.08855,
    cy: 0.65405,
    span: 0.0105,
    iters: 1000,
  },
  { name: "Mini Mandelbrot", cx: -1.762, cy: 0, span: 0.055, iters: 500 },
];

export interface Tile {
  pid0: number;
  pid1: number;
  y0: number;
  y1: number;
  x0: number;
  x1: number;
}

export interface TileStats {
  tile: Tile;
  /** Largest escape count among pixels that escaped. */
  maxEscape: number;
  /** Some pixel never escaped, so every kernel runs all max_iter steps. */
  anyInside: boolean;
}

export function makeTiles(h: number, w: number, t: number): Tile[] {
  const tiles: Tile[] = [];
  for (let pid0 = 0; pid0 < Math.ceil(h / t); pid0++) {
    for (let pid1 = 0; pid1 < Math.ceil(w / t); pid1++) {
      tiles.push({
        pid0,
        pid1,
        y0: pid0 * t,
        y1: Math.min((pid0 + 1) * t, h),
        x0: pid1 * t,
        x1: Math.min((pid1 + 1) * t, w),
      });
    }
  }
  return tiles;
}

export function shuffle<T>(items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

/** `View::pixel_mapping`: square pixels sampled at their centers. */
export function pixelMapping(view: View, w: number, h: number) {
  const step = view.span / w;
  const x0 = view.cx - view.span / 2 + step / 2;
  const y0 = view.cy + step * h / 2 - step / 2;
  return { x0, y0, step };
}

export function toComplex(
  view: View,
  w: number,
  h: number,
  col: number,
  row: number,
) {
  const { x0, y0, step } = pixelMapping(view, w, h);
  return { re: x0 + col * step, im: y0 - row * step };
}

/** palette.rs `gradient`: a + b·cos(2π(c·t + d)) with d = (0, 0.1, 0.2). */
function gradient(t: number, data: Uint8ClampedArray, o: number) {
  data[o] = (0.5 + 0.5 * Math.cos(2 * Math.PI * t)) * 255;
  data[o + 1] = (0.5 + 0.5 * Math.cos(2 * Math.PI * (t + 0.1))) * 255;
  data[o + 2] = (0.5 + 0.5 * Math.cos(2 * Math.PI * (t + 0.2))) * 255;
}

/**
 * One tile program's work, written like cpu.rs: per pixel with an early
 * break. Also records what the GPU kernels would need to know about the tile.
 */
export function renderTile(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  tile: Tile,
  view: View,
  maxIter: number,
): TileStats {
  const { x0, y0, step } = pixelMapping(view, w, h);
  let maxEscape = 0;
  let anyInside = false;

  for (let row = tile.y0; row < tile.y1; row++) {
    const ci = y0 - row * step;
    for (let col = tile.x0; col < tile.x1; col++) {
      const cr = x0 + col * step;
      let zr = 0, zi = 0, n = 0;
      while (n < maxIter) {
        const zr2 = zr * zr;
        const zi2 = zi * zi;
        if (zr2 + zi2 > BAILOUT_SQ) break;
        zi = (zr + zr) * zi + ci;
        zr = zr2 - zi2 + cr;
        n++;
      }

      const o = (row * w + col) * 4;
      const mag2 = zr * zr + zi * zi;
      if (mag2 > BAILOUT_SQ) {
        const nu = n + 1 - Math.log2(0.5 * Math.log(mag2));
        gradient(Math.sqrt(Math.max(nu, 0)) * 0.12, data, o);
        if (n > maxEscape) maxEscape = n;
      } else {
        data[o] = data[o + 1] = data[o + 2] = 0;
        anyInside = true;
      }
      data[o + 3] = 255;
    }
  }
  return { tile, maxEscape, anyInside };
}

/**
 * Steps `mandelbrot_early_exit` runs on a tile. After `done` steps a pixel
 * that escapes at count n is still alive iff done < n, so the tile stops at
 * the first check boundary at or past its largest escape count.
 */
export function earlyExitSteps(
  s: TileStats,
  maxIter: number,
  checkEvery: number,
) {
  if (s.anyInside) return maxIter;
  return Math.min(maxIter, Math.ceil(s.maxEscape / checkEvery) * checkEvery);
}

/** Paint a not-yet-launched tile as a navy checker square. */
export function fillTile(data: Uint8ClampedArray, w: number, tile: Tile) {
  const odd = (tile.pid0 + tile.pid1) % 2 === 1;
  const [r, g, b] = odd ? [10, 22, 54] : [16, 31, 70];
  for (let y = tile.y0; y < tile.y1; y++) {
    for (let x = tile.x0; x < tile.x1; x++) {
      const o = (y * w + x) * 4;
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      data[o + 3] = 255;
    }
  }
}
