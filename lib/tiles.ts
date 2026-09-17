/** Tile bookkeeping and canvas helpers shared by the playground demos. */

/** One tile program's piece of an image: its program id and pixel bounds. */
export interface Tile {
  pid0: number;
  pid1: number;
  y0: number;
  y1: number;
  x0: number;
  x1: number;
}

/** Every tile of an `h × w` image split into `t × t` tiles, row by row. */
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

/** Shuffle in place: tile programs finish in no particular order. */
export function shuffle<T>(items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

/** How far a launch has got. */
export interface Progress {
  done: number;
  total: number;
  ms: number;
  running: boolean;
}

/** The image pixel under a pointer, whatever size the canvas is drawn at. */
export function canvasPixel(
  canvas: HTMLCanvasElement,
  e: MouseEvent,
  w: number,
  h: number,
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * w);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * h);
  return {
    x: Math.min(Math.max(x, 0), w - 1),
    y: Math.min(Math.max(y, 0), h - 1),
  };
}

/** Stroke the interior tile boundaries of a `w × h` canvas. */
export function strokeTileGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  tile: number,
) {
  ctx.beginPath();
  for (let x = tile; x < w; x += tile) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
  }
  for (let y = tile; y < h; y += tile) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
  }
  ctx.stroke();
}
