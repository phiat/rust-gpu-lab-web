import { useEffect, useRef, useState } from "preact/hooks";

/**
 * A browser mirror of life/src/gpu.rs: the `life_step` stencil kernel run
 * tile program by tile program over a padded buffer with a ghost ring, and
 * checked against the crate's CPU reference every generation.
 */

const ROWS = 48;
const COLS = 64;
const CELL = 10;
const TILE_SIZES = [4, 8, 16];
const DENSITY = 0.3;
const VIEW_NAMES = [["ul", "um", "ur"], ["ml", "mm", "mr"], ["dl", "dm", "dr"]];

type Pattern = "soup" | "gliders";

interface Sim {
  B: number;
  tileRows: number;
  tileCols: number;
  bufRows: number;
  bufCols: number;
  /** Current generation, padded with the ghost ring. */
  front: Uint8Array;
  back: Uint8Array;
  /** Per buffer cell: 255 while alive, fading after it dies (as in main.rs). */
  heat: Uint8Array;
  /** CPU reference world, rows x cols, no padding. */
  cells: Uint8Array;
  generation: number;
  /** First generation where the kernel and the CPU disagreed, if any. */
  mismatchAt: number | null;
}

interface Selection {
  pid: [number, number];
  /** Cell within the tile, 0..B-1. */
  local: [number, number];
}

const MASK = (1n << 64n) - 1n;

/** xorshift64*, same as `cpu::random_soup`. */
function randomSoup(seed: bigint): Uint8Array {
  let x = seed | 1n;
  const threshold = BigInt(Math.floor(DENSITY * 0xffffffff));
  const out = new Uint8Array(ROWS * COLS);
  for (let k = 0; k < out.length; k++) {
    x ^= x >> 12n;
    x ^= (x << 25n) & MASK;
    x ^= x >> 27n;
    const r = ((x * 0x2545f4914f6cdd1dn) & MASK) >> 32n;
    out[k] = r < threshold ? 1 : 0;
  }
  return out;
}

/** Gliders heading in all four directions, so some cross every edge. */
function gliders(seed: number): Uint8Array {
  const out = new Uint8Array(ROWS * COLS);
  const shape = [[0, 1], [1, 2], [2, 0], [2, 1], [2, 2]];
  let s = seed * 2654435761 >>> 0;
  const rand = (n: number) => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s % n;
  };
  for (let k = 0; k < 10; k++) {
    const r0 = rand(ROWS);
    const c0 = rand(COLS);
    const [flipR, flipC] = [k % 2 === 1, (k >> 1) % 2 === 1];
    for (const [dr, dc] of shape) {
      const r = (r0 + (flipR ? 2 - dr : dr)) % ROWS;
      const c = (c0 + (flipC ? 2 - dc : dc)) % COLS;
      out[r * COLS + c] = 1;
    }
  }
  return out;
}

/** `Layout::pad`: embed the world, filling ghost tiles by wrap-around. */
function pad(cells: Uint8Array, B: number): Uint8Array {
  const [br, bc] = [ROWS + 2 * B, COLS + 2 * B];
  const buf = new Uint8Array(br * bc);
  for (let r = 0; r < br; r++) {
    const sr = (r + ROWS - (B % ROWS)) % ROWS;
    for (let c = 0; c < bc; c++) {
      const sc = (c + COLS - (B % COLS)) % COLS;
      buf[r * bc + c] = cells[sr * COLS + sc];
    }
  }
  return buf;
}

function makeSim(B: number, cells: Uint8Array): Sim {
  const front = pad(cells, B);
  return {
    B,
    tileRows: ROWS / B,
    tileCols: COLS / B,
    bufRows: ROWS + 2 * B,
    bufCols: COLS + 2 * B,
    front,
    back: new Uint8Array(front.length),
    heat: front.map((v) => (v ? 255 : 0)),
    cells: cells.slice(),
    generation: 0,
    mismatchAt: null,
  };
}

/**
 * What tile program `pid` does, as in `life_step`: the interior tile it
 * computes (ghosts alias the opposite edge) and, per view, the shift and
 * the block it loads.
 */
function programViews(sim: Sim, [p0, p1]: [number, number]) {
  const { B, tileRows, tileCols } = sim;
  let i = p0 === 0 ? tileRows : p0;
  i = p0 === tileRows + 1 ? 1 : i;
  let j = p1 === 0 ? tileCols : p1;
  j = p1 === tileCols + 1 ? 1 : j;
  const up = i - 1;
  const left = j - 1;
  // Offset -1 is (B-1 at block i-1), 0 is (0 at i), +1 is (1 at i).
  const offsets = [B - 1, 0, 1];
  const rowBlocks = [up, i, i];
  const colBlocks = [left, j, j];
  const views = VIEW_NAMES.map((names, a) =>
    names.map((name, b) => ({
      name,
      shift: [offsets[a], offsets[b]] as const,
      block: [rowBlocks[a], colBlocks[b]] as const,
    }))
  );
  return { i, j, views };
}

type Views = ReturnType<typeof programViews>["views"];

/** Buffer cell that `view` reads for local cell (r, c). */
function readCell(
  B: number,
  view: Views[number][number],
  r: number,
  c: number,
): [number, number] {
  return [
    view.shift[0] + view.block[0] * B + r,
    view.shift[1] + view.block[1] * B + c,
  ];
}

/** One generation: every tile program, ghosts included, then swap. */
function kernelStep(sim: Sim) {
  const { B, bufCols, front, back } = sim;
  for (let p0 = 0; p0 < sim.tileRows + 2; p0++) {
    for (let p1 = 0; p1 < sim.tileCols + 2; p1++) {
      const { i, j, views } = programViews(sim, [p0, p1]);
      const flat = views.flat();
      for (let r = 0; r < B; r++) {
        for (let c = 0; c < B; c++) {
          let sum = 0;
          for (const view of flat) {
            const [br, bc] = readCell(B, view, r, c);
            sum += front[br * bufCols + bc];
          }
          const alive = front[(i * B + r) * bufCols + j * B + c];
          // select(is3, one, select(is4, t_mm, zero))
          back[(p0 * B + r) * bufCols + p1 * B + c] = sum === 3
            ? 1
            : sum === 4
            ? alive
            : 0;
        }
      }
    }
  }
  [sim.front, sim.back] = [back, front];
}

/** `cpu::step`: the toroidal reference, straight from the rule. */
function cpuStep(cells: Uint8Array): Uint8Array {
  const next = new Uint8Array(cells.length);
  for (let r = 0; r < ROWS; r++) {
    const up = ((r + ROWS - 1) % ROWS) * COLS;
    const mid = r * COLS;
    const down = ((r + 1) % ROWS) * COLS;
    for (let c = 0; c < COLS; c++) {
      const l = (c + COLS - 1) % COLS;
      const rt = (c + 1) % COLS;
      const sum = cells[up + l] + cells[up + c] + cells[up + rt] +
        cells[mid + l] + cells[mid + c] + cells[mid + rt] +
        cells[down + l] + cells[down + c] + cells[down + rt];
      next[mid + c] = sum === 3 || (sum === 4 && cells[mid + c] === 1) ? 1 : 0;
    }
  }
  return next;
}

function advance(sim: Sim) {
  kernelStep(sim);
  sim.cells = cpuStep(sim.cells);
  sim.generation++;
  const { B, bufCols, front, heat } = sim;
  for (let k = 0; k < front.length; k++) {
    heat[k] = front[k] ? 255 : Math.max(0, heat[k] - 24);
  }
  if (sim.mismatchAt === null) {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (front[(B + r) * bufCols + B + c] !== sim.cells[r * COLS + c]) {
          sim.mismatchAt = sim.generation;
          return;
        }
      }
    }
  }
}

/** Dark blue background, cyan trail, near-white live cells (`shade`). */
function shade(heat: number): [number, number, number] {
  return [
    8 + Math.floor((heat * 200) / 255),
    12 + Math.floor((heat * 240) / 255),
    28 + Math.floor((heat * 227) / 255),
  ];
}

function draw(canvas: HTMLCanvasElement, sim: Sim, sel: Selection) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { B, bufRows, bufCols, tileRows, tileCols } = sim;
  const W = bufCols * CELL;
  const H = bufRows * CELL;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }

  const image = ctx.createImageData(bufCols, bufRows);
  for (let k = 0; k < sim.heat.length; k++) {
    const [r, g, b] = shade(sim.heat[k]);
    image.data.set([r, g, b, 255], k * 4);
  }
  const small = new OffscreenCanvas(bufCols, bufRows);
  small.getContext("2d")!.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, W, H);

  // Dim the ghost ring.
  ctx.fillStyle = "rgba(4, 10, 28, 0.62)";
  const g = B * CELL;
  ctx.fillRect(0, 0, W, g);
  ctx.fillRect(0, H - g, W, g);
  ctx.fillRect(0, g, g, H - 2 * g);
  ctx.fillRect(W - g, g, g, H - 2 * g);

  // Tile grid, with the interior boundary drawn stronger.
  ctx.strokeStyle = "rgba(255, 230, 167, 0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let t = 1; t < tileCols + 2; t++) {
    ctx.moveTo(t * g + 0.5, 0);
    ctx.lineTo(t * g + 0.5, H);
  }
  for (let t = 1; t < tileRows + 2; t++) {
    ctx.moveTo(0, t * g + 0.5);
    ctx.lineTo(W, t * g + 0.5);
  }
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 230, 167, 0.5)";
  ctx.strokeRect(g + 0.5, g + 0.5, W - 2 * g - 1, H - 2 * g - 1);

  const { i, j, views } = programViews(sim, sel.pid);
  const isGhost = i !== sel.pid[0] || j !== sel.pid[1];

  // The interior tile a ghost program stands in for.
  if (isGhost) {
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#80CAF9";
    ctx.lineWidth = 2;
    ctx.strokeRect(j * g + 1, i * g + 1, g - 2, g - 2);
    ctx.setLineDash([]);
  }

  // The nine cells summed for the selected cell.
  ctx.fillStyle = "rgba(128, 202, 249, 0.5)";
  ctx.strokeStyle = "#80CAF9";
  ctx.lineWidth = 1;
  for (const view of views.flat()) {
    const [br, bc] = readCell(B, view, sel.local[0], sel.local[1]);
    ctx.fillRect(bc * CELL, br * CELL, CELL, CELL);
    ctx.strokeRect(bc * CELL + 0.5, br * CELL + 0.5, CELL - 1, CELL - 1);
  }

  // The block this program writes, and the cell being written.
  ctx.strokeStyle = "#E6A758";
  ctx.lineWidth = 2;
  ctx.strokeRect(sel.pid[1] * g + 1, sel.pid[0] * g + 1, g - 2, g - 2);
  ctx.fillStyle = "rgba(230, 167, 88, 0.85)";
  ctx.fillRect(
    (sel.pid[1] * B + sel.local[1]) * CELL + 2,
    (sel.pid[0] * B + sel.local[0]) * CELL + 2,
    CELL - 4,
    CELL - 4,
  );
}

export default function LifeStencil() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tile, setTile] = useState(8);
  const [pattern, setPattern] = useState<Pattern>("soup");
  const [speed, setSpeed] = useState(8);
  const [playing, setPlaying] = useState(false);
  const [sel, setSel] = useState<Selection>({ pid: [1, 1], local: [0, 0] });
  const [, setTick] = useState(0);
  const seedRef = useRef(42);
  const simRef = useRef<Sim>(makeSim(8, randomSoup(42n)));
  const selRef = useRef(sel);
  selRef.current = sel;

  const redraw = () => {
    if (canvasRef.current) {
      draw(canvasRef.current, simRef.current, selRef.current);
    }
  };
  const refresh = () => {
    redraw();
    setTick((t) => t + 1);
  };

  const reseed = (next: Pattern, seed: number) => {
    const cells = next === "soup" ? randomSoup(BigInt(seed)) : gliders(seed);
    simRef.current = makeSim(simRef.current.B, cells);
    refresh();
  };

  useEffect(() => {
    // Start playing unless the reader prefers reduced motion.
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setPlaying(true);
    }
  }, []);

  useEffect(redraw, [sel]);

  useEffect(() => {
    if (!playing) {
      redraw();
      return;
    }
    const canvas = canvasRef.current;
    let visible = true;
    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    if (canvas) observer.observe(canvas);
    let frame = 0;
    let last = performance.now();
    let owed = 0;
    const loop = (now: number) => {
      const dt = Math.min(250, now - last);
      last = now;
      if (visible && !document.hidden) {
        owed += (dt / 1000) * speed;
        if (owed >= 1) {
          for (; owed >= 1; owed--) advance(simRef.current);
          refresh();
        }
      }
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [playing, speed]);

  const sim = simRef.current;
  const { i, j, views } = programViews(sim, sel.pid);
  const isGhost = i !== sel.pid[0] || j !== sel.pid[1];
  const reads = views.flat().map((v) =>
    readCell(sim.B, v, sel.local[0], sel.local[1])
  );
  const inRing = ([r, c]: [number, number]) =>
    r < sim.B || c < sim.B || r >= sim.B + ROWS || c >= sim.B + COLS;
  const ringReads = reads.filter(inRing).length;
  const total = (sim.tileRows + 2) * (sim.tileCols + 2);
  const interior = sim.tileRows * sim.tileCols;

  const selectAt = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor(((clientX - rect.left) / rect.width) * sim.bufCols);
    const row = Math.floor(((clientY - rect.top) / rect.height) * sim.bufRows);
    if (row < 0 || col < 0 || row >= sim.bufRows || col >= sim.bufCols) return;
    const next: Selection = {
      pid: [Math.floor(row / sim.B), Math.floor(col / sim.B)],
      local: [row % sim.B, col % sim.B],
    };
    const cur = selRef.current;
    if (
      next.pid[0] !== cur.pid[0] || next.pid[1] !== cur.pid[1] ||
      next.local[0] !== cur.local[0] || next.local[1] !== cur.local[1]
    ) setSel(next);
  };

  const onKeyDown = (e: KeyboardEvent) => {
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    const move = moves[e.key];
    if (!move) return;
    e.preventDefault();
    const row = Math.min(
      sim.bufRows - 1,
      Math.max(0, sel.pid[0] * sim.B + sel.local[0] + move[0]),
    );
    const col = Math.min(
      sim.bufCols - 1,
      Math.max(0, sel.pid[1] * sim.B + sel.local[1] + move[1]),
    );
    setSel({
      pid: [Math.floor(row / sim.B), Math.floor(col / sim.B)],
      local: [row % sim.B, col % sim.B],
    });
  };

  const changeTile = (B: number) => {
    // Same world, new tiling: keep the cells, rebuild the buffer.
    const cells = simRef.current.cells;
    simRef.current = makeSim(B, cells);
    setTile(B);
    setSel({ pid: [1, 1], local: [0, 0] });
    refresh();
  };

  return (
    <div class="life">
      <div class="control-groups">
        <fieldset>
          <legend>World</legend>
          <label>
            Start from
            <select
              value={pattern}
              onChange={(e) => {
                const next = e.currentTarget.value as Pattern;
                setPattern(next);
                reseed(next, seedRef.current);
              }}
            >
              <option value="soup">Random soup, 30% alive</option>
              <option value="gliders">Ten gliders</option>
            </select>
          </label>
          <label>
            Tile size
            <select
              value={tile}
              onChange={(e) => changeTile(Number(e.currentTarget.value))}
            >
              {TILE_SIZES.map((t) => <option value={t}>{t} × {t}</option>)}
            </select>
          </label>
        </fieldset>
        <fieldset>
          <legend>Run</legend>
          <label class="range">
            <span>
              Speed <output>{speed} gens/s</output>
            </span>
            <input
              type="range"
              min={1}
              max={30}
              value={speed}
              onInput={(e) => setSpeed(Number(e.currentTarget.value))}
            />
          </label>
        </fieldset>
        <div class="control-actions">
          <button type="button" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            disabled={playing}
            onClick={() => {
              advance(simRef.current);
              refresh();
            }}
          >
            Step
          </button>
          <button
            type="button"
            onClick={() => {
              seedRef.current += 1;
              reseed(pattern, seedRef.current);
            }}
          >
            Reseed
          </button>
        </div>
      </div>

      <div class="life-body">
        <div
          class="canvas-wrap"
          style={{ aspectRatio: `${sim.bufCols} / ${sim.bufRows}` }}
        >
          <canvas
            ref={canvasRef}
            class="life-canvas"
            tabIndex={0}
            role="img"
            aria-label={`Game of Life buffer, ${sim.bufRows} by ${sim.bufCols} cells, with a ghost ring ${sim.B} cells wide. Use arrow keys to move the selected cell.`}
            onPointerMove={(e) => selectAt(e.clientX, e.clientY)}
            onPointerDown={(e) => selectAt(e.clientX, e.clientY)}
            onKeyDown={onKeyDown}
          />
        </div>

        <div class="inspector" aria-live="polite">
          <h3>
            Tile program <code>({sel.pid[0]}, {sel.pid[1]})</code>
            {isGhost && <span class="tag">ghost</span>}
          </h3>
          <p>
            {isGhost
              ? `It computes interior tile (${i}, ${j}) on the opposite edge and writes the result here, so the ring always mirrors the far side.`
              : `An interior tile. It computes itself: i = ${i}, j = ${j}.`}
          </p>
          <table class="views">
            <caption>Views loaded, and the block each one reads</caption>
            <tbody>
              {views.map((row) => (
                <tr>
                  {row.map((view) => (
                    <td>
                      <code>{view.name}</code>
                      <span>
                        [{view.block[0]}, {view.block[1]}]
                      </span>
                      <small>
                        shift {view.shift[0]}, {view.shift[1]}
                      </small>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p class="note">
            Cell ({sel.local[0]},{" "}
            {sel.local[1]}) of this tile sums the nine highlighted cells.{" "}
            {ringReads > 0
              ? `${ringReads} of them are in the ghost ring, which already holds a copy of the far edge.`
              : "None of them are in the ghost ring."}
          </p>
        </div>
      </div>

      <dl class="readout-inline">
        <div>
          <dt>Generation</dt>
          <dd>{sim.generation}</dd>
        </div>
        <div>
          <dt>World</dt>
          <dd>
            <code>
              [{ROWS}, {COLS}]
            </code>
          </dd>
        </div>
        <div>
          <dt>Buffer with ghost ring</dt>
          <dd>
            <code>
              [{sim.bufRows}, {sim.bufCols}]
            </code>
          </dd>
        </div>
        <div>
          <dt>Tile programs</dt>
          <dd>
            {total} ({total - interior} ghosts)
          </dd>
        </div>
        <div>
          <dt>Matches the CPU reference</dt>
          <dd class={sim.mismatchAt === null ? "" : "is-partial"}>
            {sim.mismatchAt === null
              ? "Every generation"
              : `No, from generation ${sim.mismatchAt}`}
          </dd>
        </div>
      </dl>
    </div>
  );
}
