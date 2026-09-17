import { useEffect, useRef, useState } from "preact/hooks";
import {
  earlyExitSteps,
  fillTile,
  HOME,
  LOCATIONS,
  renderTile,
  type TileStats,
  toComplex,
  type View,
} from "../lib/mandelbrot.ts";
import {
  canvasPixel,
  makeTiles,
  type Progress,
  shuffle,
  strokeTileGrid,
  type Tile,
} from "../lib/tiles.ts";

const SIZES = [
  { label: "270 × 480", h: 270, w: 480 },
  { label: "540 × 960", h: 540, w: 960 },
  { label: "1080 × 1920", h: 1080, w: 1920 },
];
const TILE_SIZES = [16, 32, 64, 128];
const ITERATIONS = [100, 250, 500, 1000, 2000];
const CHECK_EVERY = [4, 16, 64];

type Order = "shuffled" | "rows";

interface Hover {
  tile: Tile;
  re: number;
  im: number;
}

const nextFrame = (delayMs: number) =>
  new Promise<void>((resolve) =>
    delayMs > 0
      ? setTimeout(resolve, delayMs)
      : requestAnimationFrame(() => resolve())
  );

/** Short, exact-enough decimal for CLI flags and URLs. */
const num = (x: number) => String(Number(x.toPrecision(10)));

function readHash(): { view: View; iters?: number; tile?: number } | null {
  const params = new URLSearchParams(location.hash.slice(1));
  const parts = params.get("view")?.split(",").map(Number);
  if (!parts || parts.length !== 3 || parts.some((p) => !Number.isFinite(p))) {
    return null;
  }
  const iters = Number(params.get("iters"));
  const tile = Number(params.get("tile"));
  return {
    view: { cx: parts[0], cy: parts[1], span: Math.abs(parts[2]) || HOME.span },
    iters: ITERATIONS.includes(iters) ? iters : undefined,
    tile: TILE_SIZES.includes(tile) ? tile : undefined,
  };
}

export default function TileMandelbrot() {
  const [size, setSize] = useState(SIZES[1]);
  const [tileSize, setTileSize] = useState(64);
  const [maxIter, setMaxIter] = useState(250);
  const [order, setOrder] = useState<Order>("shuffled");
  const [parallel, setParallel] = useState(8);
  const [delay, setDelay] = useState(50);
  const [showGrid, setShowGrid] = useState(true);
  const [showCost, setShowCost] = useState(false);
  const [checkEvery, setCheckEvery] = useState(16);
  const [view, setView] = useState<View>(HOME);
  const [progress, setProgress] = useState<Progress>({
    done: 0,
    total: 0,
    ms: 0,
    running: false,
  });
  const [hover, setHover] = useState<Hover | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const runRef = useRef(0);
  const inflightRef = useRef<Tile[]>([]);
  const statsRef = useRef<TileStats[]>([]);
  const shareableRef = useRef(false);
  // The render loop is async, so it reads live settings through refs.
  const live = useRef({
    parallel,
    delay,
    showGrid,
    showCost,
    checkEvery,
    hover,
  });
  live.current = { parallel, delay, showGrid, showCost, checkEvery, hover };

  const { w, h } = size;
  const gridRows = Math.ceil(h / tileSize);
  const gridCols = Math.ceil(w / tileSize);

  function drawOverlay() {
    const ctx = overlayRef.current?.getContext("2d");
    if (!ctx) return;
    const { showGrid, showCost, checkEvery, hover } = live.current;
    ctx.clearRect(0, 0, w, h);
    const scale = w / 960;

    if (showCost) {
      ctx.font = `600 ${
        Math.round(Math.max(11, tileSize / 4.5))
      }px "Spline Sans Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const s of statsRef.current) {
        const t = s.tile;
        const steps = earlyExitSteps(s, maxIter, checkEvery);
        ctx.fillStyle = `rgba(10, 22, 54, ${0.12 + 0.62 * (steps / maxIter)})`;
        ctx.fillRect(t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0);
        if (tileSize >= 32 && t.x1 - t.x0 >= 24 && t.y1 - t.y0 >= 16) {
          ctx.fillStyle = "#FFE6A7";
          ctx.fillText(String(steps), (t.x0 + t.x1) / 2, (t.y0 + t.y1) / 2);
        }
      }
    }

    if (showGrid) {
      ctx.strokeStyle = "rgba(255, 230, 167, 0.28)";
      ctx.lineWidth = Math.max(1, scale);
      strokeTileGrid(ctx, w, h, tileSize);
    }

    ctx.lineWidth = Math.max(2, 2 * scale);
    for (const t of inflightRef.current) {
      ctx.fillStyle = "rgba(230, 167, 88, 0.35)";
      ctx.fillRect(t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0);
      ctx.strokeStyle = "#E6A758";
      ctx.strokeRect(t.x0 + 1, t.y0 + 1, t.x1 - t.x0 - 2, t.y1 - t.y0 - 2);
    }
    if (hover) {
      const t = hover.tile;
      ctx.strokeStyle = "#80CAF9";
      ctx.strokeRect(t.x0 + 1, t.y0 + 1, t.x1 - t.x0 - 2, t.y1 - t.y0 - 2);
    }
  }

  // Restore a shared view from the URL.
  useEffect(() => {
    const shared = readHash();
    if (!shared) return;
    shareableRef.current = true;
    setView(shared.view);
    if (shared.iters) setMaxIter(shared.iters);
    if (shared.tile) setTileSize(shared.tile);
  }, []);

  // Keep the URL in sync once the view has been changed or shared.
  useEffect(() => {
    if (!shareableRef.current) return;
    const params = new URLSearchParams({
      view: [view.cx, view.cy, view.span].map(num).join(","),
      iters: String(maxIter),
      tile: String(tileSize),
    });
    history.replaceState(null, "", `#${params}`);
  }, [view, maxIter, tileSize]);

  // Launch: every tile program runs once.
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    const token = ++runRef.current;
    const image = ctx.createImageData(w, h);
    const tiles = makeTiles(h, w, tileSize);
    for (const t of tiles) fillTile(image.data, w, t);
    ctx.putImageData(image, 0, 0);
    statsRef.current = [];

    const queue = order === "shuffled" ? shuffle([...tiles]) : [...tiles];
    const start = performance.now();
    let done = 0;
    setProgress({ done, total: tiles.length, ms: 0, running: true });

    (async () => {
      while (queue.length) {
        const batch = queue.splice(0, live.current.parallel);
        inflightRef.current = batch;
        drawOverlay();
        await nextFrame(live.current.delay);
        if (runRef.current !== token) return;
        for (const t of batch) {
          statsRef.current.push(renderTile(image.data, w, h, t, view, maxIter));
          ctx.putImageData(image, 0, 0, t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0);
        }
        done += batch.length;
        setProgress({
          done,
          total: tiles.length,
          ms: performance.now() - start,
          running: queue.length > 0,
        });
      }
      inflightRef.current = [];
      drawOverlay();
    })();

    return () => {
      runRef.current++;
      inflightRef.current = [];
    };
  }, [w, h, tileSize, maxIter, view, order]);

  useEffect(drawOverlay, [showGrid, showCost, checkEvery, hover]);

  // Share of the fixed kernel's pixel-steps the early-exit kernel would run.
  let workShare: number | null = null;
  if (!progress.running && progress.total > 0) {
    let early = 0;
    let fixed = 0;
    for (const s of statsRef.current) {
      const px = (s.tile.x1 - s.tile.x0) * (s.tile.y1 - s.tile.y0);
      early += px * earlyExitSteps(s, maxIter, checkEvery);
      fixed += px * maxIter;
    }
    workShare = fixed ? early / fixed : null;
  }

  const pixelAt = (e: MouseEvent) => canvasPixel(overlayRef.current!, e, w, h);

  function onMove(e: MouseEvent) {
    const { x, y } = pixelAt(e);
    const pid0 = Math.floor(y / tileSize);
    const pid1 = Math.floor(x / tileSize);
    const tile: Tile = {
      pid0,
      pid1,
      y0: pid0 * tileSize,
      y1: Math.min((pid0 + 1) * tileSize, h),
      x0: pid1 * tileSize,
      x1: Math.min((pid1 + 1) * tileSize, w),
    };
    setHover({ tile, ...toComplex(view, w, h, x, y) });
  }

  function changeView(next: View) {
    shareableRef.current = true;
    setView(next);
  }

  function onClick(e: MouseEvent) {
    const { x, y } = pixelAt(e);
    const { re, im } = toComplex(view, w, h, x, y);
    changeView({
      cx: re,
      cy: im,
      span: e.shiftKey ? view.span * 2 : view.span / 2,
    });
  }

  const presetIndex = LOCATIONS.findIndex((l) =>
    l.cx === view.cx && l.cy === view.cy && l.span === view.span
  );
  const zoom = HOME.span / view.span;
  const flags = [
    `--width ${w} --height ${h} --iters ${maxIter} --tile ${tileSize}`,
    `--center=${num(view.cx)},${num(view.cy)} --span ${num(view.span)}`,
  ];
  if (checkEvery !== 16) flags.push(`--check-every ${checkEvery}`);
  const command = `cargo run --release -p mandelbrot -- render \\\n  ${
    flags.join(" \\\n  ")
  }`;

  return (
    <div class="mandel">
      <div class="control-groups">
        <fieldset>
          <legend>View</legend>
          <label>
            Location
            <select
              value={presetIndex}
              onChange={(e) => {
                const loc = LOCATIONS[Number(e.currentTarget.value)];
                if (!loc) return;
                changeView({ cx: loc.cx, cy: loc.cy, span: loc.span });
                setMaxIter(loc.iters);
              }}
            >
              {presetIndex === -1 && (
                <option value={-1}>
                  Custom ({zoom >= 1000 ? zoom.toExponential(1) : num(zoom)}×)
                </option>
              )}
              {LOCATIONS.map((l, i) => (
                <option key={l.name} value={i}>{l.name}</option>
              ))}
            </select>
          </label>
          <label>
            Image
            <select
              value={size.label}
              onChange={(e) =>
                setSize(SIZES.find((s) => s.label === e.currentTarget.value)!)}
            >
              {SIZES.map((s) => <option key={s.label}>{s.label}</option>)}
            </select>
          </label>
          <label>
            Iterations
            <select
              value={maxIter}
              onChange={(e) => {
                shareableRef.current = true;
                setMaxIter(Number(e.currentTarget.value));
              }}
            >
              {ITERATIONS.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </fieldset>

        <fieldset>
          <legend>Tile programs</legend>
          <label>
            Tile size
            <select
              value={tileSize}
              onChange={(e) => {
                shareableRef.current = true;
                setTileSize(Number(e.currentTarget.value));
              }}
            >
              {TILE_SIZES.map((t) => (
                <option key={t} value={t}>{t} × {t}</option>
              ))}
            </select>
          </label>
          <label>
            Launch order
            <select
              value={order}
              onChange={(e) => setOrder(e.currentTarget.value as Order)}
            >
              <option value="shuffled">Shuffled</option>
              <option value="rows">Row by row</option>
            </select>
          </label>
          <label class="range">
            <span>
              At once <output>{parallel}</output>
            </span>
            <input
              type="range"
              min={1}
              max={64}
              value={parallel}
              onInput={(e) => setParallel(e.currentTarget.valueAsNumber)}
            />
          </label>
          <label class="range">
            <span>
              Delay <output>{delay} ms</output>
            </span>
            <input
              type="range"
              min={0}
              max={300}
              step={10}
              value={delay}
              onInput={(e) => setDelay(e.currentTarget.valueAsNumber)}
            />
          </label>
        </fieldset>

        <fieldset>
          <legend>Overlay</legend>
          <label class="check">
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.currentTarget.checked)}
            />
            Tile grid
          </label>
          <label class="check">
            <input
              type="checkbox"
              checked={showCost}
              onChange={(e) => setShowCost(e.currentTarget.checked)}
            />
            Early-exit steps
          </label>
          <label>
            Check every
            <select
              value={checkEvery}
              onChange={(e) => setCheckEvery(Number(e.currentTarget.value))}
            >
              {CHECK_EVERY.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </fieldset>

        <div class="control-actions">
          <button type="button" onClick={() => setView({ ...view })}>
            Relaunch
          </button>
          <button
            type="button"
            disabled={presetIndex === 0}
            onClick={() => {
              changeView(HOME);
              setMaxIter(LOCATIONS[0].iters);
            }}
          >
            Reset view
          </button>
        </div>
      </div>

      <div class="canvas-wrap" style={{ aspectRatio: `${w} / ${h}` }}>
        <canvas ref={canvasRef} width={w} height={h} />
        <canvas
          ref={overlayRef}
          width={w}
          height={h}
          class="overlay"
          aria-label="Click to zoom in, Shift-click to zoom out"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          onClick={onClick}
        />
      </div>

      <p class="hover-readout" aria-live="off">
        {hover
          ? (
            <>
              <code>
                program_id ({hover.tile.pid0}, {hover.tile.pid1})
              </code>{" "}
              covers rows {hover.tile.y0}–{hover.tile.y1} and columns{" "}
              {hover.tile.x0}–{hover.tile.x1}. The pixel under the cursor is
              {" "}
              <code>
                c = {hover.re.toFixed(6)} {hover.im < 0 ? "−" : "+"}{" "}
                {Math.abs(hover.im).toFixed(6)}i
              </code>.
            </>
          )
          : "Hover a tile to see its program id. Click to zoom in, Shift-click to zoom out."}
      </p>

      <dl class="readout-inline">
        <div>
          <dt>Tensor</dt>
          <dd>
            <code>[{h}, {w}]</code>
          </dd>
        </div>
        <div>
          <dt>Partition</dt>
          <dd>
            <code>[{tileSize}, {tileSize}]</code>
          </dd>
        </div>
        <div>
          <dt>Launch grid</dt>
          <dd>
            <code>({gridRows}, {gridCols}, 1)</code>
          </dd>
        </div>
        <div>
          <dt>Finished</dt>
          <dd>
            {progress.done} of {progress.total}
            {progress.running ? "" : ` in ${progress.ms.toFixed(0)} ms`}
          </dd>
        </div>
        <div>
          <dt>Early-exit work</dt>
          <dd>
            {workShare === null
              ? "Waiting for the render"
              : `${(workShare * 100).toFixed(1)}% of the fixed kernel`}
          </dd>
        </div>
      </dl>

      <figure class="codeblock">
        <figcaption>
          <span>Render this view on the GPU</span>
          <button type="button" class="copy" data-copy>Copy</button>
        </figcaption>
        <pre><code>{command}</code></pre>
      </figure>
    </div>
  );
}
