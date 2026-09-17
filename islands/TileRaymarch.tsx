import { useEffect, useRef, useState } from "preact/hooks";
import {
  type Camera,
  DEFAULT_CAMERA,
  DISTANCE_RANGE,
  HIT,
  makeFrame,
  makeTarget,
  OUT_OF_STEPS,
  paintTile,
  PITCH_RANGE,
  QUALITIES,
  renderTile,
  SKY,
  type Target,
  TILE,
  type View,
} from "../lib/raymarch.ts";
import {
  canvasPixel,
  type Progress,
  shuffle,
  strokeTileGrid,
} from "../lib/tiles.ts";

const SIZES = [
  { label: "180 × 320", h: 180, w: 320 },
  { label: "360 × 640", h: 360, w: 640 },
  { label: "720 × 1280", h: 720, w: 1280 },
];
const CHECK_EVERY = [
  { value: 4, label: "4 steps" },
  { value: 16, label: "16 steps" },
  { value: 64, label: "64 steps" },
  { value: 0, label: "Never" },
];
const VIEWS: { value: View; label: string }[] = [
  { value: "shaded", label: "Shaded" },
  { value: "steps", label: "Steps per pixel" },
  { value: "tiles", label: "Steps per tile" },
];

/** Milliseconds of tile programs to run before handing a frame back. */
const SLICE_MS = 12;
/** Wait this long after the last change before relaunching at full size. */
const SETTLE_MS = 150;
const TAU = Math.PI * 2;
const DEG = 180 / Math.PI;

interface Hover {
  x: number;
  y: number;
}

const clamp = (x: number, [lo, hi]: readonly [number, number]) =>
  Math.min(Math.max(x, lo), hi);

/** Short decimal for CLI flags. */
const num = (x: number, digits = 2) => String(Number(x.toFixed(digits)));

/** About 80 preview rays across, so a preview costs a few milliseconds. */
const previewStride = (w: number) =>
  Math.min(TILE, 2 ** Math.ceil(Math.log2(w / 80)));

export default function TileRaymarch() {
  const [size, setSize] = useState(SIZES[1]);
  const [qualityIndex, setQualityIndex] = useState(1);
  const [checkEvery, setCheckEvery] = useState(16);
  const [camera, setCamera] = useState<Camera>(DEFAULT_CAMERA);
  const [time, setTime] = useState(0);
  const [view, setView] = useState<View>("shaded");
  const [showGrid, setShowGrid] = useState(true);
  const [markSkipped, setMarkSkipped] = useState(false);
  const [hover, setHover] = useState<Hover | null>(null);
  const [relaunch, setRelaunch] = useState(0);
  const [started, setStarted] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<Progress>({
    done: 0,
    total: 0,
    ms: 0,
    running: false,
  });

  const rootRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<ImageData | null>(null);
  const fullRef = useRef<Target | null>(null);
  const previewRef = useRef<Target | null>(null);
  const runRef = useRef(0);
  const dragRef = useRef<{ id: number; x: number; y: number } | null>(null);
  // The launch loop is async, so it reads display settings through a ref.
  const live = useRef({ view, showGrid, markSkipped, hover });
  live.current = { view, showGrid, markSkipped, hover };

  const { w, h } = size;
  const quality = QUALITIES[qualityIndex];
  const gridRows = Math.ceil(h / TILE);
  const gridCols = Math.ceil(w / TILE);

  function paint(tiles?: [number, number][]) {
    const ctx = canvasRef.current?.getContext("2d");
    const image = imageRef.current;
    const full = fullRef.current;
    if (!ctx || !image || !full) return;
    const { view } = live.current;
    if (tiles) {
      for (const [r, c] of tiles) {
        paintTile(image.data, full, previewRef.current, view, r, c);
      }
    } else {
      for (let r = 0; r < full.gridRows; r++) {
        for (let c = 0; c < full.gridCols; c++) {
          paintTile(image.data, full, previewRef.current, view, r, c);
        }
      }
    }
    ctx.putImageData(image, 0, 0);
  }

  function drawOverlay() {
    const ctx = overlayRef.current?.getContext("2d");
    const full = fullRef.current;
    if (!ctx || !full) return;
    const { view, showGrid, markSkipped, hover } = live.current;
    ctx.clearRect(0, 0, w, h);
    const scale = w / 640;

    if (markSkipped) {
      ctx.fillStyle = "rgba(10, 26, 64, 0.45)";
      ctx.strokeStyle = "rgba(128, 202, 249, 0.75)";
      ctx.lineWidth = Math.max(1, scale);
      for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
          const tile = r * gridCols + c;
          if (!full.done[tile] || full.anyHit[tile]) continue;
          const x0 = c * TILE;
          const y0 = r * TILE;
          ctx.fillRect(x0, y0, TILE, TILE);
          ctx.beginPath();
          for (let d = 8; d < TILE * 2; d += 8) {
            ctx.moveTo(x0 + Math.max(0, d - TILE), y0 + Math.min(d, TILE));
            ctx.lineTo(x0 + Math.min(d, TILE), y0 + Math.max(0, d - TILE));
          }
          ctx.stroke();
        }
      }
    }

    if (view === "tiles") {
      ctx.font = `600 ${
        Math.round(TILE / 3.2)
      }px "Spline Sans Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
          const tile = r * gridCols + c;
          // Skip tiles the image edge cuts too short to hold a label.
          if (!full.done[tile] || h - r * TILE < TILE / 2) continue;
          const steps = full.tileSteps[tile];
          const cy = Math.min((r + 0.5) * TILE, (r * TILE + h) / 2);
          // Navy on the orange-to-white end of the heat ramp, gold below.
          ctx.fillStyle = steps / full.maxSteps > 0.34 ? "#0A1A40" : "#FFE6A7";
          ctx.fillText(String(steps), (c + 0.5) * TILE, cy);
        }
      }
    }

    if (showGrid) {
      ctx.strokeStyle = "rgba(255, 230, 167, 0.3)";
      ctx.lineWidth = Math.max(1, scale);
      strokeTileGrid(ctx, w, h, TILE);
    }

    if (hover) {
      const x0 = Math.floor(hover.x / TILE) * TILE;
      const y0 = Math.floor(hover.y / TILE) * TILE;
      ctx.strokeStyle = "#80CAF9";
      ctx.lineWidth = Math.max(2, 2 * scale);
      ctx.strokeRect(
        x0 + 1,
        y0 + 1,
        Math.min(TILE, w - x0) - 2,
        Math.min(TILE, h - y0) - 2,
      );
    }
  }

  // Launch once the demo is near the viewport, not on page load.
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      setStarted(true);
      observer.disconnect();
    }, { rootMargin: "200px" });
    observer.observe(root);
    return () => observer.disconnect();
  }, []);

  // Launch: a coarse preview right away, then every tile program at full
  // size once the settings stop changing.
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!started || !ctx) return;
    const token = ++runRef.current;
    const frame = makeFrame(w, h, camera, time);

    const preview = makeTarget(w, h, previewStride(w), quality.maxSteps);
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        renderTile(frame, quality, checkEvery, preview, r, c);
      }
    }
    previewRef.current = preview;
    const full = makeTarget(w, h, 1, quality.maxSteps);
    fullRef.current = full;
    if (imageRef.current?.width !== w || imageRef.current?.height !== h) {
      imageRef.current = ctx.createImageData(w, h);
    }
    paint();
    drawOverlay();

    const tiles: [number, number][] = [];
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) tiles.push([r, c]);
    }
    const queue = shuffle(tiles);
    setProgress({ done: 0, total: queue.length, ms: 0, running: true });
    if (dragging) return () => runRef.current++;

    let raf = 0;
    let busy = 0;
    let done = 0;
    const slice = () => {
      if (runRef.current !== token) return;
      const start = performance.now();
      const batch: [number, number][] = [];
      do {
        const [r, c] = queue[done++];
        renderTile(frame, quality, checkEvery, full, r, c);
        batch.push([r, c]);
      } while (done < queue.length && performance.now() - start < SLICE_MS);
      busy += performance.now() - start;
      paint(batch);
      drawOverlay();
      setProgress({
        done,
        total: queue.length,
        ms: busy,
        running: done < queue.length,
      });
      if (done < queue.length) raf = requestAnimationFrame(slice);
    };
    const timer = setTimeout(() => {
      raf = requestAnimationFrame(slice);
    }, SETTLE_MS);

    return () => {
      runRef.current++;
      clearTimeout(timer);
      cancelAnimationFrame(raf);
    };
  }, [
    started,
    dragging,
    w,
    h,
    qualityIndex,
    checkEvery,
    camera,
    time,
    relaunch,
  ]);

  useEffect(() => paint(), [view]);
  useEffect(drawOverlay, [view, showGrid, markSkipped, hover]);

  // Zoom with the wheel only while the image has focus, so the page still
  // scrolls past it.
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const onWheel = (e: WheelEvent) => {
      if (document.activeElement !== overlay || e.deltaY === 0) return;
      e.preventDefault();
      zoom(1 + Math.sign(e.deltaY) * 0.08);
    };
    overlay.addEventListener("wheel", onWheel, { passive: false });
    return () => overlay.removeEventListener("wheel", onWheel);
  }, []);

  function orbit(dYaw: number, dPitch: number) {
    setCamera((cam) => ({
      ...cam,
      yaw: ((cam.yaw + dYaw) % TAU + TAU) % TAU,
      pitch: clamp(cam.pitch + dPitch, PITCH_RANGE),
    }));
  }

  function zoom(factor: number) {
    setCamera((cam) => ({
      ...cam,
      distance: clamp(cam.distance * factor, DISTANCE_RANGE),
    }));
  }

  const pixelAt = (e: PointerEvent) =>
    canvasPixel(overlayRef.current!, e, w, h);

  function onPointerDown(e: PointerEvent) {
    if (e.button !== 0) return;
    const overlay = e.currentTarget as HTMLCanvasElement;
    overlay.setPointerCapture(e.pointerId);
    overlay.focus({ preventScroll: true });
    dragRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY };
  }

  function onPointerMove(e: PointerEvent) {
    const drag = dragRef.current;
    if (drag?.id === e.pointerId) {
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (!dragging && Math.hypot(dx, dy) < 3) return;
      // The window turns 0.006 rad per pixel across 1280.
      const rect = overlayRef.current!.getBoundingClientRect();
      const k = 0.006 * 1280 / rect.width;
      dragRef.current = { ...drag, x: e.clientX, y: e.clientY };
      setDragging(true);
      setHover(null);
      orbit(-dx * k, dy * k);
      return;
    }
    setHover(pixelAt(e));
  }

  function endDrag(e: PointerEvent) {
    if (dragRef.current?.id !== e.pointerId) return;
    dragRef.current = null;
    setDragging(false);
  }

  function onKeyDown(e: KeyboardEvent) {
    const turn = 0.05;
    const actions: Record<string, () => void> = {
      ArrowLeft: () => orbit(turn, 0),
      ArrowRight: () => orbit(-turn, 0),
      ArrowUp: () => orbit(0, turn),
      ArrowDown: () => orbit(0, -turn),
      "+": () => zoom(0.92),
      "=": () => zoom(0.92),
      "-": () => zoom(1.08),
    };
    const action = actions[e.key];
    if (!action) return;
    e.preventDefault();
    action();
  }

  const full = fullRef.current;
  let workShare: number | null = null;
  let skipped = 0;
  if (full && !progress.running && progress.total > 0) {
    let steps = 0;
    for (let i = 0; i < full.tileSteps.length; i++) {
      steps += full.tileSteps[i];
      if (!full.anyHit[i]) skipped++;
    }
    workShare = steps / (full.tileSteps.length * full.maxSteps);
  }

  let hoverText = null;
  if (hover && full && full.width === w && full.height === h) {
    const pid0 = Math.floor(hover.y / TILE);
    const pid1 = Math.floor(hover.x / TILE);
    const tile = pid0 * gridCols + pid1;
    const id = <code>program_id ({pid0}, {pid1})</code>;
    if (!full.done[tile]) {
      hoverText = <>{id} hasn't run yet. The coarse preview stands in.</>;
    } else {
      const o = hover.y * w + hover.x;
      const steps = full.steps[o];
      const ending = full.ending[o];
      const below = (pid0 + 1) * TILE - h;
      const right = (pid1 + 1) * TILE - w;
      const outside = TILE * TILE -
        (TILE - Math.max(below, 0)) * (TILE - Math.max(right, 0));
      hoverText = (
        <>
          {id} ran {full.tileSteps[tile]} of {full.maxSteps} march steps, then
          {" "}
          {full.anyHit[tile]
            ? "shaded its pixels"
            : "skipped shading, because none of its rays hit anything"}.
          {outside > 0 &&
            ` It also marched ${outside} rays outside the image, since every tile is 32 × 32.`}
          {" "}
          The pixel under the cursor{" "}
          {ending === HIT && `hit a surface after ${steps} steps.`}
          {ending === SKY && `passed the far limit after ${steps} steps.`}
          {ending === OUT_OF_STEPS &&
            `was still marching after ${steps} steps, so it's shaded as a hit.`}
        </>
      );
    }
  }

  const flags = [`--width ${w} --height ${h} --quality ${qualityIndex + 1}`];
  if (camera !== DEFAULT_CAMERA) {
    flags.push(
      `--yaw ${num(camera.yaw * DEG)} --pitch ${
        num(camera.pitch * DEG)
      } --distance ${num(camera.distance)}`,
    );
  }
  const scene: string[] = [];
  if (time !== 0) scene.push(`--time ${num(time, 1)}`);
  if (view !== "shaded") scene.push(`--view ${view}`);
  if (scene.length) flags.push(scene.join(" "));
  const command = `cargo run --release -p raymarch -- render \\\n  ${
    flags.join(" \\\n  ")
  }`;

  return (
    <div class="raymarch" ref={rootRef}>
      <div class="control-groups">
        <fieldset>
          <legend>Camera</legend>
          <label class="range">
            <span>
              Yaw <output>{Math.round(camera.yaw * DEG)}°</output>
            </span>
            <input
              type="range"
              min={0}
              max={359}
              value={Math.round(camera.yaw * DEG)}
              onInput={(e) =>
                setCamera({
                  ...camera,
                  yaw: e.currentTarget.valueAsNumber / DEG,
                })}
            />
          </label>
          <label class="range">
            <span>
              Pitch <output>{Math.round(camera.pitch * DEG)}°</output>
            </span>
            <input
              type="range"
              min={Math.ceil(PITCH_RANGE[0] * DEG)}
              max={Math.floor(PITCH_RANGE[1] * DEG)}
              value={Math.round(camera.pitch * DEG)}
              onInput={(e) =>
                setCamera({
                  ...camera,
                  pitch: e.currentTarget.valueAsNumber / DEG,
                })}
            />
          </label>
          <label class="range">
            <span>
              Distance <output>{camera.distance.toFixed(1)}</output>
            </span>
            <input
              type="range"
              min={DISTANCE_RANGE[0]}
              max={DISTANCE_RANGE[1]}
              step={0.1}
              value={camera.distance}
              onInput={(e) =>
                setCamera({
                  ...camera,
                  distance: e.currentTarget.valueAsNumber,
                })}
            />
          </label>
          <label class="range">
            <span>
              Time <output>{time.toFixed(1)} s</output>
            </span>
            <input
              type="range"
              min={0}
              max={9}
              step={0.1}
              value={time}
              onInput={(e) => setTime(e.currentTarget.valueAsNumber)}
            />
          </label>
        </fieldset>

        <fieldset>
          <legend>Kernel</legend>
          <label>
            Image
            <select
              value={size.label}
              onChange={(e) => {
                setHover(null);
                setSize(SIZES.find((s) => s.label === e.currentTarget.value)!);
              }}
            >
              {SIZES.map((s) => <option key={s.label}>{s.label}</option>)}
            </select>
          </label>
          <label>
            Quality
            <select
              value={qualityIndex}
              onChange={(e) => setQualityIndex(Number(e.currentTarget.value))}
            >
              {QUALITIES.map((q, i) => (
                <option key={q.label} value={i}>
                  {q.label} ({q.maxSteps} / {q.shadowSteps} steps)
                </option>
              ))}
            </select>
          </label>
          <label>
            Check for finished rays every
            <select
              value={checkEvery}
              onChange={(e) => setCheckEvery(Number(e.currentTarget.value))}
            >
              {CHECK_EVERY.map((c) => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </label>
        </fieldset>

        <fieldset>
          <legend>View</legend>
          <label>
            Show
            <select
              value={view}
              onChange={(e) => setView(e.currentTarget.value as View)}
            >
              {VIEWS.map((v) => (
                <option key={v.value} value={v.value}>{v.label}</option>
              ))}
            </select>
          </label>
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
              checked={markSkipped}
              onChange={(e) => setMarkSkipped(e.currentTarget.checked)}
            />
            Tiles that skip shading
          </label>
        </fieldset>

        <div class="control-actions">
          <button type="button" onClick={() => setRelaunch((n) => n + 1)}>
            Relaunch
          </button>
          <button
            type="button"
            disabled={camera === DEFAULT_CAMERA && time === 0}
            onClick={() => {
              setCamera(DEFAULT_CAMERA);
              setTime(0);
            }}
          >
            Reset camera
          </button>
        </div>
      </div>

      <div class="canvas-wrap" style={{ aspectRatio: `${w} / ${h}` }}>
        <canvas ref={canvasRef} width={w} height={h} />
        <canvas
          ref={overlayRef}
          width={w}
          height={h}
          class={dragging ? "overlay is-dragging" : "overlay"}
          tabIndex={0}
          aria-label="Ray-marched scene. Drag or use the arrow keys to orbit, and plus or minus to zoom."
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onPointerLeave={() => setHover(null)}
          onKeyDown={onKeyDown}
        />
      </div>

      <p class="hover-readout" aria-live="off">
        {hoverText ??
          "Point at a tile to see how many steps its program ran. Drag to orbit, and click the image first to zoom with the wheel."}
      </p>

      <dl class="readout-inline">
        <div>
          <dt>Image</dt>
          <dd>
            <code>[{h}, {w}]</code>
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
            {progress.total === 0
              ? "Waiting for the launch"
              : `${progress.done} of ${progress.total}${
                progress.running ? "" : ` in ${progress.ms.toFixed(0)} ms`
              }`}
          </dd>
        </div>
        <div>
          <dt>March steps</dt>
          <dd>
            {workShare === null
              ? "Waiting for the launch"
              : `${
                (workShare * 100).toFixed(1)
              }% of a kernel without early exit`}
          </dd>
        </div>
        <div>
          <dt>Skipped shading</dt>
          <dd>
            {workShare === null
              ? "Waiting for the launch"
              : `${skipped} of ${progress.total} tiles`}
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
