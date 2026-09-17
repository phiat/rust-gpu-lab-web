import { useEffect, useRef, useState } from "preact/hooks";
import {
  fillTile,
  LOCATIONS,
  makeTiles,
  renderTile,
  shuffle,
} from "../lib/mandelbrot.ts";

const W = 960;
const H = 600;
const TILE = 60;
/** How long one location takes to fill, whatever the display's frame rate. */
const FILL_MS = 2400;
const HOLD_MS = 4500;

interface Status {
  name: string;
  done: number;
  total: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const frame = () => new Promise((r) => requestAnimationFrame(r));

/**
 * The home page's live render. Tiles land in shuffled order, the way tile
 * programs finish on a GPU, then it moves on to the next location.
 */
export default function HeroRender() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    let cancelled = false;
    let visible = true;
    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    observer.observe(canvas);
    const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const paused = () => !visible || document.hidden;

    (async () => {
      // Start somewhere other than the whole set; it's on every other page.
      const order = [...LOCATIONS.slice(1), LOCATIONS[0]];
      for (let i = 0; !cancelled; i = (i + 1) % order.length) {
        const loc = order[i];
        const image = ctx.createImageData(W, H);
        const tiles = makeTiles(H, W, TILE);
        for (const t of tiles) fillTile(image.data, W, t);
        ctx.putImageData(image, 0, 0);

        // Reduced motion: paint everything at once, and don't loop.
        const queue = reduceMotion ? [...tiles] : shuffle([...tiles]);
        const total = tiles.length;
        let done = 0;
        let elapsed = 0;
        let last = performance.now();
        setStatus({ name: loc.name, done, total });

        while (queue.length && !cancelled) {
          if (paused()) {
            await sleep(300);
            last = performance.now();
            continue;
          }
          const now = performance.now();
          elapsed += now - last;
          last = now;
          const due = reduceMotion
            ? queue.length
            : Math.max(1, Math.ceil((elapsed / FILL_MS) * total) - done);
          for (const t of queue.splice(0, due)) {
            renderTile(image.data, W, H, t, loc, loc.iters);
            ctx.putImageData(image, 0, 0, t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0);
            done++;
          }
          setStatus({ name: loc.name, done, total });
          await frame();
        }
        if (reduceMotion) break;

        for (let waited = 0; waited < HOLD_MS && !cancelled;) {
          await sleep(250);
          if (!paused()) waited += 250;
        }
      }
    })();

    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, []);

  const finished = status && status.done === status.total;

  return (
    <figure class="hero-render">
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        role="img"
        aria-label="A Mandelbrot set rendering, filled in one tile at a time"
      />
      <figcaption>
        {status
          ? (
            <>
              <strong>{status.name}.</strong> {finished
                ? `All ${status.total} tile programs finished.`
                : `${status.done} of ${status.total} tile programs finished, in no particular order.`}
            </>
          )
          : "Starting 160 tile programs."}
      </figcaption>
    </figure>
  );
}
