import { useRef, useState } from "preact/hooks";

type Rank = 1 | 2;

interface Config {
  rank: Rank;
  shape: [number, number];
  part: [number, number];
}

const PRESETS: { label: string; config: Config }[] = [
  { label: "Tutorial 02", config: { rank: 2, shape: [32, 32], part: [4, 4] } },
  {
    label: "Mental models",
    config: { rank: 2, shape: [128, 256], part: [32, 64] },
  },
  {
    label: "README (1-D)",
    config: { rank: 1, shape: [1024, 1], part: [128, 1] },
  },
  { label: "Uneven", config: { rank: 2, shape: [100, 150], part: [32, 32] } },
  { label: "Image", config: { rank: 2, shape: [480, 640], part: [64, 64] } },
];

const MAX_DIM = 65536;
const MAX_LINES_PER_AXIS = 512;
const MAX_LABELS = 64;

const clamp = (n: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, Number.isFinite(n) ? Math.round(n) : lo));

export default function GridExplorer() {
  const [cfg, setCfg] = useState<Config>(PRESETS[1].config);
  const [sel, setSel] = useState<[number, number]>([1, 2]);
  const svgRef = useRef<SVGSVGElement>(null);

  const { rank } = cfg;
  const shape = cfg.shape.slice(0, rank);
  const part = cfg.part.slice(0, rank);
  const grid = shape.map((d, i) => Math.ceil(d / part[i]));
  const fullGrid = shape.map((d, i) => Math.floor(d / part[i]));
  const total = grid.reduce((a, b) => a * b, 1);
  const partialCount = total - fullGrid.reduce((a, b) => a * b, 1);
  const pid = grid.map((g, i) => Math.min(sel[i], g - 1));
  const region = pid.map((
    p,
    i,
  ) => [p * part[i], Math.min((p + 1) * part[i], shape[i])]);
  const isPartial = region.some(([a, b], i) => b - a < part[i]);

  // SVG space: x runs along the last tensor dim, y along dim 0 (rank 2).
  const rows = rank === 2
    ? { size: shape[0], part: part[0], n: grid[0] }
    : null;
  const cols = rank === 2
    ? { size: shape[1], part: part[1], n: grid[1] }
    : { size: shape[0], part: part[0], n: grid[0] };
  const vbW = cols.size;
  const vbH = rows ? rows.size : Math.max(1, cols.size / 10);
  const drawLines = cols.n <= MAX_LINES_PER_AXIS &&
    (!rows || rows.n <= MAX_LINES_PER_AXIS);

  const selRect = rank === 2
    ? {
      x: region[1][0],
      y: region[0][0],
      w: region[1][1] - region[1][0],
      h: region[0][1] - region[0][0],
    }
    : { x: region[0][0], y: 0, w: region[0][1] - region[0][0], h: vbH };

  function update(next: Partial<Config>) {
    setCfg((c) => ({ ...c, ...next }));
  }

  function setDim(key: "shape" | "part", i: number, value: number) {
    const arr = [...cfg[key]] as [number, number];
    arr[i] = clamp(value, 1, MAX_DIM);
    update({ [key]: arr });
  }

  function onPointer(e: PointerEvent) {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    if (p.x < 0 || p.y < 0 || p.x >= vbW || p.y >= vbH) return;
    const c = Math.floor(p.x / cols.part);
    if (rows) setSel([Math.floor(p.y / rows.part), c]);
    else setSel([c, 0]);
  }

  const dims = (xs: number[]) => xs.join(", ");
  const gridTuple = `(${grid[0]}, ${grid[1] ?? 1}, 1)`;

  return (
    <div class="explorer">
      <div class="control-groups">
        <div class="presets" role="group" aria-label="Examples">
          {PRESETS.map((p) => (
            <button
              type="button"
              aria-pressed={JSON.stringify(p.config) === JSON.stringify(cfg)}
              onClick={() => {
                setCfg(p.config);
                setSel([0, 0]);
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
        <fieldset>
          <legend>Shapes</legend>
          <label>
            Rank
            <select
              value={rank}
              onChange={(e) =>
                update({ rank: Number(e.currentTarget.value) as Rank })}
            >
              <option value={1}>1-D</option>
              <option value={2}>2-D</option>
            </select>
          </label>
          <div class="dims" role="group" aria-label="Tensor shape">
            <span>Tensor shape</span>
            {shape.map((d, i) => (
              <input
                type="number"
                min={1}
                max={MAX_DIM}
                value={d}
                aria-label={`Tensor dim ${i}`}
                onInput={(e) =>
                  setDim("shape", i, e.currentTarget.valueAsNumber)}
              />
            ))}
          </div>
          <div class="dims" role="group" aria-label="Partition shape">
            <span>Partition</span>
            {part.map((d, i) => (
              <input
                type="number"
                min={1}
                max={MAX_DIM}
                value={d}
                aria-label={`Partition dim ${i}`}
                onInput={(e) =>
                  setDim("part", i, e.currentTarget.valueAsNumber)}
              />
            ))}
          </div>
        </fieldset>
      </div>

      <div class="explorer-body">
        <div class="svg-wrap">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${vbW} ${vbH}`}
            preserveAspectRatio="xMidYMid meet"
            onPointerMove={onPointer}
            onPointerDown={onPointer}
            role="img"
            aria-label={`Tensor [${
              dims(shape)
            }] split into a ${gridTuple} grid`}
          >
            <defs>
              <pattern
                id="hatch"
                width="6"
                height="6"
                patternUnits="userSpaceOnUse"
                patternTransform={`rotate(45) scale(${vbW / 400})`}
              >
                <line x1="0" y1="0" x2="0" y2="6" class="hatch-line" />
              </pattern>
            </defs>
            <rect class="tensor" x={0} y={0} width={vbW} height={vbH} />
            {cols.size % cols.part !== 0 && (
              <rect
                fill="url(#hatch)"
                x={(cols.n - 1) * cols.part}
                y={0}
                width={cols.size - (cols.n - 1) * cols.part}
                height={vbH}
              />
            )}
            {rows && rows.size % rows.part !== 0 && (
              <rect
                fill="url(#hatch)"
                x={0}
                y={(rows.n - 1) * rows.part}
                width={vbW}
                height={rows.size - (rows.n - 1) * rows.part}
              />
            )}
            {drawLines && (
              <g class="grid-lines">
                {Array.from({ length: cols.n - 1 }, (_, k) => (
                  <line
                    x1={(k + 1) * cols.part}
                    x2={(k + 1) * cols.part}
                    y1={0}
                    y2={vbH}
                  />
                ))}
                {rows && Array.from({ length: rows.n - 1 }, (_, k) => (
                  <line
                    y1={(k + 1) * rows.part}
                    y2={(k + 1) * rows.part}
                    x1={0}
                    x2={vbW}
                  />
                ))}
              </g>
            )}
            <rect
              class="selected"
              x={selRect.x}
              y={selRect.y}
              width={selRect.w}
              height={selRect.h}
            />
            {total <= MAX_LABELS && (
              <g class="labels">
                {Array.from({ length: total }, (_, i) => {
                  const r = rows ? Math.floor(i / cols.n) : 0;
                  const c = rows ? i % cols.n : i;
                  const x0 = c * cols.part;
                  const x1 = Math.min(x0 + cols.part, cols.size);
                  const y0 = rows ? r * rows.part : 0;
                  const y1 = rows ? Math.min(y0 + rows.part, rows.size) : vbH;
                  const size = Math.min(x1 - x0, y1 - y0) * 0.3;
                  // Skip slivers (partial edge tiles) too thin to read.
                  if (size < Math.max(vbW, vbH) / 60) return null;
                  return (
                    <text
                      x={(x0 + x1) / 2}
                      y={(y0 + y1) / 2}
                      font-size={size}
                      dominant-baseline="central"
                      text-anchor="middle"
                    >
                      {rows ? `${r},${c}` : c}
                    </text>
                  );
                })}
              </g>
            )}
          </svg>
          <p class="axis-note">
            {rows
              ? "Rows are tensor dim 0, grid x, program_id(0). Columns are dim 1, grid y, program_id(1)."
              : "Tensor dim 0 is grid x, program_id(0)."}
            {!drawLines && " The grid is too dense to draw."}
          </p>
        </div>

        <dl class="readout">
          <div>
            <dt>Launch grid</dt>
            <dd>{gridTuple}</dd>
          </div>
          <div>
            <dt>Tile programs</dt>
            <dd>{total.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Elements per full tile</dt>
            <dd>{part.reduce((a, b) => a * b, 1).toLocaleString()}</dd>
          </div>
          <div>
            <dt>Partial edge tiles</dt>
            <dd class={partialCount ? "is-partial" : ""}>
              {partialCount.toLocaleString()}
            </dd>
          </div>
          <div class="wide">
            <dt>Selected program</dt>
            <dd>
              {pid.map((p, i) => (
                <code class="pid">
                  program_id({i}) = <b>{p}</b>
                </code>
              ))}
            </dd>
            <dd class="note">
              Writes {region
                .map(([a, b], i) => `dim ${i} ${a}–${b}`)
                .join(", ")}.
              {isPartial && " This is a partial edge tile."}
            </dd>
          </div>
        </dl>
      </div>

      <figure class="codeblock">
        <figcaption>
          <span>Host code for this partition</span>
          <button type="button" class="copy" data-copy>Copy</button>
        </figcaption>
        <pre><code>{`let z = api::zeros::<f32>(&[${dims(shape)}]).partition([${dims(part)}]);
// launch grid ${gridTuple}: ${total.toLocaleString()} tile program${total === 1 ? "" : "s"}
// each one sees  z: &mut Tensor<f32, { [${dims(part)}] }>`}</code></pre>
      </figure>
    </div>
  );
}
