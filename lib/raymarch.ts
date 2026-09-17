/**
 * A port of the raymarch crate's `render` kernel (`raymarch/src/gpu.rs`, with
 * the scalar math from `cpu.rs`): the same scene, sphere tracing, shading and
 * debug views, organized the way the kernel is. Every 32 × 32 tile program
 * marches all of its rays in lockstep, checks every `checkEvery` steps whether
 * any ray is still marching, and stops when none is. Edge tiles march the full
 * 32 × 32 even where the image ends, because the partition only masks stores.
 *
 * JavaScript numbers are 64-bit, so this matches the crate closely but not bit
 * for bit.
 */

export const TILE = 32;
const PER_TILE = TILE * TILE;

export type View = "shaded" | "steps" | "tiles";

export interface Camera {
  /** Radians around the vertical axis; 0 looks down -z. */
  yaw: number;
  /** Radians above the horizon. */
  pitch: number;
  distance: number;
}

/** `Camera::default()`, with its 60° vertical field of view. */
export const DEFAULT_CAMERA: Camera = { yaw: 0.6, pitch: 0.28, distance: 4.6 };
const FOV = 60;

/** The window's clamps: stay above the ground and inside the inner ring. */
export const PITCH_RANGE = [0.02, 1.45] as const;
export const DISTANCE_RANGE = [1.5, 6] as const;

export interface Quality {
  label: string;
  maxSteps: number;
  shadowSteps: number;
}

/** `Quality::PRESETS`. All three check for finished rays every 16 steps. */
export const QUALITIES: Quality[] = [
  { label: "Low", maxSteps: 64, shadowSteps: 16 },
  { label: "Medium", maxSteps: 128, shadowSteps: 32 },
  { label: "High", maxSteps: 256, shadowSteps: 64 },
];

/** The parameter block (`scene::slot`), flattened into named fields. */
export interface Frame {
  width: number;
  height: number;
  ex: number;
  ey: number;
  ez: number;
  fx: number;
  fy: number;
  fz: number;
  rx: number;
  ry: number;
  rz: number;
  ux: number;
  uy: number;
  uz: number;
  focal: number;
  lx: number;
  ly: number;
  lz: number;
  /** sin and cos of the spin angle, and the blob's height. */
  rs: number;
  rc: number;
  bob: number;
}

const TARGET_Y = 0.7;

/** `Frame::params`: camera basis, sun and animation for one image. */
export function makeFrame(
  width: number,
  height: number,
  camera: Camera,
  time: number,
): Frame {
  const { yaw, pitch, distance } = camera;
  const cp = Math.cos(pitch);
  const ex = distance * cp * Math.sin(yaw);
  const ey = TARGET_Y + distance * Math.sin(pitch);
  const ez = distance * cp * Math.cos(yaw);
  const [fx, fy, fz] = normalize(-ex, TARGET_Y - ey, -ez);
  // cross(forward, (0, 1, 0)), then up = cross(right, forward).
  const [rx, ry, rz] = normalize(-fz, 0, fx);
  const [lx, ly, lz] = normalize(-0.6, 0.55, 0.45);
  const spin = time * 0.7;
  return {
    width,
    height,
    ex,
    ey,
    ez,
    fx,
    fy,
    fz,
    rx,
    ry,
    rz,
    ux: ry * fz - rz * fy,
    uy: rz * fx - rx * fz,
    uz: rx * fy - ry * fx,
    focal: 1 / Math.tan((FOV * Math.PI / 180) * 0.5),
    lx,
    ly,
    lz,
    rs: Math.sin(spin),
    rc: Math.cos(spin),
    bob: 1.2 + 0.35 * Math.sin(time * 1.6),
  };
}

// ---- Scene ----------------------------------------------------------------

function normalize(x: number, y: number, z: number): [number, number, number] {
  const len = Math.sqrt(x * x + y * y + z * z);
  return [x / len, y / len, z / len];
}

const clamp01 = (x: number) => Math.min(Math.max(x, 0), 1);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

function sdRoundBox(x: number, y: number, z: number, half: number, r: number) {
  const inner = half - r;
  const qx = Math.abs(x) - inner;
  const qy = Math.abs(y) - inner;
  const qz = Math.abs(z) - inner;
  const mx = Math.max(qx, 0);
  const my = Math.max(qy, 0);
  const mz = Math.max(qz, 0);
  const outside = Math.sqrt(mx * mx + my * my + mz * mz);
  const inside = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
  return outside + inside - r;
}

function smin(a: number, b: number, k: number) {
  const h = clamp01(0.5 + (b - a) * (0.5 / k));
  return mix(b, a, h) - k * h * (1 - h);
}

function sdColumn(x: number, y: number, z: number, cx: number, cz: number) {
  const dx = x - cx;
  const dz = z - cz;
  return Math.max(Math.sqrt(dx * dx + dz * dz) - 0.24, y - 2.6);
}

// Mirror lines for the pillar folds: 22.5° and 11.25° wedges.
const C22 = 0.92387953;
const S22 = 0.38268343;
const C11 = 0.98078528;
const S11 = 0.19509032;

/** Two rings of columns from one column each, by folding space. */
function sdPillars(x: number, y: number, z: number) {
  const ax = Math.abs(x);
  const az = Math.abs(z);
  const x45 = Math.max(ax, az);
  const z45 = Math.min(ax, az);
  let d2 = Math.max(z45 * C22 - x45 * S22, 0) * 2;
  const x22 = x45 + d2 * S22;
  const z22 = z45 - d2 * C22;
  const inner = sdColumn(x22, y, z22, 6.3751044, 1.2680871);
  d2 = Math.max(z22 * C11 - x22 * S11, 0) * 2;
  const x11 = x22 + d2 * S11;
  const z11 = z22 - d2 * C11;
  const outer = sdColumn(x11, y, z11, 10.44944, 1.02918);
  return Math.min(inner, outer);
}

function sdBlob(x: number, y: number, z: number, f: Frame) {
  const by = y - f.bob;
  const ball = Math.sqrt(x * x + by * by + z * z) - 0.5;
  const bx = f.rc * x + f.rs * z;
  const bz = f.rc * z - f.rs * x;
  return smin(ball, sdRoundBox(bx, y - 0.55, bz, 0.45, 0.08), 0.35);
}

function sdRing(x: number, y: number, z: number, f: Frame) {
  const lx = x - 1.9;
  const ly = y - 0.75;
  const lz = z + 0.9;
  const tx = f.rc * lx - f.rs * lz;
  const tz = f.rs * lx + f.rc * lz;
  const q = Math.sqrt(tx * tx + ly * ly) - 0.55;
  return Math.sqrt(q * q + tz * tz) - 0.2;
}

function sdBall(x: number, y: number, z: number) {
  const bx = x + 1.7;
  const by = y - 0.5;
  const bz = z - 1;
  return Math.sqrt(bx * bx + by * by + bz * bz) - 0.5;
}

function scene(x: number, y: number, z: number, f: Frame) {
  return Math.min(
    Math.min(
      Math.min(Math.min(y, sdPillars(x, y, z)), sdBlob(x, y, z, f)),
      sdRing(x, y, z, f),
    ),
    sdBall(x, y, z),
  );
}

/** 0 ground, 1 pillars, 2 blob, 3 ring, 4 ball. */
function material(x: number, y: number, z: number, f: Frame) {
  let d = y;
  let m = 0;
  const dists = [
    sdPillars(x, y, z),
    sdBlob(x, y, z, f),
    sdRing(x, y, z, f),
    sdBall(x, y, z),
  ];
  for (let i = 0; i < 4; i++) {
    if (dists[i] < d) {
      d = dists[i];
      m = i + 1;
    }
  }
  return m;
}

/** Albedo r, g, b and specular strength per material; ground is a checker. */
const MATERIALS = [
  [0, 0, 0, 0.1],
  [0.8, 0.76, 0.7, 0.15],
  [0.9, 0.32, 0.18, 0.5],
  [0.06, 0.45, 0.52, 0.9],
  [0.85, 0.62, 0.18, 1.2],
];

function softShadow(
  x: number,
  y: number,
  z: number,
  f: Frame,
  steps: number,
) {
  let t = 0.02;
  let res = 1;
  let prev = 1e10;
  for (let i = 0; i < steps; i++) {
    const h = scene(x + f.lx * t, y + f.ly * t, z + f.lz * t, f);
    const back = h * h / (2 * prev);
    const across = Math.sqrt(Math.max(h * h - back * back, 0));
    const est = 10 * across / Math.max(t - back, 0.0001);
    if (t < 12) res = Math.min(res, est);
    prev = h;
    t = t + Math.min(Math.max(h, 0.01), 0.25);
  }
  res = clamp01(res);
  return res * res * (3 - 2 * res);
}

function occlusion(
  x: number,
  y: number,
  z: number,
  nx: number,
  ny: number,
  nz: number,
  f: Frame,
) {
  let occ = 0;
  let weight = 1;
  let h = 0.01;
  for (let i = 0; i < 5; i++) {
    const d = scene(x + nx * h, y + ny * h, z + nz * h, f);
    occ = occ + (h - d) * weight;
    weight = weight * 0.95;
    h = h + 0.03;
  }
  return clamp01(1 - 3 * occ);
}

const EPS = 0.0004;
const TMAX = 60;
const toByte = (c: number) => (c * 255 + 0.5) | 0;

/** Sky, and for rays that stopped short of `tmax`, the lit surface. */
function shade(
  f: Frame,
  q: Quality,
  dx: number,
  dy: number,
  dz: number,
  t: number,
  out: Uint8ClampedArray,
  o: number,
) {
  const skyT = clamp01(dy * 1.4 + 0.1);
  const sun = Math.max(dx * f.lx + dy * f.ly + dz * f.lz, 0);
  const s2 = sun * sun;
  const s4 = s2 * s2;
  const s8 = s4 * s4;
  const s16 = s8 * s8;
  const s64 = s16 * s16 * s16 * s16;
  const s256 = s64 * s64 * s64 * s64;
  const halo = s8 * 0.2;
  const disc = s256 * 1.5;
  const fogR = 0.78 + halo;
  const fogG = 0.8 + halo * 0.85;
  const fogB = 0.84 + halo * 0.6;
  let r = mix(fogR, 0.3, skyT) + disc;
  let g = mix(fogG, 0.46, skyT) + disc * 0.9;
  let b = mix(fogB, 0.78, skyT) + disc * 0.7;

  if (t <= TMAX) {
    const px = f.ex + dx * t;
    const py = f.ey + dy * t;
    const pz = f.ez + dz * t;

    // Normal from four tetrahedron samples.
    const e = 0.0005;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let i = 0; i < 4; i++) {
      const kx = i === 0 || i === 3 ? 1 : -1;
      const ky = i >= 2 ? 1 : -1;
      const kz = i === 1 || i === 3 ? 1 : -1;
      const d = scene(px + kx * e, py + ky * e, pz + kz * e, f);
      sx = sx + kx * d;
      sy = sy + ky * d;
      sz = sz + kz * d;
    }
    const [nx, ny, nz] = normalize(sx, sy, sz);

    let check = Math.floor(px) + Math.floor(pz);
    check = check - 2 * Math.floor(check * 0.5);
    const m = material(px, py, pz, f);
    const [ar, ag, ab, shine] = m === 0
      ? [
        mix(0.26, 0.62, check),
        mix(0.25, 0.58, check),
        mix(0.24, 0.52, check),
        0.1,
      ]
      : MATERIALS[m];

    const off = 0.002;
    const shadow = softShadow(
      px + nx * off,
      py + ny * off,
      pz + nz * off,
      f,
      q.shadowSteps,
    );
    const ao = occlusion(px, py, pz, nx, ny, nz, f);
    const diffuse = clamp01(nx * f.lx + ny * f.ly + nz * f.lz) * shadow;
    const ambient = (0.5 + 0.5 * ny) * ao;
    const bounce = clamp01(0 - ny) * ao * 0.3;

    // Blinn-Phong highlight: half vector between the light and the eye.
    const [hx, hy, hz] = normalize(f.lx - dx, f.ly - dy, f.lz - dz);
    const nh = clamp01(nx * hx + ny * hy + nz * hz);
    const n2 = nh * nh;
    const n4 = n2 * n2;
    const n16 = n4 * n4 * n4 * n4;
    const spec = n16 * n16 * n16 * n16 * shine * diffuse;

    const litR = ar * (1.4 * diffuse + 0.35 * ambient + bounce) + spec;
    const litG = ag * (1.25 * diffuse + 0.45 * ambient + bounce) + spec * 0.9;
    const litB = ab * (1.0 * diffuse + 0.6 * ambient + bounce * 0.7) +
      spec * 0.75;

    const fog = 1 - Math.exp(t * t * -0.002);
    r = mix(litR, fogR, fog);
    g = mix(litG, fogG, fog);
    b = mix(litB, fogB, fog);
  }

  out[o] = toByte(Math.sqrt(clamp01(r)));
  out[o + 1] = toByte(Math.sqrt(clamp01(g)));
  out[o + 2] = toByte(Math.sqrt(clamp01(b)));
}

// ---- Tile programs ----------------------------------------------------------

/** How a pixel's ray ended. */
export const SKY = 0;
export const HIT = 1;
/** Still marching when the budget ran out; `t <= tmax`, so it's shaded. */
export const OUT_OF_STEPS = 2;

/** One launch's results: per-sample colors and steps, per-tile counters. */
export interface Target {
  /** Image size in pixels; samples are `stride` pixels apart. */
  width: number;
  height: number;
  stride: number;
  /** Samples per tile side, and the sampled image size. */
  side: number;
  cols: number;
  rows: number;
  gridRows: number;
  gridCols: number;
  maxSteps: number;
  shaded: Uint8ClampedArray;
  steps: Uint16Array;
  ending: Uint8Array;
  tileSteps: Uint16Array;
  anyHit: Uint8Array;
  done: Uint8Array;
}

export function makeTarget(
  width: number,
  height: number,
  stride: number,
  maxSteps: number,
): Target {
  const cols = Math.ceil(width / stride);
  const rows = Math.ceil(height / stride);
  const gridRows = Math.ceil(height / TILE);
  const gridCols = Math.ceil(width / TILE);
  return {
    width,
    height,
    stride,
    side: TILE / stride,
    cols,
    rows,
    gridRows,
    gridCols,
    maxSteps,
    shaded: new Uint8ClampedArray(cols * rows * 3),
    steps: new Uint16Array(cols * rows),
    ending: new Uint8Array(cols * rows),
    tileSteps: new Uint16Array(gridRows * gridCols),
    anyHit: new Uint8Array(gridRows * gridCols),
    done: new Uint8Array(gridRows * gridCols),
  };
}

// Per-tile working state, reused across tile programs.
const T = new Float64Array(PER_TILE);
const DX = new Float64Array(PER_TILE);
const DY = new Float64Array(PER_TILE);
const DZ = new Float64Array(PER_TILE);
const STEPS = new Uint16Array(PER_TILE);
const ACTIVE = new Uint8Array(PER_TILE);

/**
 * Run tile program `(pid0, pid1)`. `checkEvery` of 0 never checks, which is
 * the crate's "no early exit" kernel.
 */
export function renderTile(
  f: Frame,
  q: Quality,
  checkEvery: number,
  target: Target,
  pid0: number,
  pid1: number,
) {
  const { side, stride, cols, rows } = target;
  const count = side * side;
  const { width, height, focal } = f;

  // A camera ray through every sample center.
  for (let i = 0; i < side; i++) {
    const row = pid0 * TILE + (i + 0.5) * stride;
    const v = (height - row * 2) / height;
    for (let j = 0; j < side; j++) {
      const col = pid1 * TILE + (j + 0.5) * stride;
      const u = (col * 2 - width) / height;
      const k = i * side + j;
      const [dx, dy, dz] = normalize(
        f.fx * focal + f.rx * u + f.ux * v,
        f.fy * focal + f.ry * u + f.uy * v,
        f.fz * focal + f.rz * u + f.uz * v,
      );
      DX[k] = dx;
      DY[k] = dy;
      DZ[k] = dz;
      T[k] = 0.05;
      STEPS[k] = 0;
      ACTIVE[k] = 1;
    }
  }

  // Sphere tracing in chunks. A finished ray is frozen; the kernel still
  // evaluates the scene for it, so skipping that here changes nothing.
  const { maxSteps } = q;
  const chunkSize = checkEvery > 0 ? checkEvery : maxSteps;
  let done = 0;
  let tileSteps = 0;
  let alive = count;
  while (done < maxSteps) {
    const chunk = Math.min(chunkSize, maxSteps - done);
    for (let s = 0; s < chunk; s++) {
      if (alive === 0) {
        tileSteps += chunk - s;
        break;
      }
      alive = 0;
      for (let k = 0; k < count; k++) {
        if (!ACTIVE[k]) continue;
        const t = T[k];
        const d = scene(
          f.ex + DX[k] * t,
          f.ey + DY[k] * t,
          f.ez + DZ[k] * t,
          f,
        );
        if (!(d >= EPS * t) || !(t <= TMAX)) {
          ACTIVE[k] = 0;
          continue;
        }
        T[k] = t + d;
        STEPS[k]++;
        alive++;
      }
      tileSteps++;
    }
    done += chunk;
    if (alive === 0) break;
  }

  // The scalar `if`: a tile with no hits skips all surface shading.
  let anyHit = 0;
  for (let k = 0; k < count; k++) {
    if (T[k] <= TMAX) {
      anyHit = 1;
      break;
    }
  }

  for (let i = 0; i < side; i++) {
    const r = pid0 * side + i;
    if (r >= rows) break;
    for (let j = 0; j < side; j++) {
      const c = pid1 * side + j;
      if (c >= cols) break;
      const k = i * side + j;
      const o = r * cols + c;
      shade(f, q, DX[k], DY[k], DZ[k], T[k], target.shaded, o * 3);
      target.steps[o] = STEPS[k];
      target.ending[o] = T[k] > TMAX ? SKY : ACTIVE[k] ? OUT_OF_STEPS : HIT;
    }
  }

  const tile = pid0 * target.gridCols + pid1;
  target.tileSteps[tile] = tileSteps;
  target.anyHit[tile] = anyHit;
  target.done[tile] = 1;
}

/** Black, red, yellow, white: the kernel's `heat` for the debug views. */
function heat(v: number, out: Uint8ClampedArray, o: number) {
  const v3 = clamp01(v) * 3;
  out[o] = toByte(clamp01(v3));
  out[o + 1] = toByte(clamp01(v3 - 1));
  out[o + 2] = toByte(clamp01(v3 - 2));
}

/**
 * Paint one tile's pixels into a full-size RGBA buffer, from `full` if that
 * tile program has run, otherwise upscaled from `preview`.
 */
export function paintTile(
  rgba: Uint8ClampedArray,
  full: Target,
  preview: Target | null,
  view: View,
  pid0: number,
  pid1: number,
) {
  const tile = pid0 * full.gridCols + pid1;
  const source = full.done[tile] ? full : preview;
  if (!source) return;
  const { width, height } = full;
  const { stride, cols, maxSteps } = source;
  const inv = 1 / maxSteps;
  const y1 = Math.min((pid0 + 1) * TILE, height);
  const x1 = Math.min((pid1 + 1) * TILE, width);
  for (let y = pid0 * TILE; y < y1; y++) {
    const sr = Math.floor(y / stride) * cols;
    for (let x = pid1 * TILE; x < x1; x++) {
      const s = sr + Math.floor(x / stride);
      const o = (y * width + x) * 4;
      if (view === "shaded") {
        rgba[o] = source.shaded[s * 3];
        rgba[o + 1] = source.shaded[s * 3 + 1];
        rgba[o + 2] = source.shaded[s * 3 + 2];
      } else if (view === "steps") {
        heat(source.steps[s] * inv, rgba, o);
      } else {
        heat(source.tileSteps[tile] * inv, rgba, o);
      }
      rgba[o + 3] = 255;
    }
  }
}
