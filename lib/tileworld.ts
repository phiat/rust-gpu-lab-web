import { join, relative, resolve } from "@std/path";
import { parse as parseToml } from "@std/toml";

/**
 * Where the Rust workspace lives. Defaults to the sibling checkout this wiki
 * was written against; override with TILEWORLD_DIR.
 */
export const TILEWORLD_DIR = resolve(
  Deno.env.get("TILEWORLD_DIR") ?? join(Deno.cwd(), "../../tileworld"),
);

export interface SourceFile {
  path: string;
  content: string;
  lines: number;
  modified: number;
}

export interface Crate {
  name: string;
  dir: string;
  manifest: string;
  /** The crate's README.md, if it has one. */
  readme?: string;
  /** First paragraph of the README, as plain markdown. */
  description?: string;
  dependencies: string[];
  sources: SourceFile[];
  builtProfiles: string[];
}

export interface Dependency {
  name: string;
  spec: string;
}

export interface Workspace {
  root: string;
  found: boolean;
  manifest: string;
  rustVersion?: string;
  edition?: string;
  crates: Crate[];
  dependencies: Dependency[];
  cudaToolkitPath?: string;
  cudaToolkitExists: boolean;
}

export interface Render {
  /** Path relative to the workspace root, e.g. `docs/images/life.png`. */
  path: string;
  /** Containing folder relative to the root; "" for the root itself. */
  dir: string;
  name: string;
  bytes: number;
  modified: number;
}

export interface Commit {
  hash: string;
  time: number;
  subject: string;
  /** Body paragraphs, reflowed, with trailers such as Co-Authored-By removed. */
  body: string[];
  filesChanged: number;
  insertions: number;
  deletions: number;
}

export interface GitInfo {
  branch: string;
  commits: Commit[];
  /** `git status --porcelain` lines: uncommitted, untracked or deleted files. */
  changes: string[];
}

export interface Toolchain {
  gpus: { name: string; computeCap: string; driver: string }[];
  rustc?: string;
  nvcc?: string;
}

async function readText(path: string): Promise<string | undefined> {
  try {
    return await Deno.readTextFile(path);
  } catch {
    return undefined;
  }
}

async function exists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

async function collectSources(root: string, dir: string, out: SourceFile[]) {
  let entries: Deno.DirEntry[];
  try {
    entries = await Array.fromAsync(Deno.readDir(dir));
  } catch {
    return;
  }
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory) {
      await collectSources(root, path, out);
    } else if (entry.name.endsWith(".rs")) {
      const content = (await readText(path)) ?? "";
      const info = await Deno.stat(path).catch(() => null);
      out.push({
        path: relative(root, path),
        content,
        lines: content.split("\n").length,
        modified: info?.mtime?.getTime() ?? 0,
      });
    }
  }
}

function describeSpec(spec: unknown): string {
  if (typeof spec === "string") return spec;
  if (spec && typeof spec === "object") {
    const s = spec as Record<string, unknown>;
    if (s.git) return `git ${s.git}${s.rev ? ` @ ${s.rev}` : ""}`;
    if (s.version) {
      const features = Array.isArray(s.features) && s.features.length
        ? ` [${s.features.join(", ")}]`
        : "";
      return `${s.version}${features}`;
    }
    if (s.workspace) return "workspace";
    if (s.path) return `path ${s.path}`;
  }
  return JSON.stringify(spec);
}

export async function loadWorkspace(): Promise<Workspace> {
  const root = TILEWORLD_DIR;
  const manifest = await readText(join(root, "Cargo.toml"));
  if (manifest === undefined) {
    return {
      root,
      found: false,
      manifest: "",
      crates: [],
      dependencies: [],
      cudaToolkitExists: false,
    };
  }

  const parsed = parseToml(manifest) as {
    workspace?: {
      members?: string[];
      package?: { edition?: string; "rust-version"?: string };
      dependencies?: Record<string, unknown>;
    };
  };
  const ws = parsed.workspace ?? {};

  const crates: Crate[] = [];
  for (const member of ws.members ?? []) {
    const dir = join(root, member);
    const crateManifest = (await readText(join(dir, "Cargo.toml"))) ?? "";
    const crateToml = crateManifest
      ? parseToml(crateManifest) as {
        package?: { name?: string };
        dependencies?: Record<string, unknown>;
      }
      : {};
    const sources: SourceFile[] = [];
    await collectSources(root, join(dir, "src"), sources);
    const name = crateToml.package?.name ?? member;
    const readme = await readText(join(dir, "README.md"));
    const builtProfiles: string[] = [];
    for (const profile of ["debug", "release"]) {
      if (await exists(join(root, "target", profile, name))) {
        builtProfiles.push(profile);
      }
    }
    crates.push({
      name,
      dir: member,
      manifest: crateManifest,
      readme,
      description: readme ? firstParagraph(readme) : undefined,
      dependencies: Object.keys(crateToml.dependencies ?? {}),
      sources,
      builtProfiles,
    });
  }

  const cargoConfig = await readText(join(root, ".cargo", "config.toml"));
  let cudaToolkitPath: string | undefined;
  if (cargoConfig) {
    const env = (parseToml(cargoConfig) as { env?: Record<string, unknown> })
      .env;
    const v = env?.CUDA_TOOLKIT_PATH;
    cudaToolkitPath = typeof v === "string"
      ? v
      : (v as { value?: string } | undefined)?.value;
  }

  return {
    root,
    found: true,
    manifest,
    rustVersion: ws.package?.["rust-version"],
    edition: ws.package?.edition,
    crates,
    dependencies: Object.entries(ws.dependencies ?? {}).map((
      [name, spec],
    ) => ({ name, spec: describeSpec(spec) })),
    cudaToolkitPath,
    cudaToolkitExists: cudaToolkitPath ? await exists(cudaToolkitPath) : false,
  };
}

/** The first prose paragraph of a markdown document, skipping the title. */
export function firstParagraph(markdown: string): string | undefined {
  return markdown
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .find((block) => block && !/^(#|```|\||- |\d+\. |>)/.test(block))
    ?.replace(/\s*\n\s*/g, " ");
}

/** "4 minutes ago", for file listings. */
export function timeAgo(ms: number, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - ms) / 1000));
  const units: [number, string][] = [
    [86400, "day"],
    [3600, "hour"],
    [60, "minute"],
  ];
  for (const [size, name] of units) {
    const n = Math.floor(s / size);
    if (n >= 1) return `${n} ${name}${n === 1 ? "" : "s"} ago`;
  }
  return "just now";
}

const SEGMENT = /^[\w-][\w.-]*$/;
const SKIP_DIRS = new Set(["target", "src", "node_modules"]);
const IMAGE_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
};

/** The content type for a render's file name, or undefined if it isn't one. */
export function imageType(name: string): string | undefined {
  return IMAGE_TYPES[name.slice(name.lastIndexOf(".") + 1).toLowerCase()];
}

/** Output order for known pipeline stages; anything else sorts after. */
const STAGES = ["input", "gray", "blurred", "edges"];

function stageRank(name: string): number {
  const i = STAGES.indexOf(name.replace(/\.\w+$/, ""));
  return i === -1 ? STAGES.length : i;
}

const isFolder = (entry: Deno.DirEntry) =>
  entry.isDirectory && !entry.name.startsWith(".") &&
  !SKIP_DIRS.has(entry.name) && SEGMENT.test(entry.name);

/** Images directly in `dir`, then in its subfolders down to `depth` levels. */
async function imagesIn(
  dir: string,
  rel: string,
  depth: number,
  out: Render[],
) {
  for await (const entry of Deno.readDir(dir)) {
    if (isFolder(entry) && depth > 0) {
      const sub = rel ? `${rel}/${entry.name}` : entry.name;
      await imagesIn(join(dir, entry.name), sub, depth - 1, out);
      continue;
    }
    if (!entry.isFile || !imageType(entry.name)) continue;
    if (!SEGMENT.test(entry.name)) continue;
    const info = await Deno.stat(join(dir, entry.name));
    out.push({
      path: rel ? `${rel}/${entry.name}` : entry.name,
      dir: rel,
      name: entry.name,
      bytes: info.size,
      modified: info.mtime?.getTime() ?? 0,
    });
  }
}

/**
 * PNG and JPEG images in the workspace root and up to two folders deep: output
 * folders such as `filters-out/` (gitignored, so local only) and the README
 * screenshots in `docs/images/`. Root renders come newest first; files inside a
 * folder keep pipeline order.
 */
export async function listRenders(): Promise<Render[]> {
  const renders: Render[] = [];
  try {
    await imagesIn(TILEWORLD_DIR, "", 2, renders);
  } catch {
    // Workspace missing: no renders.
  }
  return renders.sort((a, b) =>
    a.dir === b.dir
      ? a.dir
        ? stageRank(a.name) - stageRank(b.name) || a.name.localeCompare(b.name)
        : b.modified - a.modified
      : a.dir.localeCompare(b.dir)
  );
}

/** Renders grouped by folder, root first, each group newest folder first. */
export function groupRenders(renders: Render[]): [string, Render[]][] {
  const groups = new Map<string, Render[]>();
  for (const r of renders) {
    groups.set(r.dir, [...(groups.get(r.dir) ?? []), r]);
  }
  const newest = (rs: Render[]) => Math.max(...rs.map((r) => r.modified));
  return [...groups].sort(([a, ra], [b, rb]) =>
    a === "" ? -1 : b === "" ? 1 : newest(rb) - newest(ra)
  );
}

/** Resolve a render path from a URL, allowing at most two folder levels. */
export function renderPath(path: string): string | null {
  const parts = path.split("/");
  if (parts.length > 3 || !parts.every((p) => SEGMENT.test(p))) return null;
  if (!imageType(path) || parts.some((p) => SKIP_DIRS.has(p))) return null;
  return join(TILEWORLD_DIR, ...parts);
}

async function run(cmd: string, args: string[]): Promise<string | undefined> {
  try {
    const { success, stdout } = await new Deno.Command(cmd, {
      args,
      stdout: "piped",
      stderr: "null",
    }).output();
    return success ? new TextDecoder().decode(stdout).trim() : undefined;
  } catch {
    return undefined;
  }
}

const TRAILER = /^[A-Za-z][\w-]*: .+$/;

function parseCommit(record: string): Commit | undefined {
  const [hash, time, subject, rest = ""] = record.split("\x1f");
  if (!hash || !subject) return undefined;
  const stat = rest.match(
    /(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?\s*$/,
  );
  const paragraphs = (stat ? rest.slice(0, stat.index) : rest)
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean)
    .filter((p) => !p.split("\n").every((line) => TRAILER.test(line)));
  return {
    hash,
    time: Number(time) * 1000,
    subject,
    body: paragraphs.map((p) => p.replace(/\s*\n\s*/g, " ")),
    filesChanged: Number(stat?.[1] ?? 0),
    insertions: Number(stat?.[2] ?? 0),
    deletions: Number(stat?.[3] ?? 0),
  };
}

/** Branch, recent commits and uncommitted changes, or undefined without git. */
export async function loadGit(limit = 10): Promise<GitInfo | undefined> {
  const git = (...args: string[]) => run("git", ["-C", TILEWORLD_DIR, ...args]);
  const [branch, log, status] = await Promise.all([
    git("rev-parse", "--abbrev-ref", "HEAD"),
    git(
      "log",
      `-n${limit}`,
      "--format=%x1e%h%x1f%at%x1f%s%x1f%b",
      "--shortstat",
    ),
    git("status", "--porcelain"),
  ]);
  if (branch === undefined || log === undefined) return undefined;
  return {
    branch,
    commits: log.split("\x1e").map(parseCommit).filter((c): c is Commit => !!c),
    changes: (status ?? "").split("\n").filter(Boolean),
  };
}

let toolchainCache: Promise<Toolchain> | undefined;

/** Probe the local GPU and compilers once per server process. */
export function probeToolchain(cudaToolkitPath?: string): Promise<Toolchain> {
  toolchainCache ??= (async () => {
    const smi = await run("nvidia-smi", [
      "--query-gpu=name,compute_cap,driver_version",
      "--format=csv,noheader",
    ]);
    const gpus = (smi ?? "").split("\n").filter(Boolean).map((line) => {
      const [name, computeCap, driver] = line.split(",").map((s) => s.trim());
      return { name, computeCap, driver };
    });
    const nvccPath = cudaToolkitPath
      ? join(cudaToolkitPath, "bin", "nvcc")
      : "nvcc";
    const nvcc = (await run(nvccPath, ["--version"]))
      ?.split("\n")
      .find((l) => l.includes("release"))
      ?.trim();
    return { gpus, rustc: await run("rustc", ["--version"]), nvcc };
  })();
  return toolchainCache;
}
