import { extract } from "@std/front-matter/yaml";
import { join } from "@std/path";

/** Notes are plain markdown files; edit or add one and refresh. */
export const CONTENT_DIR = join(Deno.cwd(), "content");

export interface NoteMeta {
  slug: string;
  title: string;
  summary: string;
  order: number;
  section: string;
}

export interface Note extends NoteMeta {
  body: string;
}

interface FrontMatter {
  title?: string;
  summary?: string;
  order?: number;
  section?: string;
}

async function readNote(slug: string): Promise<Note | null> {
  let raw: string;
  try {
    raw = await Deno.readTextFile(join(CONTENT_DIR, `${slug}.md`));
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return null;
    throw err;
  }
  const { attrs, body } = extract<FrontMatter>(raw);
  return {
    slug,
    title: attrs.title ?? slug,
    summary: attrs.summary ?? "",
    order: attrs.order ?? 999,
    section: attrs.section ?? "Notes",
    body,
  };
}

/** Every note with its body, in reading order. */
export async function loadNotes(): Promise<Note[]> {
  const entries = await Array.fromAsync(Deno.readDir(CONTENT_DIR));
  const notes = await Promise.all(
    entries
      .filter((entry) => entry.isFile && entry.name.endsWith(".md"))
      .map((entry) => readNote(entry.name.slice(0, -3))),
  );
  return notes
    .filter((note) => note !== null)
    .sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
}

/** Every note's front matter, in reading order. */
export async function listNotes(): Promise<NoteMeta[]> {
  return (await loadNotes()).map(({ body: _body, ...meta }) => meta);
}

export function getNote(slug: string): Promise<Note | null> {
  if (!/^[\w-]+$/.test(slug)) return Promise.resolve(null);
  return readNote(slug);
}
