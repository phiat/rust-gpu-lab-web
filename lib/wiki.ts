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

export async function listNotes(): Promise<NoteMeta[]> {
  const notes: NoteMeta[] = [];
  for await (const entry of Deno.readDir(CONTENT_DIR)) {
    if (!entry.isFile || !entry.name.endsWith(".md")) continue;
    const note = await readNote(entry.name.slice(0, -3));
    if (note) {
      const { body: _body, ...meta } = note;
      notes.push(meta);
    }
  }
  return notes.sort((a, b) =>
    a.order - b.order || a.title.localeCompare(b.title)
  );
}

export function getNote(slug: string): Promise<Note | null> {
  if (!/^[\w-]+$/.test(slug)) return Promise.resolve(null);
  return readNote(slug);
}
