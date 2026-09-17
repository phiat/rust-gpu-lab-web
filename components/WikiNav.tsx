import type { NoteMeta } from "../lib/wiki.ts";

export function groupBySection(notes: NoteMeta[]): [string, NoteMeta[]][] {
  const groups = new Map<string, NoteMeta[]>();
  for (const note of notes) {
    const list = groups.get(note.section) ?? [];
    list.push(note);
    groups.set(note.section, list);
  }
  return [...groups];
}

export function WikiNav(
  { notes, current }: { notes: NoteMeta[]; current?: string },
) {
  return (
    <nav class="wiki-nav" aria-label="Notes">
      {groupBySection(notes).map(([section, items]) => (
        <div>
          <h2>{section}</h2>
          <ul>
            {items.map((note) => (
              <li>
                <a
                  href={`/wiki/${note.slug}`}
                  aria-current={note.slug === current ? "page" : undefined}
                >
                  {note.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}
