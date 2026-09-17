import { define } from "../../utils.ts";
import { getNote, listNotes } from "../../lib/wiki.ts";
import { renderMarkdown } from "../../lib/markdown.ts";

export interface SearchEntry {
  slug: string;
  title: string;
  summary: string;
  section: string;
  headings: { text: string; id: string }[];
}

export const handler = define.handlers({
  async GET() {
    const notes = await listNotes();
    const entries: SearchEntry[] = [];
    for (const meta of notes) {
      const note = await getNote(meta.slug);
      if (!note) continue;
      const { headings } = renderMarkdown(note.body);
      entries.push({
        slug: meta.slug,
        title: meta.title,
        summary: meta.summary,
        section: meta.section,
        headings: headings.map(({ text, id }) => ({ text, id })),
      });
    }
    return Response.json(entries);
  },
});
