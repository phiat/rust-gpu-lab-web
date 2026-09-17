import { define } from "../../utils.ts";
import { loadNotes } from "../../lib/wiki.ts";
import { headingsOf } from "../../lib/markdown.ts";

export interface SearchEntry {
  slug: string;
  title: string;
  summary: string;
  section: string;
  headings: { text: string; id: string }[];
}

export const handler = define.handlers({
  async GET() {
    const entries: SearchEntry[] = (await loadNotes()).map((note) => ({
      slug: note.slug,
      title: note.title,
      summary: note.summary,
      section: note.section,
      headings: headingsOf(note.body).map(({ text, id }) => ({ text, id })),
    }));
    return Response.json(entries);
  },
});
