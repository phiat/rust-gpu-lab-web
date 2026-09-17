import { HttpError } from "fresh";
import { Head } from "fresh/runtime";
import { define } from "../../utils.ts";
import { getNote, listNotes } from "../../lib/wiki.ts";
import { renderMarkdown } from "../../lib/markdown.ts";
import { WikiNav } from "../../components/WikiNav.tsx";

export default define.page(async function NotePage(ctx) {
  const [note, notes] = await Promise.all([
    getNote(ctx.params.slug),
    listNotes(),
  ]);
  if (!note) throw new HttpError(404);

  const { html, headings } = renderMarkdown(note.body);
  const index = notes.findIndex((n) => n.slug === note.slug);
  const prev = notes[index - 1];
  const next = notes[index + 1];
  const toc = headings.filter((h) => h.depth === 2);

  return (
    <div class="wiki-layout">
      <Head>
        <title>{`${note.title} · tileworld`}</title>
      </Head>

      <WikiNav notes={notes} current={note.slug} />

      <article class="note">
        <nav class="breadcrumb" aria-label="Breadcrumb">
          <a href="/wiki">Wiki</a>
          <span aria-hidden="true">/</span>
          <span>{note.section}</span>
        </nav>
        <h1>{note.title}</h1>
        {note.summary && <p class="lede">{note.summary}</p>}
        <div
          class="prose"
          // Notes are local markdown files, so their HTML is trusted.
          // deno-lint-ignore react-no-danger
          dangerouslySetInnerHTML={{ __html: html }}
        />

        <nav class="pager" aria-label="Previous and next notes">
          {prev
            ? (
              <a href={`/wiki/${prev.slug}`} rel="prev">
                <small>Previous</small>
                {prev.title}
              </a>
            )
            : <span />}
          {next && (
            <a href={`/wiki/${next.slug}`} rel="next" class="next">
              <small>Next</small>
              {next.title}
            </a>
          )}
        </nav>
        <p class="source-path">
          Edit this note in <code>content/{note.slug}.md</code>
        </p>
      </article>

      {toc.length > 1 && (
        <aside class="toc" aria-labelledby="toc-heading">
          <h2 id="toc-heading">On this page</h2>
          <ul>
            {toc.map((h) => (
              <li>
                <a href={`#${h.id}`}>{h.text}</a>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
});
