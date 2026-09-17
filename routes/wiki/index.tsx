import { Head } from "fresh/runtime";
import { define } from "../../utils.ts";
import { listNotes } from "../../lib/wiki.ts";
import { groupBySection } from "../../components/WikiNav.tsx";

export default define.page(async function WikiIndex() {
  const notes = await listNotes();

  return (
    <div class="page">
      <Head>
        <title>Wiki · tileworld</title>
      </Head>
      <header class="page-header">
        <h1>Wiki</h1>
        <p class="lede">
          Short notes on cuTile Rust, checked against the book at the pinned
          {" "}
          <code>d92c160</code> rev. Press <kbd>/</kbd> to search them.
        </p>
      </header>

      <div class="note-index">
        {groupBySection(notes).map(([section, items]) => (
          <section>
            <h2>{section}</h2>
            <ul>
              {items.map((note) => (
                <li>
                  <a href={`/wiki/${note.slug}`}>{note.title}</a>
                  <p>{note.summary}</p>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
});
