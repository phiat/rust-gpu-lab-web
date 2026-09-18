# Ideas

Things the site could do next, from a review on 2026-09-17 (17 notes, 8 project
write-ups, four playground demos). Roughly in order of value within each group.
Strike an item when it lands.

## Features

1. **Full-text search.** `SearchPalette` matches titles, summaries and headings
   only, so "lowbias32", "Margolus" or "pinned" find nothing or the wrong note.
   `/api/search` already loads note bodies: index paragraphs (split on blank
   lines, strip markdown), return a snippet with the match highlighted. The one
   change that alters how the site gets used.

2. **A falling sand demo in the playground.** `sand`'s rules are integer math
   with a hash RNG and its CPU reference is 300 lines; `LifeStencil` already has
   the pattern (padded buffer, per-tile-program stepping, hover to see what a
   program reads). The new teaching moment is the Margolus overlay: shift the
   2×2 block grid on alternate passes and watch a grain fall through. Cloth
   would need a rasterizer and its lesson (batch coloring) is hard to see; sand
   is the better demo.

3. **"What's new" on the home page.** The home page shows the last workspace
   commit but not what the site says about it. A strip of the latest Project
   notes (highest `order`, or a `date` field in front matter) would show a
   returning reader what changed.

4. **A cross-demo numbers note.** Every project note has its own Results table;
   the interesting story runs across them: which demos gain from CUDA graphs
   (life 5×, cloth 3.3×, raymarch 1.01×) and why, transfer share per frame,
   compile times. `cuda-graphs.md#when-it-pays-off` is halfway there. Could be
   generated from front matter.

## Polish

5. **Image dimensions.** Note images have no `width`/`height`, so the three
   screenshots shift the layout on load. The markdown renderer can read PNG
   dimensions from `static/images` at render time (8 bytes at a fixed offset)
   and emit them.

6. **Long lists.** The wiki sidebar has eight Project entries in a flat list and
   the home "Reading order" runs 15 items in one column. Either collapse sidebar
   sections with `<details open>`, or lay the home reading order out in two
   columns (Start here + Concepts, then Projects). The second is CSS only.

7. **Open Graph tags.** A pasted note link shows no preview. `[slug].tsx` has
   `note.title` and `note.summary` in hand; four meta tags in `<Head>`.

8. **h3 in the table of contents.** `headingsOf` collects h2 and h3 but the "On
   this page" aside keeps h2 only. No note uses h3 today; if one does, the
   headings vanish silently. Render h3 indented, or drop them from `headingsOf`.

9. **Bind to localhost by default.** `deno task start` listens on 0.0.0.0 and
   the project page runs `git` in a local directory. Default the task to
   `--host 127.0.0.1` and let the README say how to open it up.

10. **`robots` and a sitemap.** Only matters if the site goes public.

## Considered and skipped

- **Lazy-loading the project page's highlighted source.** Measured at 45 ms on
  localhost; lazy loading would break find-in-page for no visible gain.
- **A cloth demo in the browser.** See item 2.
