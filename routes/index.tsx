import { Head } from "fresh/runtime";
import { define } from "../utils.ts";
import { listNotes } from "../lib/wiki.ts";
import {
  listRenders,
  loadGit,
  loadWorkspace,
  probeToolchain,
  timeAgo,
} from "../lib/tileworld.ts";
import HeroRender from "../islands/HeroRender.tsx";

export default define.page(async function Home() {
  const [notes, ws, renders, git] = await Promise.all([
    listNotes(),
    loadWorkspace(),
    listRenders(),
    loadGit(1),
  ]);
  const tools = await probeToolchain(ws.cudaToolkitPath);
  const path = notes.filter((n) => n.section !== "Reference");
  const reference = notes.filter((n) => n.section === "Reference");
  const latestRender = renders.toSorted((a, b) => b.modified - a.modified)[0];
  const lastCommit = git?.commits[0];
  const sources = ws.crates.flatMap((c) => c.sources);
  const lastEdit = sources.sort((a, b) => b.modified - a.modified)[0];
  const gpu = tools.gpus[0];

  return (
    <>
      <Head>
        <title>tileworld</title>
      </Head>

      <section class="hero">
        <div class="hero-copy">
          <h1>
            Write the kernel for one tile. The GPU runs it for all of them.
          </h1>
          <p class="lede">
            tileworld is a workspace for learning{" "}
            <a href="https://github.com/NVlabs/cutile-rs">cuTile Rust</a>: GPU
            kernels written as ordinary, ownership-checked Rust. The notes
            explain the model, the playground lets you try it, and the project
            page shows the code as it grows.
          </p>
          <div class="actions">
            <a class="button primary" href="/wiki/getting-started">
              Start with the basics
            </a>
            <a class="button" href="/playground">Open the playground</a>
          </div>
        </div>
        <HeroRender />
      </section>

      <div class="home-columns">
        <section class="reading-order" aria-labelledby="reading-order">
          <h2 id="reading-order">Reading order</h2>
          <ol>
            {path.map((note) => (
              <li>
                <a href={`/wiki/${note.slug}`}>{note.title}</a>
                <p>{note.summary}</p>
              </li>
            ))}
          </ol>
          {reference.length > 0 && (
            <p class="also">
              For lookup: {reference.map((note, i) => (
                <>
                  {i > 0 && " and "}
                  <a href={`/wiki/${note.slug}`}>{note.title.toLowerCase()}</a>
                </>
              ))}. Press <kbd>/</kbd> to search every note.
            </p>
          )}
        </section>

        <aside class="glance" aria-labelledby="glance">
          <h2 id="glance">In the workspace</h2>
          {latestRender && (
            <a class="glance-render" href="/project#renders">
              <img
                src={`/project/renders/${latestRender.path}?v=${latestRender.modified}`}
                alt={`Latest render, ${latestRender.path}`}
                loading="lazy"
              />
              <span>
                Latest render: <code>{latestRender.path}</code>,{" "}
                {timeAgo(latestRender.modified)}
              </span>
            </a>
          )}
          {ws.found
            ? (
              <dl class="facts">
                {ws.crates.map((crate) => (
                  <div>
                    <dt>
                      <code>{crate.name}</code>
                    </dt>
                    <dd>
                      {crate.builtProfiles.length
                        ? `Built (${crate.builtProfiles.join(", ")})`
                        : "Not built yet"}, {crate.sources.length}{" "}
                      file{crate.sources.length === 1 ? "" : "s"}
                    </dd>
                  </div>
                ))}
                {lastCommit && (
                  <div>
                    <dt>Last commit</dt>
                    <dd>
                      <a href="/project#history">{lastCommit.subject}</a>,{" "}
                      {timeAgo(lastCommit.time)}
                      {git && git.changes.length > 0 &&
                        `. ${git.changes.length} uncommitted change${
                          git.changes.length === 1 ? "" : "s"
                        }.`}
                    </dd>
                  </div>
                )}
                {!lastCommit && lastEdit && (
                  <div>
                    <dt>Last edit</dt>
                    <dd>
                      <code>{lastEdit.path.split("/").pop()}</code>,{" "}
                      {timeAgo(lastEdit.modified)}
                    </dd>
                  </div>
                )}
                <div>
                  <dt>GPU</dt>
                  <dd>
                    {gpu
                      ? `${
                        gpu.name.replace(/^NVIDIA (GeForce )?/, "")
                      }, compute ${gpu.computeCap}`
                      : "Not detected"}
                  </dd>
                </div>
              </dl>
            )
            : (
              <p class="empty">
                No workspace found. Set <code>TILEWORLD_DIR</code>{" "}
                to the tileworld checkout and restart the server.
              </p>
            )}
          <a class="more" href="/project">See the full project page</a>
        </aside>
      </div>
    </>
  );
});
