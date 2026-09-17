import { Head } from "fresh/runtime";
import { define } from "../utils.ts";
import {
  groupRenders,
  listRenders,
  loadGit,
  loadWorkspace,
  probeToolchain,
  timeAgo,
} from "../lib/tileworld.ts";
import { codeBlock, renderMarkdown } from "../lib/markdown.ts";
import { listNotes } from "../lib/wiki.ts";

/** Minimum CUDA toolkit per compute capability, from the cutile README. */
function cudaRequirement(computeCap: string): string {
  const [major, minor] = computeCap.split(".").map(Number);
  if (major < 8) return "unsupported (below sm_80)";
  if (major === 8) return "needs CUDA 13.2 or newer";
  if (major === 9 && minor === 0) return "needs CUDA 13.3 or newer";
  if (major >= 10) return "needs CUDA 13.2 or newer";
  return "check the cutile README";
}

export default define.page(async function Project() {
  const [ws, renders, notes, git] = await Promise.all([
    loadWorkspace(),
    listRenders(),
    listNotes(),
    loadGit(),
  ]);
  const noteSlugs = new Set(notes.map((n) => n.slug));
  const tools = await probeToolchain(ws.cudaToolkitPath);
  const recent = ws.crates
    .flatMap((c) => c.sources)
    .sort((a, b) => b.modified - a.modified)
    .slice(0, 5);

  return (
    <div class="page">
      <Head>
        <title>Project · tileworld</title>
      </Head>
      <header class="page-header">
        <h1>The workspace</h1>
        <p class="lede">
          Read from <code>{ws.root}</code> each time this page loads. Set{" "}
          <code>TILEWORLD_DIR</code> to read a different checkout.
        </p>
      </header>

      {!ws.found
        ? (
          <p class="callout">
            There's no <code>Cargo.toml</code> at <code>{ws.root}</code>. Point
            {" "}
            <code>TILEWORLD_DIR</code>{" "}
            at the tileworld workspace and restart the server.
          </p>
        )
        : (
          <>
            <dl class="toolchain">
              <div data-ok={String(Boolean(tools.rustc))}>
                <dt>Rust</dt>
                <dd>
                  <strong>{ws.rustVersion ?? "Not pinned"}</strong>
                  <span>
                    Edition {ws.edition ?? "?"}
                    {tools.rustc && `, host rustc ${tools.rustc.split(" ")[1]}`}
                  </span>
                </dd>
              </div>
              <div data-ok={String(ws.cudaToolkitExists)}>
                <dt>CUDA toolkit</dt>
                <dd>
                  <strong>{ws.cudaToolkitPath ?? "Not set"}</strong>
                  <span>
                    {tools.nvcc?.replace(/^Cuda compilation tools, /, "") ??
                      (ws.cudaToolkitExists
                        ? "Found"
                        : "That path doesn't exist on this machine")}
                  </span>
                </dd>
              </div>
              {tools.gpus.length
                ? tools.gpus.map((gpu) => (
                  <div data-ok={String(Number(gpu.computeCap) >= 8)}>
                    <dt>GPU</dt>
                    <dd>
                      <strong>{gpu.name}</strong>
                      <span>
                        Compute {gpu.computeCap},{" "}
                        {cudaRequirement(gpu.computeCap)}. Driver {gpu.driver}.
                      </span>
                    </dd>
                  </div>
                ))
                : (
                  <div data-ok="false">
                    <dt>GPU</dt>
                    <dd>
                      <strong>Not detected</strong>
                      <span>nvidia-smi didn't respond.</span>
                    </dd>
                  </div>
                )}
            </dl>

            {git && (
              <section id="history" aria-labelledby="history-heading">
                <h2 id="history-heading">History</h2>
                <p class="history-status">
                  On <code>{git.branch}</code>. {git.changes.length === 0
                    ? "Nothing uncommitted."
                    : `${git.changes.length} uncommitted change${
                      git.changes.length === 1 ? "" : "s"
                    }:`}
                </p>
                {git.changes.length > 0 && (
                  <ul class="changes">
                    {git.changes.map((line) => (
                      <li>
                        <code>{line.trim()}</code>
                      </li>
                    ))}
                  </ul>
                )}
                <ol class="commits">
                  {git.commits.map((commit, i) => (
                    <li>
                      <details open={i === 0}>
                        <summary>
                          <span class="commit-subject">{commit.subject}</span>
                          <span class="commit-meta">
                            <code>{commit.hash}</code> {timeAgo(commit.time)}
                            {commit.filesChanged > 0 &&
                              `, ${commit.filesChanged} file${
                                commit.filesChanged === 1 ? "" : "s"
                              }, +${commit.insertions} −${commit.deletions}`}
                          </span>
                        </summary>
                        {commit.body.map((paragraph, i) => (
                          <p key={i}>{paragraph}</p>
                        ))}
                      </details>
                    </li>
                  ))}
                </ol>
              </section>
            )}

            {renders.length > 0 && (
              <section id="renders" aria-labelledby="renders-heading">
                <h2 id="renders-heading">Renders</h2>
                {groupRenders(renders).map(([dir, group]) => (
                  <div class="render-group">
                    {dir && (
                      <h3>
                        <code>{dir}/</code>
                        <span>
                          {group.length}{" "}
                          image{group.length === 1 ? "" : "s"}, written{" "}
                          {timeAgo(Math.max(...group.map((r) => r.modified)))}
                        </span>
                      </h3>
                    )}
                    <div class={dir ? "renders stages" : "renders"}>
                      {group.map((r) => (
                        <figure>
                          <a
                            href={`/project/renders/${r.path}?v=${r.modified}`}
                          >
                            <img
                              src={`/project/renders/${r.path}?v=${r.modified}`}
                              alt={`Render ${r.path}`}
                              loading="lazy"
                            />
                          </a>
                          <figcaption>
                            <code>{r.name}</code>
                            <span>
                              {(r.bytes / 1024 / 1024).toFixed(1)} MB
                              {!dir && `, ${timeAgo(r.modified)}`}
                            </span>
                          </figcaption>
                        </figure>
                      ))}
                    </div>
                  </div>
                ))}
              </section>
            )}

            {!git && recent.length > 0 && (
              <section aria-labelledby="recent-heading">
                <h2 id="recent-heading">Recently changed</h2>
                <ul class="recent">
                  {recent.map((src) => (
                    <li>
                      <a href={`#${src.path}`}>
                        <code>{src.path}</code>
                      </a>
                      <span>{timeAgo(src.modified)}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <section aria-labelledby="deps-heading">
              <h2 id="deps-heading">Dependencies</h2>
              <div class="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Crate</th>
                      <th scope="col">Version</th>
                      <th scope="col">Used by</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ws.dependencies.map((dep) => (
                      <tr>
                        <td>
                          <code>{dep.name}</code>
                        </td>
                        <td class="spec">{dep.spec}</td>
                        <td>
                          {ws.crates
                            .filter((c) => c.dependencies.includes(dep.name))
                            .map((c) => c.name)
                            .join(", ") || "Nothing yet"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {ws.crates.map((crate) => (
              <section class="crate" aria-labelledby={`crate-${crate.name}`}>
                <div class="crate-head">
                  <h2 id={`crate-${crate.name}`}>
                    <code>{crate.name}</code>
                  </h2>
                  <span
                    class="badge"
                    data-ok={String(crate.builtProfiles.length > 0)}
                  >
                    {crate.builtProfiles.length
                      ? `Built: ${crate.builtProfiles.join(", ")}`
                      : "Not built"}
                  </span>
                  {noteSlugs.has(crate.name) && (
                    <a href={`/wiki/${crate.name}`}>Read the notes</a>
                  )}
                </div>
                {crate.description && (
                  <div
                    class="crate-desc"
                    // README text comes from the local workspace.
                    // deno-lint-ignore react-no-danger
                    dangerouslySetInnerHTML={{
                      __html: renderMarkdown(crate.description).html,
                    }}
                  />
                )}
                <p class="crate-meta">
                  {crate.sources.length} source file
                  {crate.sources.length === 1 ? "" : "s"},{" "}
                  {crate.sources.reduce((n, s) =>
                    n + s.lines, 0)} lines
                </p>
                {crate.readme && (
                  <details id={`${crate.dir}/README.md`} class="source">
                    <summary>
                      <code>{crate.dir}/README.md</code>
                      <span>{crate.readme.split("\n").length} lines</span>
                    </summary>
                    <Readme crate={crate.dir} markdown={crate.readme} />
                  </details>
                )}
                {crate.sources.map((src) => (
                  <details id={src.path} class="source">
                    <summary>
                      <code>{src.path}</code>
                      <span>
                        {src.lines} lines, changed {timeAgo(src.modified)}
                      </span>
                    </summary>
                    <SourceCode code={src.content} lang="rust" />
                  </details>
                ))}
                <details class="source">
                  <summary>
                    <code>{crate.dir}/Cargo.toml</code>
                  </summary>
                  <SourceCode code={crate.manifest} lang="toml" />
                </details>
              </section>
            ))}

            <section aria-label="Workspace manifest">
              <details class="source">
                <summary>
                  <code>Cargo.toml</code>
                  <span>Workspace root</span>
                </summary>
                <SourceCode code={ws.manifest} lang="toml" />
              </details>
            </section>
          </>
        )}
    </div>
  );
});

function Readme({ crate, markdown }: { crate: string; markdown: string }) {
  // The crate heading is already on the page, so drop the README's title.
  const body = markdown.replace(/^# .*\n/, "");
  const { html } = renderMarkdown(body, {
    idPrefix: `${crate}-readme-`,
    headingShift: 1,
  });
  return (
    <div
      class="prose readme"
      // README text comes from the local workspace.
      // deno-lint-ignore react-no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function SourceCode({ code, lang }: { code: string; lang: string }) {
  // codeBlock escapes the source through highlight.js before adding markup.
  const __html = codeBlock(code, lang, { lineNumbers: true });
  // deno-lint-ignore react-no-danger
  return <div dangerouslySetInnerHTML={{ __html }} />;
}
