import { HttpError } from "fresh";
import { Head } from "fresh/runtime";
import { define } from "../utils.ts";

export default define.page(function ErrorPage({ error, url }) {
  const notFound = error instanceof HttpError && error.status === 404;
  return (
    <div class="page error-page">
      <Head>
        <title>
          {notFound ? "Page not found · tileworld" : "Error · tileworld"}
        </title>
      </Head>
      <h1>{notFound ? "No tile at these coordinates" : "This page failed"}</h1>
      <p class="lede">
        {notFound
          ? url.pathname.startsWith("/wiki/")
            ? (
              <>
                There's no note called{" "}
                <code>{url.pathname.split("/").pop()}</code>. To write it, add
                {" "}
                <code>content/{url.pathname.split("/").pop()}.md</code>{" "}
                and refresh.
              </>
            )
            : (
              <>
                Nothing lives at{" "}
                <code>{url.pathname}</code>. Check the address, or start from
                the wiki.
              </>
            )
          : "The server hit an error while rendering. The dev server output has the stack trace."}
      </p>
      <div class="actions">
        <a class="button primary" href="/wiki">Browse the wiki</a>
        <a class="button" href="/">Go home</a>
      </div>
    </div>
  );
});
