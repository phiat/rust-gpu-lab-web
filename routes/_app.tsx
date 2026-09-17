import { define } from "../utils.ts";

// Runs before first paint so a saved theme doesn't flash.
const THEME_SCRIPT =
  `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default define.page(function App({ Component }) {
  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta
          name="description"
          content="Notes, demos and a live workspace view for tileworld, a cuTile Rust learning project."
        />
        <script
          // Static inline script, no user input.
          // deno-lint-ignore react-no-danger
          dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }}
        />
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossorigin="anonymous"
        />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:ital,wght@0,400..800;1,400..600&family=Spline+Sans+Mono:wght@400..600&display=swap"
        />
        <title>tileworld</title>
      </head>
      <body>
        <Component />
      </body>
    </html>
  );
});
