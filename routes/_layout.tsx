import { define } from "../utils.ts";
import SearchPalette from "../islands/SearchPalette.tsx";
import ThemeToggle from "../islands/ThemeToggle.tsx";

const NAV = [
  { href: "/wiki", label: "Wiki" },
  { href: "/playground", label: "Playground" },
  { href: "/project", label: "Project" },
];

export default define.layout(function Layout({ Component, url }) {
  return (
    <div class="shell">
      <a class="skip-link" href="#main">Skip to content</a>
      <header class="site-header">
        <a href="/" class="brand" aria-label="tileworld home">
          <span class="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          tileworld
        </a>
        <nav aria-label="Main">
          {NAV.map((item) => (
            <a
              href={item.href}
              aria-current={url.pathname.startsWith(item.href)
                ? "page"
                : undefined}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div class="header-tools">
          <SearchPalette />
          <ThemeToggle />
        </div>
      </header>
      <main id="main">
        <Component />
      </main>
      <footer class="site-footer">
        <p>
          A notebook for learning{" "}
          <a href="https://github.com/NVlabs/cutile-rs">cuTile Rust</a>. Notes
          are markdown files in{" "}
          <code>content/</code>; the project page reads the{" "}
          <a href="https://github.com/phiat/rust-gpu-lab">rust-gpu-lab</a>{" "}
          workspace from disk.{" "}
          <a href="https://github.com/phiat/rust-gpu-lab-web">Site source</a>.
        </p>
      </footer>
    </div>
  );
});
