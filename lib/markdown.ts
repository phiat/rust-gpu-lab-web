import { Marked, type Tokens } from "marked";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import ini from "highlight.js/lib/languages/ini";
import rust from "highlight.js/lib/languages/rust";
import typescript from "highlight.js/lib/languages/typescript";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("sh", bash);
hljs.registerLanguage("toml", ini);
hljs.registerLanguage("rust", rust);
hljs.registerLanguage("rs", rust);
hljs.registerLanguage("ts", typescript);
hljs.registerLanguage("yaml", yaml);

const LANGUAGE_NAMES: Record<string, string> = {
  bash: "Shell",
  sh: "Shell",
  rust: "Rust",
  rs: "Rust",
  toml: "TOML",
  ts: "TypeScript",
  yaml: "YAML",
  text: "Text",
};

export interface Heading {
  depth: number;
  text: string;
  id: string;
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[`*_~]/g, "")
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

/** Strip inline markdown markers, for showing a heading as plain text. */
const plainText = (text: string) => text.replace(/[`*_~]/g, "");

/** A document's h2 and h3 headings, with the ids `renderMarkdown` gives them. */
export function headingsOf(src: string, idPrefix = ""): Heading[] {
  return new Marked().lexer(src)
    .filter((token): token is Tokens.Heading =>
      token.type === "heading" && (token.depth === 2 || token.depth === 3)
    )
    .map(({ depth, text }) => ({
      depth,
      text: plainText(text),
      id: idPrefix + slugify(text),
    }));
}

export const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Highlight code; unknown languages are just escaped. */
export function highlight(code: string, lang: string): string {
  return hljs.getLanguage(lang)
    ? hljs.highlight(code, { language: lang }).value
    : escapeHtml(code);
}

/**
 * A code block with a language label and a copy button (wired up in
 * client.ts). With `lineNumbers`, a gutter is rendered beside the code.
 */
export function codeBlock(
  code: string,
  lang: string,
  { lineNumbers = false }: { lineNumbers?: boolean } = {},
): string {
  const body = code.replace(/\n$/, "");
  const label = LANGUAGE_NAMES[lang] ?? lang;
  const gutter = lineNumbers
    ? `<span class="gutter" aria-hidden="true">${
      body.split("\n").map((_, i) => i + 1).join("\n")
    }</span>`
    : "";
  return `<figure class="codeblock"><figcaption><span>${
    escapeHtml(label)
  }</span><button type="button" class="copy" data-copy>Copy</button></figcaption><pre${
    lineNumbers ? ' class="numbered"' : ""
  }>${gutter}<code class="hljs language-${escapeHtml(lang)}">${
    highlight(body, lang)
  }</code></pre></figure>`;
}

/**
 * Render markdown to HTML. Relative links to other notes (`./foo.md`) become
 * `/wiki/foo`, and h2/h3 headings get ids for the table of contents.
 *
 * To embed a document inside another page, `idPrefix` keeps heading ids
 * unique and `headingShift` demotes headings (1 turns `##` into `<h3>`).
 */
export function renderMarkdown(
  src: string,
  { idPrefix = "", headingShift = 0 }: {
    idPrefix?: string;
    headingShift?: number;
  } = {},
): {
  html: string;
  headings: Heading[];
} {
  const headings = headingsOf(src, idPrefix);
  const marked = new Marked();

  marked.use({
    walkTokens(token) {
      if (token.type === "link") {
        const link = token as Tokens.Link;
        const m = link.href.match(/^\.\/([\w-]+)\.md(#.*)?$/);
        if (m) link.href = `/wiki/${m[1]}${m[2] ?? ""}`;
      }
    },
    renderer: {
      heading({ tokens, depth, text }) {
        const inner = this.parser.parseInline(tokens);
        const id = idPrefix + slugify(text);
        const level = Math.min(6, depth + headingShift);
        return `<h${level} id="${id}"><a class="anchor" href="#${id}" aria-label="Link to this section">#</a>${inner}</h${level}>\n`;
      },
      code({ text, lang }) {
        return codeBlock(text, (lang ?? "").split(/\s/)[0] || "text");
      },
    },
  });

  const html = marked.parse(src, { async: false }) as string;
  return { html, headings };
}
