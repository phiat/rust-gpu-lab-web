import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { SearchEntry } from "../routes/api/search.ts";

interface Result {
  href: string;
  title: string;
  context: string;
  score: number;
}

const MAX_RESULTS = 12;

function search(index: SearchEntry[], query: string): Result[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return index.map((e) => ({
      href: `/wiki/${e.slug}`,
      title: e.title,
      context: e.summary,
      score: 0,
    }));
  }
  const terms = q.split(/\s+/);
  const matches = (text: string) => {
    const t = text.toLowerCase();
    return terms.every((term) => t.includes(term));
  };

  const results: Result[] = [];
  for (const e of index) {
    const noteText = `${e.title} ${e.summary} ${e.section}`;
    if (matches(noteText)) {
      results.push({
        href: `/wiki/${e.slug}`,
        title: e.title,
        context: e.summary,
        score: e.title.toLowerCase().includes(q) ? 3 : 1,
      });
    }
    for (const h of e.headings) {
      if (
        matches(`${h.text} ${e.title}`) &&
        h.text.toLowerCase().includes(terms[0])
      ) {
        results.push({
          href: `/wiki/${e.slug}#${h.id}`,
          title: h.text,
          context: `In ${e.title}`,
          score: h.text.toLowerCase().includes(q) ? 2 : 1,
        });
      }
    }
  }
  return results.sort((a, b) => b.score - a.score).slice(0, MAX_RESULTS);
}

export default function SearchPalette() {
  const [index, setIndex] = useState<SearchEntry[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const results = useMemo(
    () => (index ? search(index, query) : []),
    [index, query],
  );

  function open() {
    const dialog = dialogRef.current;
    if (!dialog || dialog.open) return;
    dialog.showModal();
    inputRef.current?.focus();
    if (!index) {
      fetch("/api/search")
        .then((r) => r.json())
        .then(setIndex)
        .catch(() => setFailed(true));
    }
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const typing = target?.closest(
        "input, textarea, select, [contenteditable]",
      );
      const shortcut = (e.key === "k" && (e.metaKey || e.ctrlKey)) ||
        (e.key === "/" && !typing);
      if (shortcut) {
        e.preventDefault();
        open();
      }
    }
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  function onInputKey(e: KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter" && results[active]) {
      e.preventDefault();
      location.href = results[active].href;
      dialogRef.current?.close();
    }
  }

  return (
    <>
      <button type="button" class="search-trigger" onClick={open}>
        <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
          <circle
            cx="8.5"
            cy="8.5"
            r="5.5"
            fill="none"
            stroke="currentColor"
            stroke-width="1.7"
          />
          <path
            d="M12.8 12.8L17 17"
            stroke="currentColor"
            stroke-width="1.7"
            stroke-linecap="round"
          />
        </svg>
        <span>Search notes</span>
        <kbd>/</kbd>
      </button>

      <dialog
        ref={dialogRef}
        class="search-dialog"
        aria-label="Search notes"
        onClose={() => {
          setQuery("");
          setActive(0);
        }}
        onClick={(e) => {
          if (e.target === dialogRef.current) dialogRef.current?.close();
        }}
      >
        <div class="search-box">
          <input
            ref={inputRef}
            type="search"
            placeholder="Search notes, e.g. partition, JIT, select"
            value={query}
            aria-controls="search-results"
            aria-activedescendant={results[active]
              ? `search-result-${active}`
              : undefined}
            onInput={(e) => {
              setQuery(e.currentTarget.value);
              setActive(0);
            }}
            onKeyDown={onInputKey}
          />
          <ul id="search-results" role="listbox">
            {failed && (
              <li class="search-empty">
                Couldn't load the note index. Check that the server is running.
              </li>
            )}
            {!failed && index && results.length === 0 && (
              <li class="search-empty">
                No notes match “{query}”. Try a shorter word.
              </li>
            )}
            {results.map((r, i) => (
              <li
                key={r.href}
                id={`search-result-${i}`}
                role="option"
                aria-selected={i === active}
              >
                <a
                  href={r.href}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => dialogRef.current?.close()}
                >
                  <strong>{r.title}</strong>
                  <span>{r.context}</span>
                </a>
              </li>
            ))}
          </ul>
          <p class="search-hint">
            <kbd>↑</kbd> <kbd>↓</kbd> to move, <kbd>Enter</kbd> to open,{" "}
            <kbd>Esc</kbd> to close
          </p>
        </div>
      </dialog>
    </>
  );
}
