import { useEffect, useState } from "preact/hooks";

type Mode = "system" | "light" | "dark";

const NEXT: Record<Mode, Mode> = {
  system: "light",
  light: "dark",
  dark: "system",
};

const LABEL: Record<Mode, string> = {
  system: "Theme: match system",
  light: "Theme: light",
  dark: "Theme: dark",
};

function apply(mode: Mode) {
  const root = document.documentElement;
  if (mode === "system") delete root.dataset.theme;
  else root.dataset.theme = mode;
  try {
    if (mode === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", mode);
  } catch {
    // Storage blocked: the choice lasts for this page view only.
  }
}

export default function ThemeToggle() {
  const [mode, setMode] = useState<Mode>("system");

  useEffect(() => {
    const current = document.documentElement.dataset.theme;
    if (current === "light" || current === "dark") setMode(current);
  }, []);

  return (
    <button
      type="button"
      class="icon-button"
      aria-label={`${LABEL[mode]}. Switch to ${
        LABEL[NEXT[mode]].toLowerCase()
      }`}
      title={LABEL[mode]}
      onClick={() => {
        const next = NEXT[mode];
        apply(next);
        setMode(next);
      }}
    >
      <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
        {mode === "light" && (
          <g fill="none" stroke="currentColor" stroke-width="1.6">
            <circle cx="10" cy="10" r="3.5" />
            <path d="M10 1.5v2.5M10 16v2.5M1.5 10H4M16 10h2.5M4 4l1.8 1.8M14.2 14.2L16 16M4 16l1.8-1.8M14.2 5.8L16 4" />
          </g>
        )}
        {mode === "dark" && (
          <path
            d="M15.5 12.5A6.5 6.5 0 0 1 7.5 4.5a6.5 6.5 0 1 0 8 8z"
            fill="none"
            stroke="currentColor"
            stroke-width="1.6"
            stroke-linejoin="round"
          />
        )}
        {mode === "system" && (
          <g>
            <circle
              cx="10"
              cy="10"
              r="7"
              fill="none"
              stroke="currentColor"
              stroke-width="1.6"
            />
            <path d="M10 3a7 7 0 0 1 0 14z" fill="currentColor" />
          </g>
        )}
      </svg>
    </button>
  );
}
