// Import CSS files here for hot module reloading to work.
import "./assets/styles.css";

// Copy buttons on code blocks. Server-rendered blocks (wiki, project page)
// and island-rendered ones share the same markup, so one delegated listener
// covers all of them.
async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard API needs a secure context; fall back for plain http.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

document.addEventListener("click", async (event) => {
  const button = (event.target as Element | null)?.closest<HTMLButtonElement>(
    "[data-copy]",
  );
  const code = button?.closest(".codeblock")?.querySelector("code");
  if (!button || !code) return;
  const ok = await copyText(code.textContent ?? "");
  button.textContent = ok ? "Copied" : "Copy failed";
  button.dataset.state = ok ? "done" : "failed";
  setTimeout(() => {
    button.textContent = "Copy";
    delete button.dataset.state;
  }, 1600);
});
