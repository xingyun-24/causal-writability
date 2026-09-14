import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const output = fileURLToPath(new URL("../out/pages/", import.meta.url));
const prefix = "/causal-writability/";

test("static release contains the same page and preserves client-side interactivity", async () => {
  const html = await readFile(path.join(output, "index.html"), "utf8");
  assert.match(html, /A Chosen Future/);
  assert.match(html, /<script\b[^>]*type="module"[^>]*src="\/causal-writability\/assets\//);
  assert.ok(!html.includes("Local preview") && !html.includes("/_local/"));
  assert.ok(html.includes("https://github.com/xingyun-24/causal-writability"));
  assert.ok(html.includes(`${prefix}videos/overview.mp4`));
  assert.ok(!html.includes("overview-en"));
  assert.ok(html.includes("FIGURE 6B"));
  for (const match of html.matchAll(/(?:src|href|poster)="(\/[^"#?]+)(?:[^\"]*)"/g)) {
    assert.ok(match[1].startsWith(prefix), `Unprefixed asset: ${match[1]}`);
    await access(path.join(output, decodeURIComponent(match[1].slice(prefix.length))));
  }
  for (const name of ["spring_raw_projection_interactive", "spring_matched_difference_interactive", "pendulum-project-pca", "pendulum-difference-pca", "freefall_raw_projection_interactive", "freefall_matched_difference_interactive"]) {
    await access(path.join(output, "interactives", `${name}.html`));
  }
});
