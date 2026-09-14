import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const { default: worker } = await import("../dist/server/index.js");
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

test("renders the shared narrative and active experimental videos", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /A Chosen Future/);
  assert.match(html, /Rewritten/);
  for (const id of ["choice", "rewrite", "commit", "realize", "pendulum", "freefall", "resources"]) assert.ok(html.includes(`id="${id}"`));
  // Eight Spring, two active Pendulum-tab clips, four Free Fall; optional local trailer.
  assert.ok([14, 15].includes((html.match(/<video\b/g) ?? []).length));
  assert.ok(!html.includes("VIDEO PENDING"));
  assert.ok(!html.includes("main32.pdf"));
  assert.ok(!html.includes("RANK-1 ORACLE"));
  assert.ok(!html.includes('src="null"'));
  assert.ok(!html.includes("Visual style comparison"));
  assert.ok(!html.includes("B · Editorial"));
  assert.ok(html.includes('id="fm-timing"'));
  assert.ok(html.includes("When the write acts matters"));
  assert.ok(html.includes("first and second halves of the denoising sequence"));
  assert.ok(html.includes("not training checkpoints"));
  const paths = [...new Set([...html.matchAll(/src="(\/[^"?#]+\.(?:mp4|png|svg))"/g)].map(m => m[1]))];
  for (const path of paths) await access(new URL(`../public${path}`, import.meta.url));
  assert.ok(!paths.some(path => /\/(paper|pendulum)\/.*\.png$/.test(path)), "Scientific figures must use vectors");
});

test("all six interactive embeds exist and carry Plotly offline", async () => {
  for (const name of ["spring_raw_projection_interactive", "spring_matched_difference_interactive", "pendulum-project-pca", "pendulum-difference-pca", "freefall_raw_projection_interactive", "freefall_matched_difference_interactive"]) {
    const html = await readFile(new URL(`../public/interactives/${name}.html`, import.meta.url), "utf8");
    assert.ok(html.includes("plotly.js v2.35.2"));
    assert.ok(!/<script[^>]+src=/.test(html));
  }
});

test("the header links to the paper and canonical GitHub repository", async () => {
  const html = await (await render()).text();
  const header = html.match(/<header\b[^>]*>([\s\S]*?)<\/header>/)?.[1];
  assert.ok(header);
  assert.match(header, /href="\/paper\/main\.pdf"/);
  const github = header.match(/<a\b[^>]*href="https:\/\/github\.com\/xingyun-24\/causal-writability"[^>]*>/)?.[0];
  assert.ok(github);
  assert.match(github, /target="_blank"/);
  assert.match(github, /rel="noopener noreferrer"/);
  assert.ok(!header.includes("huggingface.co"));
});

test("appendix catalogue includes every final figure with vector assets and paper page links", async () => {
  const figures = JSON.parse(await readFile(new URL("../public/paper/appendix/manifest.json", import.meta.url), "utf8"));
  assert.equal(figures.length, 27);
  assert.deepEqual(figures.map(f => f.number).sort((a,b) => a-b), Array.from({length:27}, (_,i) => i+7));
  assert.equal(new Set(figures.map(f => f.group)).size, 6);
  for (const figure of figures) {
    assert.ok(figure.page >= 14 && figure.page <= 34);
    const svg = await readFile(new URL(`../public${figure.src}`, import.meta.url), "utf8");
    assert.match(svg, /<path\b/);
    assert.ok(figure.vector_paths > 0);
  }
});

test("Free Fall raw projections match the frozen held-out differences", async () => {
  const data = JSON.parse(await readFile(new URL("../public/interactives/freefall-pca-data.json", import.meta.url), "utf8"));
  assert.equal(data.metadata.block_zero_based, 1);
  assert.equal(data.metadata.fit_pairs, 64);
  assert.equal(data.raw_points.length, 128);
  assert.equal(data.difference_points.length, 64);
  assert.deepEqual(Object.values(data.metadata.groups), [32, 32, 32, 32]);
  assert.ok(data.metadata.max_archived_score_absolute_error < 1e-6);
  for (const delta of data.difference_points) {
    const pair = data.raw_points.filter(p => p.pair_id === delta.pair_id);
    const aligned = pair.find(p => p.condition === "aligned");
    const conflict = pair.find(p => p.condition === "conflict");
    for (const pc of ["pc1", "pc2", "pc3"]) assert.ok(Math.abs(aligned[pc] - conflict[pc] - delta[pc]) < 1e-8);
  }
});
