import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../public/interactives/", import.meta.url);

test("Spring route interactive is scientifically frozen and self-contained", async () => {
  const [html, dataBuffer, fallback, manifestBuffer] = await Promise.all([
    readFile(new URL("spring-route-geometry.html", root), "utf8"),
    readFile(new URL("spring-route-geometry-data.json", root)),
    readFile(new URL("spring-route-geometry-fallback.png", root)),
    readFile(new URL("manifest.json", root)),
  ]);
  const data = JSON.parse(dataBuffer);
  const manifest = JSON.parse(manifestBuffer);

  assert.deepEqual(manifest.checkpoint, {
    model: "Large Short", seed: 3408, step: "50K", functional_block: "B6",
  });
  assert.equal(data.traces.length, 2);
  assert.equal(data.traces.reduce((total, trace) => total + trace.points.length, 0), 128);
  assert.ok(data.traces.every((trace) => trace.heldout_n === 64 && trace.fit.length === 241));
  assert.deepEqual(manifest.phase_r2, {
    fast: 0.9350764011473169,
    slow: 0.8810842823860321,
  });

  assert.match(html, /data-view="both"/);
  assert.match(html, /Reset view/);
  assert.match(html, /No UMAP or t-SNE/);
  assert.doesNotMatch(html, /<script[^>]+src=/);
  const payload = html.match(/<script[^>]*id="payload"[^>]*>([\s\S]*?)<\/script>/);
  assert.ok(payload);
  const embedded = JSON.parse(payload[1]);
  assert.deepEqual(embedded.traces, data.traces);
  assert.deepEqual(embedded.bounds, data.bounds);
  assert.equal(fallback.subarray(1, 4).toString(), "PNG");
});
