import { build } from "vite";
import react from "@vitejs/plugin-react";
import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "out/pages");
const base = "/causal-writability/";
const shared = { root, configFile: false, envDir: false, publicDir: false, base, plugins: [react()] };

await build({ ...shared, build: { outDir: output, emptyOutDir: true } });
await build({ ...shared, build: { ssr: "static-site/render.tsx", outDir: "out/pages-ssr", emptyOutDir: true } });
const { renderPage } = await import(pathToFileURL(path.join(root, "out/pages-ssr/render.js")).href);
const index = path.join(output, "index.html");
const html = (await readFile(index, "utf8")).replace("<!--app-html-->", renderPage());
if (!html.includes("A Chosen Future") || html.includes("<!--app-html-->")) throw new Error("Missing prerendered page");
if (html.includes('/_local/') || html.includes('Local preview')) throw new Error("Local preview leaked into release");
await writeFile(index, html);

// Ship media used by the page, excluding local previews and historical handoff files.
async function copy(relative) {
  const destination = path.join(output, relative);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(path.join(root, "public", relative), destination, { recursive: true });
}
await copy("paper/main.pdf");
for (const directory of ["paper/vector", "paper/appendix", "pendulum", "interactives"]) {
  for (const name of await readdir(path.join(root, "public", directory))) {
    if (/\.(svg|html|json|png)$/.test(name)) await copy(path.join(directory, name));
  }
}
const videos = [
  "selection-fast-red-endpoint.mp4", "selection-fast-purple.mp4",
  "selection-slow-blue-endpoint.mp4", "selection-slow-purple.mp4",
  "controller-natural-conflict.mp4", "controller-state-edit.mp4",
  "realization-natural-conflict.mp4", "realization-vh8-edit.mp4",
  "overview.mp4", "overview-poster.png",
];
for (const name of videos) await copy(`videos/${name}`);
for (const directory of ["videos/pendulum", "videos/freefall/originals"]) {
  for (const name of await readdir(path.join(root, "public", directory))) {
    if (name.endsWith(".mp4")) await copy(path.join(directory, name));
  }
}
await writeFile(path.join(output, ".nojekyll"), "");
console.log(`GitHub Pages build ready at ${output} (base ${base})`);
