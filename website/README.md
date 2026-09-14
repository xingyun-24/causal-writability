# Causal Writability project page

2026-09-14: the active page again uses Leqian's `website-test_v1` structure,
with the current Pendulum section and the new Free Fall/PCA additions.
Leqian's original style is the selected project-page design. The A/B switcher
has been removed; old `?style=clean` links also display the selected design.
The previous comparison is recorded in `STYLE_COMPARISON_ZH.md` and Git history.
The page is deployed at https://xingyun-24.github.io/causal-writability/.

Scientific figures now use vector SVGs. `scripts/export_vector_panels.py`
exports current-paper subpanels with outlined fonts, keeping embedded decoded
frames unchanged. `scripts/export_appendix.py` exports all 27 appendix figures
(7--33) and resolves their numbers and PDF pages from the final manuscript.
The Supplement section groups them into six topics and mounts images only
when a group is expanded; each figure links to the full caption in the paper.

Active project repository: `xingyun-24/causal-writability`, branch `main`.
The selected page is maintained here. The previous private
integration branch remains a historical collaborator handoff, not the release
destination. Publish only after author review.

## Run

Node >=22.13 is required by the existing vinext/Vite stack.

```bash
npm ci
npm run dev -- --host 127.0.0.1 --port 4317
npm test
npm run build:pages
```

## Deployment

The GitHub Pages build renders the same `app/page.tsx` and components into
static HTML and hydrates them with React, retaining video controls, Pendulum
tabs, interactive PCA views and the appendix gallery. `static-site/build.mjs` copies
only release media and uses the `/causal-writability/` base path. It does not
publish `.env.local` or `public/_local`.

Changes to `website/` on `main` trigger `.github/workflows/pages.yml`. The
workflow can also be started manually. The old `export_github_pages.py` is a
historical non-interactive export and is not used for deployment.

## Publication settings

Edit `app/site-content.ts` for the trailer MP4, poster, optional captions,
repository, arXiv and project URL. The header links directly to Paper and GitHub;
model downloads remain in the repository documentation, not the page header.
The canonical GitHub repository is public. The final bilingual overview film
is included in `public/videos/overview.mp4`.
While the trailer is unset, Hero
shows the current paper's Figure 1 without a pretend play button.

The paper link is `public/paper/main.pdf`, the authored arXiv-ready version.
`scripts/export_current_paper.py --source /path/to/arxiv-source --out public/paper/current`
exports its six main figures without modifying their contents. Pendulum
images use `../reproduction/pendulum/paper/final/`.

## Integrated evidence

- Eight Spring MP4s: four natural cue comparisons, a controller comparison,
  and a condition-V/head-8 comparison. Uses Leqian's audited video set from
  `website-test_v1` with a shared two-video transport, not legacy placeholders.
- Four current Pendulum MP4s, including the Top-4 B_082 intervention.
- Four original-resolution Free Fall panel-kit MP4s under `videos/freefall/originals/`.
- Six standalone, offline Plotly 2.35.2 views: raw projections and matched
  differences for Spring, Pendulum and Free Fall. The page loads them near
  the viewport and mounts only the selected view.

New Pendulum embeds use Leqian's coordinates unchanged. Their difference-view
color is **aligned decoded frequency**, matching Spring, rather than the
absolute frequency gap used in his original two-plot page. Free Fall uses
decoded gravity; its difference view uses aligned decoded gravity. The
original Pendulum JSON remains intact as a source record.

Free Fall uses B1, hist32, 100K and the archived 64-fit/64-held-out split.
The raw view has 128 endpoints, 32 per input-color/gravity group. The actual
third PC is reconstructed from fit residuals; the paper's rank-2 controller
is not changed. See `../reproduction/freefall/PROJECT_PAGE_PCA.md`.

## Source history and scope

`LEQIAN_PROJECT_PAGE_HANDOFF.md` and `references/PROJECT_PAGE_REQUIREMENTS_ORIGINAL.md`
are historical requirements, not authority for current numbers or figure
labels. The new source replaces old titles, controller descriptions and
placeholders, while reusing the audited assets and paper's current evidence.
`PendulumGenerality.tsx` remains active, with its current Top-4 example and
updated numerical details. The original Spring video layout is preserved.

No new pretrained layer-scan candidate is inserted into the paper or page.
No model training or recoloring of decoded frames is part of the page build.
