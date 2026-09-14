# Project-page handoff for Leqian

> **2026-08-25 update:** For the current Pendulum paper/page assignment, follow
> `../handoff/leqian_20260825/LEQIAN_PENDULUM_TASK.md` and
> `../handoff/leqian_20260825/PENDULUM_DELIVERY_SPEC.md`. Those files supersede
> the older controller wording, figure numbering, and Generality placeholders in
> this document. Leqian owns Pendulum integration; Projectile remains a separate
> owner deliverable.

## Ownership and scope

This branch intentionally provides a scientifically frozen asset bundle rather
than a finished page redesign. Leqian owns the final React/CSS composition,
responsive behavior, interaction design, and visual polish. Please preserve
that ownership in the final implementation commit/PR.

The page must remain anonymous while the paper is under double-blind review.
Do not add author names, affiliations, analytics, or a non-anonymous deployment
URL before the review policy permits it.

## Frozen core copy and selling points

The rest of the page copy is flexible. These are the sentences the page should
make memorable.

### Level A: preserve nearly verbatim

**Paper title**

> How a World Model Settles on a Future: Physics, Predictive Shortcuts, and
> Causal Control in Video Generation

**Hero question**

> When a world model supports multiple futures, what determines which
> continuation controls its generated future?

**Hero thesis**

> A world model can follow a predictive shortcut even while the rejected
> continuation remains causally accessible.

**Closing thesis**

> Physical failure can reflect competition for control even when a usable
> dynamics route is present.

### Level B: preserve the scientific meaning; wording and layout are flexible

**Solution selection**

> Cue strength and physical history select among training-supported joint
> continuations.

**Causal access**

> The rejected continuation remains accessible through a compact,
> state-structured causal route.

**Prospective controller**

> Direction chooses the route family; boundary phase chooses where on it.

**Writeability and commitment**

> A future is causally writable while an internal edit can redirect it, and
> causally committed once this direct access closes.

**Training and error fate**

> Training simultaneously rescues some errors and consolidates the ones that
> remain.

> Mechanistic divergence precedes behavioral divergence.

**Downstream authority**

> Availability is not authority.

> Condition-to-target self-attention writes give the route control over the
> generated future.

**Independent solutions**

> Shared causal semantics, distinct downstream realizations.

### Compact selling-point list

If a visitor only scans the page, they should leave with these five points:

1. Selection is evidence-dependent, not a fixed color-over-physics hierarchy.
2. Shortcut behavior does not imply that the rejected future is absent.
3. A low-capacity, donor-free controller can call the rejected continuation
   from the input boundary state.
4. Writeability predicts later error fate and closes into causal commitment.
5. Independently trained models share causal route geometry but implement
   downstream authority differently.

### Optional proof points

Use these as compact evidence cards, not as a second abstract:

- **23/26** endpoint shortcut-frequency failures return to the
  history-consistent frequency band under the ambiguous cue in the displayed
  checkpoint.
- Across the three canonical Large Short 50K runs, the top-four edit recovers
  to within **0.004--0.006** of the full matched edit.
- Across **384 held-out interventions**, the donor-free controller has median
  normalized recovery **0.978** across the six run-by-direction groups.
- In **15/15 training runs**, overall physics-following behavior rises from 5K
  to 100K while writeability among the remaining strict failures contracts.
- All six directed transfers between three independently trained runs retain
  pooled median recovery **0.942--0.991**.
- In the low-physics run, editing only condition-token V in one of nine heads
  raises clean physics from **0% naturally to 56.3% at gain 8**; stronger gains
  increasingly leave the supported modes.

Keep the stated scopes. In particular, 23/26 is a frequency-band count, not a
separately thresholded joint motion-and-color success rate.

## Frozen scientific order

Use this narrative order:

1. Hero: the model selects among training-supported joint continuations.
2. Decoded solution-selection surface.
3. Matched causal rewrite: the rejected continuation is still accessible.
4. Donor-free state-conditioned controller.
5. Causal writeability and commitment.
6. Condition-to-target realization.
7. Shared causal semantics and distinct downstream implementations.
8. Generality only after Pendulum/Projectile replication is complete.

The current source already follows Selection → Control → Commitment →
Realization at the chapter level. The important redesign is inside Control:
matched causal access must precede the prospective controller. Do not present
the controller as the sole evidence that the rejected continuation exists.

## Correct Figure 3 controller contract

The prospective controller reads only:

- requested target direction r;
- input-history boundary phase theta_star.

For each requested direction:

z_hat_r(theta_star)
  = beta_0_r
  + beta_c_r cos(theta_star)
  + beta_s_r sin(theta_star).

The executable path is:

target direction + boundary phase
→ predicted top-four route coordinates
→ frozen V4 reconstruction
→ additive condition-state edit
→ decoded future.

Preferred one-line explanation:

> Direction chooses the route family; boundary phase chooses where on it.

The equivalent pooled implementation uses
[1, cos(theta), sin(theta), s, s cos(theta), s sin(theta)].

Remove the retired main-page story using
[1, omega, cos(theta), sin(theta), omega cos(theta), omega sin(theta)].
That six-term continuous-omega audit was not the controller used for the
decoded causal intervention.

The retrospective coordinate-difference model uses decoded future frequency
and phase differences. It may appear only in technical/supplementary details
with this label:

> Reads the realized decoded future; structural diagnostic, not prospective
> control.

## Figure 3 reproduction: keep three top-level targets

Do not expand the public reproduction menu beyond Behavior, Commitment, and
Causal editing. Split Causal editing internally:

### 3A. Canonical matched causal rewrite

matched aligned/conflict pair
→ condition-state difference
→ intervention at the functional site
→ decoded rollout
→ normalized frequency recovery.

Claim licensed:

> The alternative continuation remains causally accessible.

### 3B. Donor-free state-conditioned synthesis

fit trajectories
→ target direction + boundary phase
→ predicted causal coordinates
→ frozen V4 reconstruction
→ held-out condition-side intervention
→ decoded rollout.

Claim licensed:

> The physical parameterization is prospectively executable without a held-out
> donor activation.

## Frozen video set

All files below are browser-ready H.264 MP4, 64 frames at 20 fps. They preserve
the decoded pixels; no ball is recolored and no trajectory is retimed.
Machine-readable hashes and source contracts are in
public/videos/manifest.json.

### Solution selection: fast history

- /videos/selection-fast-red-endpoint.mp4
  - red endpoint cue;
  - natural red/slow shortcut continuation.
- /videos/selection-fast-purple.mp4
  - same physical history and generation seed;
  - purple cue;
  - natural blue-ish/fast history-consistent continuation.

Page label:

> Same fast physical history; only the appearance cue changes.

### Solution selection: slow history

- /videos/selection-slow-blue-endpoint.mp4
  - blue endpoint cue;
  - natural blue/fast shortcut continuation.
- /videos/selection-slow-purple.mp4
  - same physical history and generation seed;
  - purple cue;
  - natural red-ish/slow history-consistent continuation.

Page label:

> Same slow physical history; only the appearance cue changes.

These four videos are natural generations. Do not label the purple examples as
activation interventions.

### State-conditioned controller

- /videos/controller-natural-conflict.mp4
- /videos/controller-state-edit.mp4

Contract:

- same held-out receiver, noise, and future window;
- natural conflict selects red/slow;
- the fit-only phase-by-direction controller recovers blue/fast;
- condition prefix only at the frozen rewrite site;
- no held-out donor activation, matched difference, output state, or norm.

### Downstream V/h8 realization

- /videos/realization-natural-conflict.mp4
- /videos/realization-vh8-edit.mp4

Contract:

- same selection-clean Run-C fast-target receiver and future window;
- only condition-token V in head 8 is edited;
- natural red/slow becomes blue/fast.

This replay is a representative successful medoid selected from 28 eligible
successes. Do not imply that every member of the 48-trajectory cohort becomes
clean physics.

## Interactive Spring route geometry

The branch now includes a dependency-free, iframe-ready interactive:

- `/interactives/spring-route-geometry.html` — standalone canvas interaction;
- `/interactives/spring-route-geometry-data.json` — compact machine-readable data;
- `/interactives/spring-route-geometry-fallback.png` — no-JavaScript/static fallback;
- `/interactives/manifest.json` — checkpoint, held-out counts, fit quality, and hashes.

The frozen source is the same canonical Figure 3 checkpoint used by the paper:

- Large Short;
- seed 3408;
- 50K;
- functional block B6;
- 64 disjoint held-out trajectories for each target direction.

The displayed coordinates are:

1. common route-offset coordinate from the frozen raw/uncentered top-four PCA;
2. phase coordinate 1;
3. phase coordinate 2.

The two thin curves are first-harmonic predictions learned only on the fit
trajectories. The 128 displayed markers are held out. This is neither UMAP nor
t-SNE, and held-out points are never used to refit the PCA, phase plane, or
curves.

### Where it belongs

Place the interactive inside the **State-structured causal route** chapter:

```text
matched causal rewrite
→ compact route / 2-D phase loops
→ interactive 3-D route geometry
→ donor-free state-conditioned controller
```

It should not appear before the matched causal rewrite: readers first need to
know that the plotted edit changes decoded video. It should not appear after
writeability/commitment, because it explains route structure rather than route
closure.

### Recommended composition

Desktop:

- full-width section or a 65/35 split;
- interactive on the left (or full width);
- short interpretation on the right;
- use a 4:3 or 16:10 visible frame, at least 520 px high.

Mobile:

- stack heading, interactive, then interpretation;
- keep the canvas at least 360 px high;
- preserve the built-in Fast/Slow/Both and Reset controls.

Suggested heading:

> **A compact causal route is organized by boundary phase**

Suggested one-line explanation:

> Target direction selects the route family; boundary phase moves the edit
> around it.

Suggested evidence line:

> PCA and phase alignment are fit on disjoint trajectories; every displayed
> point is held out.

Suggested React embedding:

```tsx
<iframe
  src="/interactives/spring-route-geometry.html"
  title="Interactive three-dimensional view of the held-out Spring causal route"
  loading="lazy"
  sandbox="allow-scripts allow-same-origin"
  className="routeGeometryFrame"
/>
```

The iframe has its own controls, responsive canvas, hover labels, mouse/touch
rotation, wheel/pinch zoom, and no external JavaScript dependency. Do not add
auto-rotation: motion would compete with the decoded videos and reduce
accessibility.

### Visual and claim guardrails

- Keep the page-level background white and do not wrap the iframe in a heavy
  rounded card.
- Phase uses the frozen cyclic teal/neutral palette. Do not recolor target fast
  and target slow red/blue; direction is already encoded by marker shape,
  spatial separation, filters, and direct labels.
- Default to **Both** so the two separated route families are visible.
- Keep the fit curves thin and subordinate to the held-out points.
- Do not call the axes raw PC1/PC2/PC3. The first is the common route-offset
  coordinate; the other two are fit-only phase-aligned coordinates inside the
  frozen top-four route.
- Do not place seed, checkpoint, or block in the headline. They are preserved
  in the manifest and may appear inside an expandable “Technical details” row.
- Preserve the static PNG as the `noscript` and social-preview fallback.

## Updated figure bundle

Use the complete SVG figures for responsive page composition:

- /paper/svg/figure1-visual-thesis.svg
- /paper/svg/figure2-solution-selection.svg
- /paper/svg/figure3-causal-route.svg
- /paper/svg/figure4-writeability.svg
- /paper/svg/figure5-downstream.svg

Their hashes and roles are in /paper/svg/manifest.json. The existing Figure
1--5 PNG crop filenames under public/paper are retained as quick previews and
fallback assets. Both were regenerated from the audited paper-v1 sources. The
current paper snapshot is:

- /paper/main.pdf

New supplement assets supplied for optional use:

- /paper/svg/figure6-multiseed-behavior.svg
- /paper/svg/figure7-controller-ablation.svg
- /paper/svg/figure14-cross-step-exact100k.svg
- /paper/svg/figure15-implementation-multiplicity.svg
- /paper/svg/figure17-fm-time-decoded.svg

Matching PNG previews are also present at the corresponding top-level
/paper/fig*.png paths.

The old main32.pdf and old Figure 6–10 crops are historical website artifacts;
do not use them as scientific authority.

## P0 text fixes in the current page source

Before publishing, fix these statements in app/page.tsx:

1. Paper link:
   - old: /paper/main32.pdf
   - current: /paper/main.pdf
2. Figure 3 controller:
   - remove the omega/omega-cos/omega-sin basis;
   - use the phase-by-direction contract above;
   - full held-out coordinate R2 is .83–.88;
   - direction-only R2 is .51–.60.
3. Figure 4d:
   - use “all three matched-seed 50K comparisons”;
   - Short/Long use checkpoint-local banks, not trajectory-paired banks.
4. Figure 5d:
   - clean physics peaks at gain 8;
   - continuous recovery can overshoot at larger gain;
   - do not say frequency recovery itself peaks at gain 8.
5. Cross-step supplement:
   - exact checkpoints are 5/10/20/50/80/100K;
   - 90 directed off-diagonal maps, not 126;
   - blank diagonal means not evaluated;
   - oracle-coordinate temporal compatibility, not non-oracle control.
6. Target-conditioned table:
   - call the comparator “target-local full-edit reference”;
   - the 96.5/88.6/91.9 values are marginal rate ratios;
   - do not call them fractions of an oracle ceiling.
7. Supplement numbering:
   - current paper appendix is Figures 6–17 and Tables 1–6.
8. Remove “Absolute-state basis ablations” from the active story. The current
   controller ablation is target direction plus boundary phase.

## Recommended media behavior

- Show paired videos side by side with a shared group heading.
- Use muted, loop, playsInline, controls, and preload=metadata.
- Prefer click-to-play or viewport-aware autoplay; do not start all eight
  decoders off-screen on page load.
- Give both videos in a matched pair the same rendered dimensions.
- Preserve the native 1:1 frame; use object-fit: contain.
- Put the scientific contract above each pair, not in hover-only text.
- Distinguish natural cue changes from internal interventions with explicit
  labels. Teal remains reserved for interventions.
- Include a visible fallback link to the MP4 for accessibility.

## Files that should stay source-of-truth

- public/videos/manifest.json for video identity and provenance;
- public/paper/main.pdf for the frozen paper snapshot;
- the current paper figure assets listed above;
- this handoff for narrative and claim boundaries.

Do not use old website copy, old supplement numbers, or the retired controller
basis as evidence.
