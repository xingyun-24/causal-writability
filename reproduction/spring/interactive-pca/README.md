# Spring Interactive PCA

- `spring_raw_projection_interactive.html`: 256 raw aligned/conflict activations
  projected onto the frozen matched-difference PC1/PC2/PC3 basis.
- `spring_matched_difference_interactive.html`: the corresponding 128 difference points.
- `data.json`: the plotted values and experiment settings.
- `build.py`: build the HTML files from `records/` using the bundled Plotly library.
- `capture.py`: regenerate projections and decoded frequency measurements with the
  Spring release runtime and supplied checkpoint/basis.

Both HTML files contain their plotting library and open without an internet
connection. Their layout, marker styles, colorscale, axes and controls follow
YuanMan's original Plotly 2.35.2 view.

The condition legend is moved away from the colorbar to avoid their overlap in
the original template. Narrow screens start farther out and use short PC axis
labels. Point data, marker styles and the plotting template are unchanged by
these layout adjustments.

## Data

Large Short seed 3408, 50K checkpoint, zero-based block 6. The PCA basis was fitted
only on 128 fit pairs; the pages display the other 128 held-out pairs. Each basis
vector spans the condition residuals concatenated over all 20 flow-matching calls.

The raw view contains 64 points in each input-color/observed-motion group:
red-slow, red-fast, blue-slow and blue-fast. Color on the plot is the freshly
decoded frequency, not input RGB; circle/diamond denotes aligned/conflict.
All 256 decoded natural rollouts passed the existing evaluator.

The difference view uses the same three PCA vectors. Its color is the matched
aligned rollout's decoded frequency; it is not a frequency measured from an
intervention. The first three difference coordinates reproduce the archived
held-out coordinates within 3e-5 absolute error. No coordinates were jittered,
normalized per group or changed for appearance.

These are raw PC1/PC2/PC3 views. They differ from the earlier Spring interactive
page, whose axes were route offset and two direction-specific phase coordinates.
The paper PDF and existing paper figures have not been changed.
