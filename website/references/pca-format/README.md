# Interactive PCA Format Reference

`raw_residual_pca_projection_interactive.html` is YuanMan's original format
reference. Its data are the earlier Block-2, 125-pair Free Fall analysis, not
the current Block-1 train-only result. Do not embed it as current paper evidence.

The new Spring pages, ready to embed, are:

- `/interactives/spring_raw_projection_interactive.html`
- `/interactives/spring_matched_difference_interactive.html`

They use current seed-3408, 50K, B6 projections with a fit-only basis and contain
their Plotly library for offline viewing. The first shows 256 raw state points
and the second 128 difference points. Marker color is decoded frequency, not
input RGB. Original HTML appearance is retained, with legend/colorbar separation
and narrow-screen adjustments.

For Pendulum, use the current fit-only basis, 64 held-out pairs (128 raw points),
and the same projection conventions. Supply actual PC1/2/3, not a fabricated
third coordinate from the 2-D phase-plane figure. Full details are in FORMAT_ZH.md.
