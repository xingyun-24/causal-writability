# Free fall panel kit: train-only PCA, rank 2

This package supersedes the earlier panel kit for PCA-dependent claims. It uses 64 pairs to fit an uncentered Block-1 PCA basis and reserves 64 disjoint pairs for final evaluation. The official controller is direction-specific `[1,g_target]` and never reads a held-out aligned activation or generated gravity.

Headline held-out results:

- Top-2 oracle projection: 62/64 E3 correct; 47/64 with `0.9 <= R <= 1.1`.
- Fit-only controller: 61/64 E3 correct; 46/64 with `0.9 <= R <= 1.1`; mean R 0.993727.
- Full matched edit: 64/64 E3 correct.

The four paper figures are separate under `figures/`. Machine-readable tables, full videos, representative strips, scripts, hashes, and audit records are included. Read `QA.md` before using the figures in a paper; the archived layer scan has narrower evaluator diagnostics than the newly generated F1 and F3 videos.
