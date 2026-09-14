# QA

- Frozen bank: 128 pairs. PCA/controller fit: 64. Final held-out evaluation: 64. Low/high counts are 32/32 in each split.
- Pair IDs and base seeds are disjoint across fit and held-out. One legacy crossing at base seed 50208 was repaired before PCA; see `split_manifest.json`.
- Boundary audit: 128/128 pass at tolerance 1e-12; maximum recorded boundary error is 0. Renderer nuisance hashes and generation seeds match within every pair.
- PCA: rank 2 selected from fit energy only (99.545247%); held-out energy (99.554430%) is diagnostic only. Exported feature components are orthonormal.
- Intervention site: after zero-based Block 1, selected as the deepest maximum-E3 site on the 64-pair fit split.
- F1: 1,408/1,408 futures regenerated and valid; measured future RGB, coverage, gap/jump diagnostics, MP4 path, and SHA256 are recorded. Raw points and medians are descriptive.
- F3: all four conditions contain the same 64 held-out receivers. All 256 videos are valid. Top-2 oracle E3 is 62/64; fit-only controller E3 is 61/64. Invalid outputs remain in denominators.
- F4: `layer_scan.csv` has 1,920 receiver-level rows (64 held-out x 30 sites). It reuses the archived full-matched layer scan. The archive did not retain RMSE, coverage, jump, or RGB diagnostics for each layer edit; those fields are blank in `evaluator_audit.csv`.
- Figures: four separate figures exported as editable SVG, PDF, and 600-dpi PNG using the supplied Pendulum palette and canvas sizes. PDF render and font audits are recorded in `figure_QA.json`.
- Rebuild: scripts, input hashes, producer names, file sizes, and SHA256 values are recorded in `manifest.json`. Model repository HEAD was 8386c6f0f7d23f38d1e74157da61c0ba47c246db with a dirty working tree; delivery scripts are therefore pinned by SHA256.
- Scientific scope: the 128-pair bank was selected using baseline aligned/conflict E3 behavior and had appeared in prior analyses. Held-out isolation is correct for the rebuilt PCA/controller pipeline, but this is not a fresh unseen population. The F4 source scan predates the repaired split; site selection is recomputed using fit IDs only.
