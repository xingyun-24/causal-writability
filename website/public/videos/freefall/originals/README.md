# Free Fall MP4 archive

These four files were already stored in YuanMan's
`freefall_panel_kit_trainonly_pca2_rebuilt/videos/recovery/`. They are copied
byte-for-byte, without transcoding, scaling, frame interpolation or a new rollout.
The model is hist32 / step-100000, zero-based B1, rank-2 train-only PCA,
20 FM calls; both receivers belong to the repaired 64-pair held-out split.

| Receiver | Natural conflict | Fit-only controller |
|---|---|---|
| `eval_50073_high` | `eval_50073_high__natural_conflict.mp4` | `eval_50073_high__fit_only_controller.mp4` |
| `eval_50281_low` | `eval_50281_low__natural_conflict.mp4` | `eval_50281_low__fit_only_controller.mp4` |

All four are 128×128, 20 fps, 64 future frames, in their original order.
Generation seeds are 17050073 and 17050281, shared within each pair.
`manifest.json` records original and copied hashes, PCA/controller hashes,
saved-frame provenance and decoded-frame checks.

**Meaning of original:** the rollout script saved NPZ RGB frames, not MP4.
These are the existing MP4 exports made by `build_panel_kit.py` for the panel kit;
no separate MP4 produced directly at rollout time was found. They are not the
older pooled-controller examples. RGB↔YUV conversion in that existing export
accounts for at most 2 intensity levels of difference from the saved RGB frames.
The earlier browser copies in the parent directory are retained separately.
