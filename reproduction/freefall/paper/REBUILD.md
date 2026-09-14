# Deterministic rebuild

Run `scripts/build_panel_kit.py` with the retained source bundle, regenerated F1 directory, exported train-only basis, and aligned-reference NPZ. The exact invocation used for this release was:

```text
python build_panel_kit.py --work "C:\Users\34655\Documents\ChatGPT\New project\freefall_pca_redo" --old-kit "C:\Users\34655\Desktop\freefall\freefall_panel_kit" --f1-root "C:\Users\34655\Documents\ChatGPT\New project\freefall_pca_redo\f1_extracted" --basis "C:\Users\34655\Documents\ChatGPT\New project\freefall_pca_redo\delivery_inputs\pca_components.npy" --aligned-reference "C:\Users\34655\Documents\ChatGPT\New project\freefall_pca_redo\delivery_inputs\aligned_reference_eval_50281_low.npz" --out "C:\Users\34655\Desktop\freefall\freefall_panel_kit_trainonly_pca2_rebuilt"
```

The source model repository had uncommitted changes. Reproducibility therefore depends on the SHA256-pinned scripts in this package and the source artifacts named in the manifest. After rebuilding, run `python scripts/audit_freefall_contract.py <kit-path>`.
