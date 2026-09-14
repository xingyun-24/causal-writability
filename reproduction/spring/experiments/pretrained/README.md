# Pretrained Wan entry points

These are the experiment producers for the selected pretrained checkpoints,
not a new set of model results. Run from `reproduction/spring` using its
environment and `PYTHONPATH=src`.

- `scripts/run_pretrained_wan_pca_causal_controller.py`: fit-only physical controller and decoded evaluation.
- `scripts/run_short_dual_direction_layer_scan.py`: direct-adaptation layer scans.
- `scripts/build_local_strict_pairs_from_hue.py`: checkpoint-local pair selection from stored behavior.
- `scripts/run_preregistered_b14_b15_confirmation.py`: specified-site scans on a frozen curriculum cohort.

```bash
export PYTHONPATH="$PWD/src"
python experiments/pretrained/scripts/run_pretrained_wan_pca_causal_controller.py --help
python experiments/pretrained/scripts/run_preregistered_b14_b15_confirmation.py --help
```

The configs preserve architecture and training settings but use relative
paths. Inference uses the explicit `--checkpoint` argument. Download the seven
selected pretrained weights using the root `CHECKPOINTS.md`. These models use
the zero-context pretrained Wan pathway; they are not the scratch-trained
488M Spring checkpoints.

The controller program additionally requires its frozen geometry and matched
input banks, supplied explicitly through its CLI. The local-layer program
requires the selected pair list and inputs. Small summary tables are under
`results/pretrained/`; these do not replace activation bases or input banks.
The seven uploaded models do not cover every intermediate checkpoint used
in the appendix. Full pretrained end-to-end reruns are not claimed here.

`wan-base-native.safetensors` is an upstream-base placeholder used only for
training initialization and is not one of the uploaded task checkpoints;
set it to the compatible upstream base before retraining. No new training
or additional candidate figures are published by including these scripts.
