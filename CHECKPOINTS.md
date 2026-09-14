# Checkpoints

The model files are hosted at
[xingyun-24/causal-writability](https://huggingface.co/xingyun-24/causal-writability).
The files can be downloaded without signing in.

| Group | Files | Models |
|---|---:|---|
| Spring | 6 | Short/Long, seeds 3407/3408/3409, 50K |
| Pendulum | 2 | Short/Long, seed 3407, 50K |
| Free Fall | 1 | hist32, 100K |
| Pretrained Wan | 7 | Direct and Neutral-first at 2K/5K/10K; neutral-only 10K start |

All 16 files are uploaded, totaling about 29.41 GB. Individual links and sizes
are listed in [checkpoints.json](checkpoints.json). The 15-seed x 6-checkpoint
Spring ensemble and extra pretrained intermediate checkpoints are not uploaded.

```bash
hf download xingyun-24/causal-writability \
  spring/spring-short-3407-50k.safetensors --local-dir weights
hf download xingyun-24/causal-writability \
  --include 'pendulum/*' --local-dir weights
```

Paths are preserved, for example `weights/spring/spring-short-3407-50k.safetensors`.
The Spring CLI also supports `cw weights --model spring-short-3407-50k --out ...`
using the HF client.

The [frozen Wan2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth)
is downloaded from upstream rather than duplicated. PCA bases, controller
parameters and input manifests are separate assets described in each task's
README. These are task-specific research checkpoints, not interchangeable
generic video generators.
