# Models

| Model | Parameters | Training | Main use |
|---|---:|---|---|
| Spring Short, seeds 3407/3408/3409 | 488,158,912 | 50K, 8 visible motion frames | Behavior, shared edits, downstream interventions |
| Spring Long, seeds 3407/3408/3409 | 488,158,912 | 50K, 65 visible motion frames | History comparison |
| Spring Short longitudinal ensemble | 488,158,912 | 15 seeds, 5K/10K/20K/50K/80K/100K | Training-time behavior and writability |
| Wan2.1 VAE | Frozen upstream model | Not trained here | Pixel/latent conversion |

The six core DiT checkpoints occupy about 5.46 GiB. Controller bundles contain
four float32 basis vectors across 20 flow-matching calls, plus phase coefficients.
The functional sites for Short seeds 3407/3408/3409 are blocks 3/6/4 respectively.

These are synthetic spring-video research models. They are not general video
generators or validated physical simulators. Color and motion are correlated in
training. A successful motion edit may also change generated appearance.

The registry identifies model variants and file sizes. Download instructions
are in the repository's `CHECKPOINTS.md`; the model repository is private
until publication is approved. The full 15-seed training trajectory weights
are not part of this release.

The pretrained Wan curriculum experiments use separate 1.3B checkpoints and are
not interchangeable with these 488M models. Their summary tables are included,
but this first runner targets the Spring-from-scratch models only.
