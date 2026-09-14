# Evaluation Scope

The existing deterministic Spring tracker fits frequency to decoded future pixels.
Physics-following means a valid output in the observed-motion frequency band.
Normalized recovery R compares an intervention to the matched natural aligned and
conflict frequencies; it is not the same as frequency-band correctness.

- Figure 2c: each cue-by-motion-band point averages 32 histories.
- Figure 3d: each natural rate pools the complete 64 x 11 cue grid.
- Figure 4b: the per-checkpoint grid rate is conditional on validity, then averaged
  across training seeds; writability comes from separate checkpoint-local strict banks.
- Figure 5a: conflict endpoint or matched gray-input cohorts, 128 histories each.
- Figure 6c: 48 selection-clean fast-motion/red-cue failures, not the full grid.

The color-only training rule has a 50% reference on the balanced full grid and a
0% reference on conflicting endpoints. Neither is a universal chance threshold.

The new demo runner uses the shared strict bank. It tests the implementation and
produces per-video measurements; it is not a rerun of every population result.
