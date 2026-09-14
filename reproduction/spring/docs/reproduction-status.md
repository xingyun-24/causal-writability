# Tested Scope

The release was installed as a wheel in a newly provisioned Python 3.12.14
environment, using the locked dependencies. Code, model files, VAE and controller
were copied into an independent working directory, without links to the original
experiment data. Network-dependent model lookup was disabled during execution.

Passed on a CUDA GPU:

- Small raw dataset generation and Short/Long VAE encoding.
- One optimizer step of the 488M model on the small encoded dataset.
- Natural video generation and decoded-video evaluation.
- Matched residual intervention using Short seed 3407 at block 3.
- Frozen phase-controller intervention using Short seed 3409 at block 4.
- Condition V-head 8 intervention at block 9, gain 8.

Six unit tests cover weight installation, unique model IDs, the frozen receiver
split and renderer, and the condition/future boundaries of the residual and
attention hooks. File-access tracing of the end-to-end smoke and controller runs
found no accesses to the original experiment directories outside the independent
working copy. This is observed file-access validation, not an OS-level mount sandbox.

For the one held-out test receiver, the seed-3407 paired write gave R=1.0168.
For seed 3409, the phase controller gave R=0.9734; the gain-8 V-head write
overshot to R=1.8238. These are implementation checks on one fixed example,
not population estimates or a replacement for the paper's dose-sweep results.

Not covered by this test: complete 50K/100K retraining, every checkpoint-local
strict bank, all appendix experiments, pretrained Wan curriculum reruns, or the
Pendulum/Free Fall pipelines. The authors selected MIT for their original
code; public trained-model hosting remains pending. Archived results are distinct
from newly generated test outputs.
