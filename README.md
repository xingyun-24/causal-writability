# Causal Writability in Video Models

Materials for **A Chosen Future Can Still Be Rewritten: Causal Writability in
Video Models**.

## Start Here

- [Project page: videos and interactive PCA](https://xingyun-24.github.io/causal-writability/)
- [Paper and LaTeX sources](paper/)
- [Core model downloads](CHECKPOINTS.md)
- [Spring code and reproduction](reproduction/spring/)
- [Pendulum code and reproduction](reproduction/pendulum/)
- [Free Fall code and reproduction](reproduction/freefall/)
- [Project-page source and local previews](website/)

Sixteen selected checkpoints are hosted on Hugging Face:
six Spring, two Pendulum, one Free Fall, and seven pretrained-Wan checkpoints.
The full 15-seed training-history ensemble is not included.

The three task runtimes use the same Python package name, `sshv2`. Install each
task in a separate environment and follow its README.

Free Fall's dense PCA basis and archived decoded frames are hosted alongside
the weights, rather than stored as Git LFS placeholders. Follow
[the download instructions](ARTIFACTS.md) before running those workflows.
Spring's three frozen Short controllers are also downloadable. Pendulum instead
provides a fit-only basis reconstruction and controller-fitting workflow.

Spring includes tested core generation and intervention workflows. Pendulum
Short/Long loading, generation, matched editing and evaluation have passed a
small GPU test; this does not stand in for a full held-out Top-4 rerun. Free
Fall includes the final train-only producer and saved results, plus tested B1
activation extraction and project-page projection code. Plotting existing
results does not require retraining the models.

Original project code uses [MIT](LICENSE). Third-party licenses and model terms
are retained. The project page uses the same experimental assets and final
paper figures as this repository. The arXiv identifier will be added after
the submission is announced.
