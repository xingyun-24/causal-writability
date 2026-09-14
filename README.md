# Causal Writability in Video Models

Materials for **A Chosen Future Can Still Be Rewritten: Causal Writability in
Video Models**. This is the project's independent repository, currently private
while the authors prepare the public release.

## Start Here

- [Paper and LaTeX sources](paper/)
- [Core model downloads](CHECKPOINTS.md)
- [Spring code and reproduction](reproduction/spring/)
- [Pendulum code and reproduction](reproduction/pendulum/)
- [Free Fall code and reproduction](reproduction/freefall/)
- [Project-page source and local previews](website/)

Sixteen selected checkpoints are hosted in a private Hugging Face repository:
six Spring, two Pendulum, one Free Fall, and seven pretrained-Wan checkpoints.
The full 15-seed training-history ensemble is not included.

The three task runtimes use the same Python package name, `sshv2`. Install each
task in a separate environment and follow its README.

Free Fall's dense PCA basis and archived decoded frames are hosted alongside
the weights, rather than stored as Git LFS placeholders. Follow
[the download instructions](ARTIFACTS.md) before running those workflows.

Spring includes tested core generation and intervention workflows. Pendulum
Short/Long loading, generation, matched editing and evaluation have passed a
small GPU test; this does not stand in for a full held-out Top-4 rerun. Free
Fall includes the final train-only producer and saved results, plus tested B1
activation extraction and project-page projection code. Plotting existing
results does not require retraining the models.

Original project code uses [MIT](LICENSE). Third-party licenses and model terms
are retained. Public publication and deployment remain pending. The previous
collaborator integration branch is retained separately; new release work belongs
in this repository. No GitHub Pages deployment is enabled.
