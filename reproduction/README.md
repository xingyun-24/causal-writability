# Reproduction

| Task | Entry point | Current scope |
|---|---|---|
| Spring | [spring/README.md](spring/README.md) | Training, natural generation, residual and controller interventions; core tests and GPU smoke |
| Pendulum | [pendulum/README.md](pendulum/README.md) | Final Short/Long configs, new helper modules, frozen split, Top-4 scripts; GPU generation/matched-edit smoke |
| Free Fall | [freefall/README.md](freefall/README.md) | Final hist32/100K train-only controller, result tables, input/residual generation and evaluation; B1 projection extraction |

Use separate Python environments because all three runtimes define `sshv2`.
Weights are listed in [CHECKPOINTS.md](../CHECKPOINTS.md). Basis tensors and input
manifests remain separate from model weights.

The paper's corrected tables and figures are retained. A plotting check is not
a new experiment, and a one-receiver smoke test is not the full held-out
evaluation. Original code uses MIT; third-party notices remain applicable.
