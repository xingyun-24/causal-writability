import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


base = Path(__file__).resolve().parents[1] / "results" / "hist16-step100000-eval"
summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))["summary"]
conditions = ["aligned", "conflict"]
metrics = [
    ("E3_accuracy", "E3"),
    ("E0_accuracy", "E0"),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
for ax, mode in zip(axes, ("accuracy", "mae")):
    x = np.arange(len(conditions))
    width = 0.34
    all_values = []
    if mode == "mae":
        for key, _ in metrics:
            mae_key = key.replace("_accuracy", "_mae")
            all_values.extend(summary[c][mae_key] for c in conditions)
    for i, (key, label) in enumerate(metrics):
        if mode == "accuracy":
            vals = [100 * summary[c][key] for c in conditions]
            ylabel, title, ylim = "Accuracy (%)", "16-frame accuracy (100k)", (0, 105)
            labels = [f"{v:.2f}%" for v in vals]
        else:
            mae_key = key.replace("_accuracy", "_mae")
            vals = [summary[c][mae_key] for c in conditions]
            ylabel, title = "MAE of g", "16-frame gravity error (100k)"
            ylim = (0, max(all_values) * 1.25)
            labels = [f"{v:.4f}" for v in vals]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label)
        for bar, text in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1] * 0.025,
                    text, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["Aligned", "Conflict"])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Projectile Gravity V3 fixed-position | hist16 step-100000", fontsize=11)
fig.tight_layout()
out = base / "hist16_step100000_accuracy_comparison.png"
fig.savefig(out, bbox_inches="tight")
print(out)
