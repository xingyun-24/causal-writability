#!/usr/bin/env python3
"""Rebuild Pendulum appendix figures from the exact active main-panel grammar.

The plotting primitives, typography, palette, axes, line styles, and native
5.5-inch canvas are imported from the active Figure 2/3/4/5 builders.  This
file only adapts Pendulum release-table fields to those contracts.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
from PIL import Image


HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent
PENDULUM = FIGURES / "pendulum"
sys.path.insert(0, str(FIGURES))
sys.path.insert(0, str(HERE))

import figure_style as fs  # noqa: E402
import plot_helpers as f2  # noqa: E402
import plot_helpers as f3  # noqa: E402
import plot_helpers as f5  # noqa: E402


def read_csv(name: str) -> list[dict[str, str]]:
    with (PENDULUM / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str):
    return json.loads((PENDULUM / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    metadata = {"Creator": "matplotlib", "CreationDate": None, "ModDate": None}
    fig.savefig(PENDULUM / f"{stem}.pdf", metadata=metadata)
    fig.savefig(PENDULUM / f"{stem}.svg")
    fig.savefig(PENDULUM / f"{stem}.png", dpi=600)
    plt.close(fig)


def adapt_behavior() -> tuple[list[dict], list[dict]]:
    adapted: list[dict] = []
    for row in read_csv("behavior_rollouts.csv"):
        omega = float(row["omega_hat"])
        band = "slow" if row["history_band"] == "low" else "fast"
        in_slow = 2.2 <= omega <= 3.0
        in_fast = 5.2 <= omega <= 6.4
        if (band == "slow" and in_slow) or (band == "fast" and in_fast):
            route = "physics_frequency"
        elif (band == "slow" and in_fast) or (band == "fast" and in_slow):
            route = "opposite_band_frequency"
        elif 3.0 < omega < 5.2:
            route = "compromise_frequency"
        else:
            route = "off_frequency_family"
        hue = int(round(float(row["cue_u"]) * 10))
        adapted.append(
            {
                "trajectory_id": row["physical_state_id"],
                "sample_id": row["sample_id"],
                "true_band": band,
                "color_label": f"rgb{hue}",
                "omega_true": float(row["omega_true"]),
                "omega_hat_free": omega,
                "valid": row["frequency_valid"] == "True",
                "route_label": route,
                "mean_detected_rgb": [
                    float(row["appearance_r"]),
                    float(row["appearance_g"]),
                    float(row["appearance_b"]),
                ],
                "history_regime": row["history_regime"],
            }
        )
    short = [row for row in adapted if row["history_regime"] == "short"]
    long = [row for row in adapted if row["history_regime"] == "long"]
    if len(short) != 1408 or len(long) != 1408:
        raise ValueError("Pendulum behavior adapter requires 1,408 Short and Long rows")
    return short, long


def pendulum_protocol_header(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    f2.panel_label(ax, "a", "Decoded solution landscape", title_x=0.035)
    ax.text(0.42, 0.69, r"$128\times11$", ha="right", va="center",
            fontsize=7.5, color=fs.DARK_SECONDARY, fontweight="medium")
    ax.annotate("", xy=(0.465, 0.69), xytext=(0.435, 0.69), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=fs.SECONDARY_TEXT, lw=0.55))
    ax.text(0.48, 0.69, "1,408 decoded futures", ha="left", va="center",
            fontsize=7.5, color=fs.DARK_SECONDARY)
    ax.annotate("", xy=(0.715, 0.69), xytext=(0.685, 0.69), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=fs.SECONDARY_TEXT, lw=0.55))
    ax.text(0.73, 0.69, r"read out $\hat\omega$ + RGB", ha="left", va="center",
            fontsize=7.5, color=fs.DARK_SECONDARY)
    for x, color in zip((0.725, 0.742, 0.759),
                        (fs.HUE_PALETTE_11[0], fs.HUE_PALETTE_11[5], fs.HUE_PALETTE_11[10])):
        ax.add_patch(Circle((x, 0.23), 0.0062, transform=ax.transAxes,
                            facecolor=color, edgecolor=fs.WHITE, lw=0.25))
    ax.text(0.995, 0.23, "point color = future appearance", ha="right",
            va="center", fontsize=7.2, color=fs.DARK_SECONDARY,
            fontweight="medium")


def pendulum_history_header(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.76, "b", transform=ax.transAxes, ha="left", va="center",
            fontsize=10.2, fontweight="bold", color=fs.PRIMARY_INK)
    ax.text(0.035, 0.76, "Visible motion changes which future is selected",
            transform=ax.transAxes, ha="left", va="center", fontsize=8.3,
            fontweight="semibold", color=fs.PRIMARY_INK)
    ax.plot([0.715, 0.750], [0.76, 0.76], color=fs.PRIMARY_INK, lw=2.0,
            transform=ax.transAxes)
    ax.text(0.758, 0.76, "Long", transform=ax.transAxes, ha="left",
            va="center", fontsize=6.3, color=fs.PRIMARY_INK)
    ax.plot([0.855, 0.890], [0.76, 0.76], color=fs.SECONDARY_TEXT, lw=1.2,
            ls=(0, (3, 2)), transform=ax.transAxes)
    ax.text(0.898, 0.76, "Short", transform=ax.transAxes, ha="left",
            va="center", fontsize=6.3, color=fs.DARK_SECONDARY)


def build_behavior() -> None:
    short, long = adapt_behavior()
    fs.apply_paper_style()
    fig = plt.figure(figsize=(5.50, 3.48))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.72, 0.92),
                             left=0.075, right=0.985, top=0.97,
                             bottom=0.075, hspace=0.18)
    top = outer[0].subgridspec(2, 2, height_ratios=(0.14, 1.0),
                               hspace=0.12, wspace=0.16)
    header = fig.add_subplot(top[0, :])
    slow = fig.add_subplot(top[1, 0])
    fast = fig.add_subplot(top[1, 1], sharey=slow)
    pendulum_protocol_header(header)
    f2.draw_scatter(slow, short, "slow", show_ylabel=True)
    f2.draw_scatter(fast, short, "fast", show_ylabel=False)
    p0, p1 = slow.get_position(), fast.get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.032, r"input cue $u$",
             ha="center", va="center", fontsize=7.6, color=fs.DARK_SECONDARY)

    bottom = outer[1].subgridspec(2, 2, height_ratios=(0.28, 1.0),
                                  hspace=0.06, wspace=0.18)
    h = fig.add_subplot(bottom[0, :])
    slow_h = fig.add_subplot(bottom[1, 0])
    fast_h = fig.add_subplot(bottom[1, 1], sharey=slow_h)
    pendulum_history_header(h)
    f2.draw_history_axis(slow_h, short, long, "slow", show_ylabel=True)
    f2.draw_history_axis(fast_h, short, long, "fast", show_ylabel=False)
    save(fig, "pendulum_behavior")


def geometry_rows() -> tuple[dict[str, list[dict]], dict]:
    rows = read_csv("coordinate_predictions.csv")
    summary = read_json("coordinate_summary.json")
    heldout = {"fast": [], "slow": []}
    for row in rows:
        if row["split"] != "heldout":
            continue
        direction = "fast" if row["target_label"] == "high" else "slow"
        heldout[direction].append(
            {
                "direction_z": [
                    0.0,
                    float(row["phase_coordinate_1"]),
                    float(row["phase_coordinate_2"]),
                ],
                "theta_boundary": float(row["boundary_phase"]),
            }
        )
    return heldout, summary


def pendulum_rank_inset(ax) -> None:
    rows = read_csv("decoded_recovery.csv")
    by_receiver: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_receiver[row["receiver_id"]][row["condition"]] = row
    pairs = []
    for receiver in sorted(by_receiver):
        records = by_receiver[receiver]
        if "full_matched" not in records or "top4_oracle" not in records:
            continue
        a, b = records["full_matched"], records["top4_oracle"]
        if a["valid"] == "True" and b["valid"] == "True":
            pairs.append((float(a["normalized_recovery"]),
                          float(b["normalized_recovery"])))
    full = np.asarray([pair[0] for pair in pairs])
    top4 = np.asarray([pair[1] for pair in pairs])
    for full_i, top4_i in pairs:
        ax.plot([0, 1], [full_i, top4_i], color=fs.SECONDARY_TEXT,
                linewidth=0.5, alpha=0.10, zorder=1)
    ax.scatter(np.zeros_like(full), full, s=11, facecolor=fs.WHITE,
               edgecolor=fs.DARK_SECONDARY, linewidth=0.45, zorder=2)
    ax.scatter(np.ones_like(top4), top4, s=14, marker="D",
               facecolor=fs.CAUSAL_TEAL_LIGHT, edgecolor=fs.CAUSAL_TEAL,
               linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=fs.DARK_SECONDARY, linewidth=0.8,
               linestyle=(0, (3, 2)), zorder=0)
    ax.text(0.50, 1.045, "aligned future  ·  $R=1$",
            transform=ax.transAxes, fontsize=5.7, color=fs.DARK_SECONDARY,
            ha="center", va="bottom", clip_on=False)
    low = min(float(full.min()), float(top4.min())) - 0.02
    high = max(1.025, max(float(full.max()), float(top4.max())) + 0.01)
    ax.set_xlim(-0.34, 1.34)
    ax.set_ylim(low, high)
    ax.set_xticks([0, 1], ["Full write", "4-D write"])
    ax.set_ylabel("decoded recovery $R$", fontsize=6.2, labelpad=1)
    ax.tick_params(labelsize=5.8, length=2, pad=1)
    ax.set_box_aspect(1.0)
    fs.style_axes(ax, grid_axis="y")


def build_geometry() -> None:
    heldout, _summary = geometry_rows()
    fs.apply_paper_style()
    fig = plt.figure(figsize=(5.50, 2.30))
    grid = fig.add_gridspec(2, 1, height_ratios=(0.105, 0.895), hspace=0.015)
    header = fig.add_subplot(grid[0, 0])
    header.axis("off")
    header.set(xlim=(0, 1), ylim=(0, 1))
    header.text(0.0, 0.55, "a", fontsize=10.2, fontweight="bold",
                ha="left", va="center")
    header.text(0.055, 0.55, "Position and velocity organize the write",
                fontsize=8.9, fontweight="semibold", ha="left", va="center")
    plots = grid[1, 0].subgridspec(
        1, 4, width_ratios=(1.0, 1.0, 0.25, 0.78), wspace=0.16
    )
    fast = fig.add_subplot(plots[0, 0])
    slow = fig.add_subplot(plots[0, 1])
    key_container = fig.add_subplot(plots[0, 2])
    key_container.axis("off")
    rank = fig.add_subplot(plots[0, 3])
    phase_map = LinearSegmentedColormap.from_list(
        "paper_phase", fs.PHASE_CYCLIC_PALETTE, N=256
    )
    phase_norm = Normalize(0, 2 * np.pi)
    f3.draw_phase_axis(fast, heldout["fast"], "fast", {}, phase_map, phase_norm)
    f3.draw_phase_axis(slow, heldout["slow"], "slow", {}, phase_map, phase_norm)
    pendulum_rank_inset(rank)
    key = key_container.inset_axes([-0.52, 0.32, 0.82, 0.36])
    key.axis("off")
    phi = np.linspace(0, 2 * np.pi, 80)
    for j in range(len(phi) - 1):
        key.plot([np.cos(phi[j]), np.cos(phi[j + 1])],
                 [np.sin(phi[j]), np.sin(phi[j + 1])],
                 color=phase_map(phase_norm(phi[j])), linewidth=3.0,
                 solid_capstyle="round")
    key.text(0, 0, "θ*", ha="center", va="center",
             fontsize=7.0, color=fs.DARK_SECONDARY)
    key.text(1.12, 0, r"$0/2\pi$", fontsize=6.0, va="center",
             color=fs.SECONDARY_TEXT)
    key.text(0, 1.18, r"$\pi/2$", fontsize=6.0, ha="center",
             color=fs.SECONDARY_TEXT)
    key.text(-1.12, 0, r"$\pi$", fontsize=6.0, ha="right", va="center",
             color=fs.SECONDARY_TEXT)
    key.text(0, -1.23, r"$3\pi/2$", fontsize=6.0, ha="center", va="top",
             color=fs.SECONDARY_TEXT)
    key.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), aspect="equal")
    fig.subplots_adjust(left=0.04, right=0.985, top=0.98, bottom=0.06)
    save(fig, "pendulum_state_geometry")


def recovery_medians() -> dict[str, dict[str, float]]:
    rows = read_csv("decoded_recovery.csv")
    conditions = ("full_matched", "top4_oracle", "fit_only_predicted")
    result: dict[str, dict[str, float]] = {}
    for target in ("high", "low"):
        result[target] = {}
        for condition in conditions:
            values = [
                float(row["normalized_recovery"])
                for row in rows
                if row["target_label"] == target
                and row["condition"] == condition
                and row["valid"] == "True"
            ]
            result[target][condition] = float(np.median(values))
    return result


def controller_axis(ax) -> None:
    medians = recovery_medians()
    x = np.arange(3, dtype=float)
    conditions = ("full_matched", "top4_oracle", "fit_only_predicted")
    labels = ["Full", "Top-4", "Predicted"]
    for target, color, marker, offset in (
        ("high", fs.FAST_BLUE, "o", -0.035),
        ("low", fs.SLOW_RED, "D", 0.035),
    ):
        y = [medians[target][condition] for condition in conditions]
        ax.plot(x + offset, y, color=color, lw=1.25, marker=marker, ms=3.8,
                markeredgecolor=fs.WHITE, markeredgewidth=0.45)
        ax.text(2.14, y[-1], "fast" if target == "high" else "slow",
                color=color, fontsize=5.4, ha="left", va="center")
    ax.axhspan(0.75, 1.025, color=fs.CAUSAL_TEAL_LIGHT, alpha=0.34, zorder=-2)
    ax.axhline(1.0, color=fs.SECONDARY_TEXT, lw=0.65,
               ls=(0, (2, 2)), zorder=-1)
    ax.set(xlim=(-0.35, 2.48), ylim=(0.89, 1.015), xticks=x,
           xticklabels=labels, ylabel="Median frequency recovery")
    ax.tick_params(axis="x", pad=1)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
             rotation_mode="anchor")
    fs.style_axes(ax, grid_axis="y")
    f5.panel_title(ax, "a", "Predicted write approaches\nmatched Top-4", size=6.8)


def frame_strip(ax, folder: str, title: str, *, edited: bool) -> None:
    paths = sorted((PENDULUM / "frames" / folder).glob("*.png"))
    strip = np.concatenate(
        [np.asarray(Image.open(path).convert("RGB")) for path in paths], axis=1
    )
    ax.imshow(strip)
    ax.set_title(title, loc="left", fontsize=7.0, fontweight="semibold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(fs.CAUSAL_TEAL if edited else fs.CARD_BORDER)
        spine.set_linewidth(0.8)


def build_recovery() -> None:
    fs.apply_paper_style()
    fig = plt.figure(figsize=(5.50, 2.15))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.42, 0.58), wspace=0.30)
    left = fig.add_subplot(grid[0, 0])
    controller_axis(left)
    right = grid[0, 1].subgridspec(3, 1, height_ratios=(0.16, 0.42, 0.42), hspace=0.48)
    header = fig.add_subplot(right[0, 0])
    header.axis("off")
    header.text(0.0, 0.55, "b", transform=header.transAxes, fontsize=9.2,
                fontweight="bold", ha="left", va="center")
    header.text(0.085, 0.55, "The predicted write changes decoded motion",
                transform=header.transAxes, fontsize=6.8, fontweight="semibold",
                ha="left", va="center")
    natural = fig.add_subplot(right[1, 0])
    edited = fig.add_subplot(right[2, 0])
    frame_strip(natural, "natural_conflict", "Natural conflict", edited=False)
    frame_strip(edited, "state_edit", "Fit-only write", edited=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.25)
    save(fig, "pendulum_decoded_recovery")


def build_writeability() -> None:
    summary = read_json("layer_summary.json")
    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.50, 2.18), sharey=True)
    for ax, direction in zip(axes, ("low", "high")):
        record = summary["directions"][direction]
        x = np.arange(31)
        short = np.asarray(record["short"]["raw_success_curve"], dtype=float)
        long = np.asarray(record["long"]["raw_success_curve"], dtype=float)
        ax.fill_between(x, 0, long, color=fs.CAUSAL_TEAL_LIGHT,
                        alpha=0.36, zorder=0)
        ax.plot(x, short, color=fs.SECONDARY_TEXT, linestyle=(0, (3, 2)),
                linewidth=1.7, zorder=2)
        ax.plot(x, long, color=fs.CAUSAL_TEAL, linewidth=2.1, zorder=3)
        short_l50 = float(record["short"]["L50"])
        long_l50 = float(record["long"]["L50"])
        ax.axvline(short_l50, color=fs.COMMIT_AMBER, linewidth=0.75,
                   linestyle=(0, (3, 2)), alpha=0.65)
        ax.axvline(long_l50, color=fs.COMMIT_AMBER, linewidth=0.9, alpha=0.9)
        ax.text(0.04, 0.92, "Long", transform=ax.transAxes,
                color=fs.CAUSAL_TEAL, fontsize=7.2, ha="left")
        ax.text(0.04, 0.81, "Short", transform=ax.transAxes,
                color=fs.SECONDARY_TEXT, fontsize=7.2, ha="left")
        ax.set(xlim=(0, 30), ylim=(-0.02, 1.04),
               xlabel="write site (residual location)",
               title=f"target {direction}")
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        if ax is axes[0]:
            ax.set_ylabel(r"fraction with $0.75 < R < 1.25$")
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)
        fs.style_axes(ax, grid_axis="y")

    fig.text(0.09, 0.97, "a", fontsize=10.2, fontweight="bold",
             ha="left", va="top")
    fig.text(0.14, 0.97, "Visible motion delays Pendulum commitment",
             fontsize=8.9, fontweight="semibold", ha="left", va="top")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.22, wspace=0.18)
    save(fig, "pendulum_writeability")


def main() -> None:
    # Repeat the shared release-critical settings locally for static QA.  The
    # values are identical to figure_style.apply_paper_style().
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    build_behavior()
    build_geometry()
    build_recovery()
    build_writeability()


if __name__ == "__main__":
    main()
