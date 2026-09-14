from __future__ import annotations
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from figure_style import *
INPUT_RED = np.asarray([235, 48, 48], dtype=float) / 255
INPUT_BLUE = np.asarray([48, 96, 235], dtype=float) / 255
INPUT_HUES = tuple((1-u)*INPUT_RED + u*INPUT_BLUE for u in np.linspace(0,1,11))

def hue_index(row: dict) -> int:
    return int(str(row["color_label"])[3:])


def grouped(rows: list[dict]) -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        result[str(row["trajectory_id"])][hue_index(row)] = row
    return dict(result)


def clustered_curve(rows: list[dict], band: str, value_fn) -> np.ndarray:
    by_trajectory = grouped([row for row in rows if row["true_band"] == band])
    trajectory_ids = sorted(by_trajectory)
    values = np.asarray(
        [[value_fn(by_trajectory[trajectory_id][u]) for u in range(11)] for trajectory_id in trajectory_ids],
        dtype=float,
    )
    return np.nanmean(values, axis=0)


def panel_label(
    ax, letter: str, title: str, *, title_x: float = 0.060, title_size: float = 9.2
) -> None:
    ax.text(0.0, 1.00, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.2, fontweight="bold", color=PRIMARY_INK)
    ax.text(title_x, 1.00, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=title_size, fontweight="semibold", color=PRIMARY_INK)


def draw_hue_axis(ax, *, y: float = -0.072) -> None:
    x = np.linspace(0, 1, 11)
    ax.plot([0, 1], [y, y], transform=ax.get_xaxis_transform(), color=GRID, lw=0.65, clip_on=False)
    ax.scatter(x, np.full_like(x, y), transform=ax.get_xaxis_transform(), s=13, c=INPUT_HUES,
               edgecolors=WHITE, linewidths=0.25, zorder=8, clip_on=False)


def deterministic_jitter(row: dict) -> float:
    token = f"{row['trajectory_id']}:{row['color_label']}".encode("utf-8")
    value = int(hashlib.sha1(token).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    return (value - 0.5) * 0.042


def draw_scatter(ax, rows: list[dict], band: str, *, show_ylabel: bool) -> None:
    ax.axhspan(2.2, 3.0, color=SLOW_RED_LIGHT, alpha=0.34, linewidth=0)
    ax.axhspan(3.0, 5.2, color=AMBIG_PURPLE_LIGHT, alpha=0.16, linewidth=0)
    ax.axhspan(5.2, 6.4, color=FAST_BLUE_LIGHT, alpha=0.34, linewidth=0)
    subset = [row for row in rows if row["true_band"] == band]
    valid = [row for row in subset if row.get("valid")]
    invalid = [row for row in subset if not row.get("valid")]
    x = np.asarray([hue_index(row) / 10.0 + deterministic_jitter(row) for row in valid])
    y = np.asarray([float(row["omega_hat_free"]) for row in valid])
    colors = np.asarray([row["mean_detected_rgb"] for row in valid], dtype=float) / 255.0
    ax.scatter(x, y, s=10.5, c=np.clip(colors, 0, 1), alpha=0.84, edgecolors=WHITE,
               linewidths=0.20, zorder=5)
    if invalid:
        invalid_x = [hue_index(row) / 10.0 + deterministic_jitter(row) for row in invalid]
        invalid_y = [float(row["omega_hat_free"]) for row in invalid]
        ax.scatter(invalid_x, invalid_y, s=7.0, marker="x", color=INACTIVE, linewidths=0.55,
                   alpha=0.9, zorder=4)
    ax.set_xlim(-0.055, 1.055)
    ax.set_ylim(1.1, 6.75)
    ax.set_xticks([0, 0.5, 1], ["", "", ""])
    ax.set_yticks([2, 3, 4, 5, 6])
    if show_ylabel:
        ax.set_ylabel(r"decoded fitted frequency $\hat\omega$", labelpad=2)
    else:
        ax.tick_params(labelleft=False)
        ax.spines["left"].set_visible(False)
    style_axes(ax)
    draw_hue_axis(ax)
    ax.set_title(f"{band} observed motion", fontsize=8.0, fontweight="semibold",
                 color=PRIMARY_INK, pad=4)
    # Put each band label in a genuinely sparse region rather than laying text
    # over the corresponding point cloud.  All three are deliberately neutral:
    # the chromatic encodings remain reserved for measured future appearance.
    if band == "slow":
        ax.text(0.035, 0.895, "fast mode", transform=ax.transAxes, ha="left", va="center",
                fontsize=6.6, color=SECONDARY_TEXT)
        ax.text(0.035, 0.565, "between modes", transform=ax.transAxes, ha="left", va="center",
                fontsize=6.4, color=SECONDARY_TEXT)
    else:
        ax.text(0.965, 0.245, "slow mode", transform=ax.transAxes, ha="right", va="center",
                fontsize=6.6, color=SECONDARY_TEXT)


def draw_history_axis(ax, short_rows: list[dict], long_rows: list[dict], band: str,
                      *, show_ylabel: bool) -> None:
    u = np.linspace(0, 1, 11)
    value = lambda row: float(row.get("valid") and row.get("route_label") == "physics_frequency")
    short_mean = clustered_curve(short_rows, band, value)
    long_mean = clustered_curve(long_rows, band, value)
    ax.plot(u, short_mean, color=SECONDARY_TEXT, lw=1.2, ls=(0, (3, 2)), marker="o",
            markersize=2.1, markeredgewidth=0)
    ax.plot(u, long_mean, color=PRIMARY_INK, lw=2.0, marker="o", markersize=2.2,
            markeredgewidth=0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1], ["", "", ""])
    ax.set_yticks([0, 0.5, 1])
    if show_ylabel:
        ax.set_ylabel("physics-follow rate", labelpad=1)
    else:
        ax.tick_params(labelleft=False)
        ax.spines["left"].set_visible(False)
    style_axes(ax)
    draw_hue_axis(ax, y=-0.115)
    ax.set_title(f"{band} observed motion", fontsize=7.2, fontweight="semibold", color=PRIMARY_INK,
                 pad=4)


def draw_phase_axis(ax, rows: list[dict], direction: str, summary: dict,
                    phase_map, phase_norm) -> None:
    points = np.asarray([[row["direction_z"][1], row["direction_z"][2]] for row in rows])
    theta = np.asarray([row["theta_boundary"] for row in rows])
    ax.scatter(points[:, 0], points[:, 1], c=theta, cmap=phase_map,
               norm=phase_norm, s=22, edgecolor=DARK_SECONDARY, linewidth=0.34,
               alpha=0.96, zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    ax.set_title(f"target {direction}", fontsize=8.0, fontweight="normal", pad=3)
    ax.set_xlabel("phase coordinate 1", fontsize=6.7, labelpad=1)
    ax.set_ylabel("phase coordinate 2", fontsize=6.7, labelpad=1)
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax)


def panel_title(ax, letter: str, title: str, *, size: float = 7.7) -> None:
    ax.text(0.0, 1.055, letter, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.085, 1.055, title, transform=ax.transAxes, fontsize=size,
            fontweight="semibold", ha="left", va="bottom", linespacing=1.0)

