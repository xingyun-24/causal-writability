"""Authoritative plotting defaults for paper figures.

Display colors are not the renderer/input RGB values. Never recolor decoded frames.
Keep synchronized with FIGURE_STYLE_GUIDE_v1.md and figure_style.tex.
"""

import os

SLOW_RED = "#D66555"
FAST_BLUE = "#4F75B3"
AMBIG_PURPLE = "#9575B5"
CAUSAL_TEAL = "#3A9688"
COMMIT_AMBER = "#D19A45"

SLOW_RED_LIGHT = "#F3D6D1"
FAST_BLUE_LIGHT = "#D8E1F0"
AMBIG_PURPLE_LIGHT = "#E5DCEF"
CAUSAL_TEAL_LIGHT = "#D8ECE8"
COMMIT_AMBER_LIGHT = "#F2E4C9"

PRIMARY_INK = "#30343B"
SECONDARY_TEXT = "#899097"
DARK_SECONDARY = "#626970"
STRUCTURE_FILL = "#F1F0ED"
TARGET_FILL = "#E3E7E8"
WARM_CARD = "#F8F6F1"
INACTIVE = "#C8CBCB"
GRID = "#DDE0E2"
CARD_BORDER = "#D9D7D2"
WHITE = "#FFFFFF"

HUE_PALETTE_11 = (
    "#D66555",
    "#C96868",
    "#BC6B7B",
    "#AF6F8F",
    "#A272A2",
    "#9575B5",
    "#8775B5",
    "#7975B4",
    "#6B75B4",
    "#5D75B3",
    "#4F75B3",
)

# Phase is periodic, so its display ramp must meet at 0 and 2pi.  This tuple
# uses only the paper's approved teal and neutral colors.  It is reserved for
# physical phase and must not encode hue, outcome class, model identity, or
# intervention strength.
PHASE_CYCLIC_PALETTE = (
    CAUSAL_TEAL,
    CAUSAL_TEAL_LIGHT,
    WARM_CARD,
    SECONDARY_TEXT,
    PRIMARY_INK,
    SECONDARY_TEXT,
    WARM_CARD,
    CAUSAL_TEAL_LIGHT,
    CAUSAL_TEAL,
)


def apply_paper_style() -> None:
    """Apply final-size defaults without importing matplotlib at module import."""
    import matplotlib as mpl

    # Figure builders are release artifacts: identical inputs should not
    # produce spurious Git diffs.  Matplotlib otherwise randomizes SVG element
    # IDs and writes the wall-clock creation time into PDF metadata.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
            ],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "text.color": PRIMARY_INK,
            "axes.labelcolor": PRIMARY_INK,
            "axes.edgecolor": PRIMARY_INK,
            "xtick.color": DARK_SECONDARY,
            "ytick.color": DARK_SECONDARY,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
            "axes.grid": False,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "physics-shortcuts-paper-v1",
        }
    )


def style_axes(ax, *, grid_axis: str | None = None) -> None:
    """Apply the shared open-axis treatment to a Matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
