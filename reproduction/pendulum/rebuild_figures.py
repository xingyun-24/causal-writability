"""Rebuild the corrected paper figures from the collected local tables."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'paper/scripts'))
import build_pendulum_exact_style as original


def fix_behavior(fig):
    header, slow, fast, history_header, slow_h, fast_h = fig.axes
    old_h = fig.get_figheight() * 72
    new_h = old_h + 30
    positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    fig.set_size_inches(5.5, new_h / 72)
    for ax, pos in positions.items():
        shift = 6 if ax in (header, slow, fast) else 0
        ax.set_position([pos.x0, (pos.y0 * old_h + shift) / new_h,
                         pos.width, pos.height * old_h / new_h])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, y * old_h / new_h))
    header.clear()
    header.set_axis_off()
    ink, secondary = original.fs.PRIMARY_INK, original.fs.DARK_SECONDARY

    def text_at(x, top, value, size, **kwargs):
        return fig.text(x / 396, 1 - top / new_h, value, fontsize=size,
                        va='top', color=kwargs.pop('color', ink), **kwargs)

    text_at(29.7, 6, 'a', 10.2, fontweight='bold')
    text_at(42.31, 6, 'Decoded solution landscape', 9.2, fontweight='semibold')
    text_at(128, 21, r'$128\times11$', 7.5, color=secondary)
    text_at(185, 21, '1,408 decoded futures', 7.5, color=secondary)
    text_at(284, 21, r'read out $\hat\omega$ + RGB', 7.5, color=secondary)
    for left, right in ((171, 181), (264, 279)):
        header.annotate('', xy=(right / 396, 1-25/new_h),
                        xytext=(left / 396, 1-25/new_h), xycoords=fig.transFigure,
                        arrowprops=dict(arrowstyle='-|>', color=original.fs.SECONDARY_TEXT, lw=.55))
    text_at(387.54, 32, 'point color = future appearance', 7.2,
            color=secondary, ha='right', fontweight='medium')
    for ax in (slow, fast, slow_h, fast_h):
        assert np.isclose(positions[ax].height * old_h, ax.get_position().height * new_h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE / 'runs/rebuilt')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for builder, stem in [('build_behavior', 'pendulum_behavior'),
                           ('build_geometry', 'pendulum_state_geometry'),
                           ('build_recovery', 'pendulum_decoded_recovery'),
                           ('build_writeability', 'pendulum_writeability')]:
        captured = []
        original.save = lambda fig, name: captured.append(fig)
        getattr(original, builder)()
        fig = captured[0]
        if builder == 'build_behavior':
            fix_behavior(fig)
        elif builder == 'build_recovery':
            ax = fig.axes[0]
            pos = ax.get_position()
            ax.set_position([pos.x0 + 6 / 396, pos.y0, pos.width, pos.height])
        elif builder == 'build_writeability':
            for ax in fig.axes:
                for label in ax.texts:
                    if label.get_text() in {'Long', 'Short'}:
                        label.set_transform(label.get_transform() + ScaledTranslation(0, -6/72, fig.dpi_scale_trans))
        matplotlib.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
        fig.savefig(args.out / (stem + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(args.out / (stem + '.svg'), metadata={'Date': None})
        plt.close(fig)
        print(stem, flush=True)


if __name__ == '__main__':
    main()
