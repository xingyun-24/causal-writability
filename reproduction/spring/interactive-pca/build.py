"""Use YuanMan's original Plotly layout for the newly measured Spring projections."""
import argparse
from collections import Counter
import json
from pathlib import Path


def page(traces, title, legend, library):
    layout = {
        'title': title, 'template': 'plotly_white',
        'scene': {'xaxis': {'title': 'Projection on residual PC1'},
                  'yaxis': {'title': 'Projection on residual PC2'},
                  'zaxis': {'title': 'Projection on residual PC3'}, 'aspectmode': 'auto'},
        'legend': {'title': {'text': legend}, 'orientation': 'h',
                   'x': 0, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        'margin': {'l': 0, 'r': 0, 't': 55, 'b': 0},
    }
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + title + '</title><script>' + library + '</script></head>\n'
            '<body style="margin:0;background:#fff"><div id="plot" style="width:100vw;height:100vh"></div>\n'
            '<script>\nconst traces = ' + json.dumps(traces, ensure_ascii=True) + ';\n'
            'const layout = ' + json.dumps(layout, ensure_ascii=True) + ';\n'
            'if (innerWidth < 600) { layout.height=Math.min(innerHeight,Math.round(innerWidth*1.3+80)); document.getElementById("plot").style.height=layout.height+"px"; '
            'layout.scene.camera = {eye:{x:2.7,y:2.7,z:2.7}}; layout.margin.t=80; layout.margin.l=24; layout.margin.r=8; '
            'layout.scene.xaxis.title="PC1"; layout.scene.yaxis.title="PC2"; layout.scene.zaxis.title=""; '
            'layout.annotations=[{text:"PC3",x:0,y:0.67,xref:"paper",yref:"paper",showarrow:false,xanchor:"left",yanchor:"bottom",font:{size:12,color:"#444"}}]; }\n'
            "Plotly.newPlot('plot', traces, layout, {responsive:true, displaylogo:false, scrollZoom:true});\n"
            '</script></body></html>\n')


def trace(rows, name, symbol, minimum, maximum, colorbar, difference=False):
    return {
        'type': 'scatter3d', 'mode': 'markers', 'name': name,
        'x': [r['pc1'] for r in rows], 'y': [r['pc2'] for r in rows], 'z': [r['pc3'] for r in rows],
        'customdata': [[r['pair_id'], r['input_colour'], r['true_band'], r['omega_true'], r['omega_hat']] for r in rows],
        'hovertemplate': ('pair=%{customdata[0]}<br>input colour=%{customdata[1]}<br>'
                          'observed motion=%{customdata[2]}<br>true omega=%{customdata[3]:.4f}<br>'
                          + ('aligned omega' if difference else 'decoded omega')
                          + '=%{customdata[4]:.4f}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}'
                          + '<extra>' + name + '</extra>'),
        'marker': {'symbol': symbol, 'size': 6, 'opacity': .86,
                   'color': [r['omega_hat'] for r in rows], 'colorscale': 'Viridis',
                   'cmin': minimum, 'cmax': maximum,
                   'colorbar': {'title': 'Aligned \u03c9<br>(rad/s)' if difference else 'Decoded \u03c9<br>(rad/s)'} if colorbar else None,
                   'line': {'color': '#202124', 'width': .35}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted((args.root / 'records').glob('*.json'))]
    assert len(records) == 128
    points = [p for r in records for p in r['points']]
    assert len(points) == 256 and all(p['valid'] for p in points)
    groups = Counter((p['input_colour'], p['true_band']) for p in points)
    assert len(groups) == 4 and set(groups.values()) == {64}
    difference = []
    for record in records:
        aligned = record['points'][0]
        assert aligned['condition'] == 'aligned'
        difference.append({**aligned, 'condition': 'difference',
                           **dict(zip(['pc1', 'pc2', 'pc3'], record['difference_pc']))})
    library = (args.root / 'plotly-2.35.2.min.js').read_text()
    assert 'plotly.js v2.35.2' in library[:300]
    low, high = min(p['omega_hat'] for p in points), max(p['omega_hat'] for p in points)
    raw_traces = [trace([p for p in points if p['condition'] == condition], condition, symbol, low, high, condition == 'aligned')
                  for condition, symbol in [('aligned', 'circle'), ('conflict', 'diamond')]]
    (args.root / 'spring_raw_projection_interactive.html').write_text(page(raw_traces, 'Spring: raw PCA projections', 'Condition', library))
    low, high = min(p['omega_hat'] for p in difference), max(p['omega_hat'] for p in difference)
    diff_traces = [trace([p for p in difference if p['true_band'] == band], 'Target ' + band, symbol, low, high, band == 'fast', True)
                   for band, symbol in [('fast', 'circle'), ('slow', 'diamond')]]
    (args.root / 'spring_matched_difference_interactive.html').write_text(page(diff_traces, 'Spring: matched-difference PCA', 'Target direction', library))
    metadata = {'task': 'spring', 'model_seed': 3408, 'checkpoint_step': 50000, 'block_zero_based': 6,
                'basis': 'raw uncentered matched-difference PCA, first three of the frozen fit-only components',
                'fit_pairs': 128, 'heldout_pairs': 128, 'fm_representation': 'condition residuals concatenated over all 20 calls',
                'raw_activation_points': len(points), 'difference_points': len(difference),
                'raw_marker_color': 'newly decoded omega, not input RGB',
                'difference_marker_color': 'matched aligned rollout decoded omega',
                'renderer': 'YuanMan Plotly 2.35.2 template, bundled offline',
                'point_values_modified_for_appearance': False}
    (args.root / 'data.json').write_text(json.dumps({'metadata': metadata, 'raw_points': points, 'difference_points': difference}, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
