"""Adapt Leqian's measured coordinates to the shared YuanMan Plotly embeds."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plotly", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    assert len(data["endpoints"]) == 128 and len(data["differences"]) == 64
    library = args.plotly.read_text()
    assert "plotly.js v2.35.2" in library[:300]
    for difference, filename in ((False, "pendulum-project-pca.html"), (True, "pendulum-difference-pca.html")):
        points = data["differences" if difference else "endpoints"]
        key = "omega_aligned" if difference else "omega_hat"
        low, high = min(p[key] for p in points), max(p[key] for p in points)
        traces = []
        groups = [("high", "circle"), ("low", "diamond")] if difference else [("aligned", "circle"), ("conflict", "diamond")]
        for name, symbol in groups:
            selected = [p for p in points if p["target_label" if difference else "condition"] == name]
            traces.append({"type": "scatter3d", "mode": "markers", "name": name,
                           "x": [p["pc1"] for p in selected], "y": [p["pc2"] for p in selected], "z": [p["pc3"] for p in selected],
                           "customdata": [[p["receiver_id"], p["target_label"], p["omega_true"], p[key]] for p in selected],
                           "hovertemplate": "pair=%{customdata[0]}<br>target=%{customdata[1]}<br>true frequency=%{customdata[2]:.4f}<br>" + ("aligned" if difference else "decoded") + " frequency=%{customdata[3]:.4f}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra>held-out</extra>",
                           "marker": {"symbol": symbol, "size": 6, "opacity": .86, "color": [p[key] for p in selected],
                                      "colorscale": "Viridis", "cmin": low, "cmax": high, "showscale": not traces,
                                      "colorbar": {"title": "Aligned frequency" if difference else "Decoded frequency"},
                                      "line": {"color": "#202124", "width": .35}}})
        title = "Pendulum: matched-difference PCA" if difference else "Pendulum: raw PCA projections"
        layout = {"title": title, "template": "plotly_white",
                  "scene": {"xaxis": {"title": "Projection on residual PC1"}, "yaxis": {"title": "Projection on residual PC2"},
                            "zaxis": {"title": "Projection on residual PC3"}, "aspectmode": "auto"},
                  "legend": {"orientation": "h", "x": 0, "y": 1, "xanchor": "left", "yanchor": "top"},
                  "margin": {"l": 0, "r": 0, "t": 55, "b": 0}}
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                    '<title>' + title + '</title><script>' + library + '</script></head><body style="margin:0;background:#fff">'
                    '<div id="plot" style="width:100vw;height:100vh"></div><script>const traces=' + json.dumps(traces) + ';const layout=' + json.dumps(layout) + ';'
                    'if(innerWidth<600){layout.title={text:"Pendulum PCA",font:{size:15}};layout.scene.xaxis.title="PC1";layout.scene.yaxis.title="PC2";layout.scene.zaxis.title="PC3";layout.scene.camera={eye:{x:2,y:2,z:2}};layout.margin.t=75;}'
                    "Plotly.newPlot('plot',traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});</script></body></html>")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / filename).write_text(document)
    print("Exported 128 raw endpoints and 64 differences; original coordinates unchanged.")


if __name__ == "__main__":
    main()
