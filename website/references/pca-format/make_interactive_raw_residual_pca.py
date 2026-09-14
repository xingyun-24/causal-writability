#!/usr/bin/env python3
"""Create a browser-based draggable 3D Plotly view of raw PCA projections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.projections.read_text(encoding="utf-8-sig"))
    traces = []
    for condition, marker in (("aligned", "circle"), ("conflict", "diamond")):
        selected = [row for row in rows if row["condition"] == condition]
        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "name": condition,
            "x": [row["pc1"] for row in selected],
            "y": [row["pc2"] for row in selected],
            "z": [row["pc3"] for row in selected],
            "customdata": [[row["pair_id"], row["g_E3"], row.get("detected_colour")] for row in selected],
            "hovertemplate": f"pair=%{{customdata[0]}}<br>g_E3=%{{customdata[1]:.6f}}<br>detected colour=%{{customdata[2]}}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<br>PC3=%{{z:.3f}}<extra>{condition}</extra>",
            "marker": {
                "symbol": marker,
                "size": 6,
                "opacity": 0.86,
                "color": [row["g_E3"] for row in selected],
                "colorscale": "Viridis",
                "cmin": min(row["g_E3"] for row in rows),
                "cmax": max(row["g_E3"] for row in rows),
                "colorbar": {"title": "Measured g_E3"} if condition == "aligned" else None,
                "line": {"color": "#202124", "width": 0.35},
            },
        })
    layout = {
        "title": "Raw aligned/conflict activations on residual PCA directions",
        "template": "plotly_white",
        "scene": {
            "xaxis": {"title": "Projection on residual PC1"},
            "yaxis": {"title": "Projection on residual PC2"},
            "zaxis": {"title": "Projection on residual PC3"},
            "aspectmode": "auto",
        },
        "legend": {"title": {"text": "Condition"}},
        "margin": {"l": 0, "r": 0, "t": 55, "b": 0},
    }
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raw residual PCA projection</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body style="margin:0;background:#fff"><div id="plot" style="width:100vw;height:100vh"></div>
<script>
const traces = %s;
const layout = %s;
Plotly.newPlot('plot', traces, layout, {responsive:true, displaylogo:false, scrollZoom:true});
</script></body></html>
""" % (json.dumps(traces, ensure_ascii=True), json.dumps(layout, ensure_ascii=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(json.dumps({"points": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
