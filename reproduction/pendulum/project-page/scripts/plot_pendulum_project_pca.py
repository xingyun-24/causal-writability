#!/usr/bin/env python3
"""Render the Pendulum project-page Plotly HTML and a static PNG preview."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COLORS = {"low": "#2b6f9f", "high": "#b84a3f"}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _validate(data: Mapping[str, Any]) -> None:
    counts = data.get("counts", {})
    expected = {
        "fit_pairs_used_for_basis": 64,
        "heldout_pairs_projected": 64,
        "endpoint_points": 128,
        "difference_points": 64,
        "endpoint_groups": 4,
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError(f"data contract mismatch: {counts}")
    endpoint_groups = {
        (row["target_label"], row["condition"]) for row in data["endpoints"]
    }
    if endpoint_groups != {
        ("low", "aligned"), ("low", "conflict"),
        ("high", "aligned"), ("high", "conflict"),
    }:
        raise ValueError(f"unexpected endpoint groups: {endpoint_groups}")


def _html_document(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pendulum held-out residual-state geometry</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: light; --ink:#172522; --muted:#5d6b67; --line:#d9dfdc; --accent:#16786e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#fff; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1500px,96vw); margin:0 auto; padding:42px 0 52px; }}
    header {{ max-width:980px; margin-bottom:28px; }}
    .eyebrow {{ color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }}
    h1 {{ margin:10px 0 12px; font:500 clamp(30px,4vw,52px)/1.05 Georgia,serif; letter-spacing:-.025em; }}
    .lead {{ margin:0; color:#334742; font-size:16px; line-height:1.65; }}
    .protocol {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }}
    .protocol span {{ border:1px solid var(--line); border-radius:999px; padding:6px 10px; color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1.3fr) minmax(360px,.7fr); gap:18px; align-items:stretch; }}
    article {{ border:1px solid var(--line); border-radius:12px; background:#fff; min-width:0; overflow:hidden; }}
    article h2 {{ margin:22px 24px 4px; font:500 22px/1.25 Georgia,serif; }}
    article p {{ margin:0 24px 8px; color:var(--muted); font-size:13px; line-height:1.5; }}
    .plot {{ width:100%; height:660px; }}
    .note {{ margin-top:18px; padding:17px 20px; border-left:3px solid var(--accent); background:#f7faf9; color:#3f514c; font-size:13px; line-height:1.65; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.95em; }}
    @media(max-width:980px) {{ .grid {{ grid-template-columns:1fr; }} .plot {{ height:560px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Pendulum · causal residual geometry</div>
    <h1>A shared fit-only basis resolves held-out physical and shortcut states</h1>
    <p class="lead">The basis is fitted once from 64 matched activation differences. The plots show projections of 64 disjoint held-out pairs at the paper-selected residual block, concatenating all 20 flow-matching calls.</p>
    <div class="protocol"><span>Large–Short · seed 3407 · 50K</span><span>after block 12</span><span>fit 64 / held-out 64</span></div>
  </header>
  <section class="grid">
    <article><h2>Held-out endpoint activations</h2><p>Circle: aligned cue. Diamond: conflicting cue. Color: decoded natural frequency.</p><div id="endpoints" class="plot"></div></article>
    <article><h2>Matched activation differences</h2><p>Aligned minus conflict in the identical basis. Color: absolute decoded frequency gap.</p><div id="differences" class="plot"></div></article>
  </section>
  <div class="note"><strong>Projection contract.</strong> PC1–PC3 are the first three vectors of the uncentered SVD fitted on the 64 fit-set matched differences. Held-out endpoint activations are projected without refitting or recentering. The third axis is a genuine basis projection—not a constructed phase coordinate. Drag to rotate, scroll or pinch to zoom, and hover for sample identity and frequency.</div>
</main>
<script>
const DATA={payload};
const endpointTraces=[];
for (const target of ['low','high']) {{
  for (const condition of ['aligned','conflict']) {{
    const rows=DATA.endpoints.filter(d=>d.target_label===target && d.condition===condition);
    endpointTraces.push({{
      type:'scatter3d', mode:'markers', name:`${{target}} · ${{condition}}`,
      x:rows.map(d=>d.pc1), y:rows.map(d=>d.pc2), z:rows.map(d=>d.pc3),
      customdata:rows.map(d=>[d.receiver_id,d.omega_true,d.omega_hat,d.generation_seed,d.phase]),
      hovertemplate:'%{{customdata[0]}}<br>target: '+target+'<br>condition: '+condition+'<br>ω true: %{{customdata[1]:.3f}}<br>ω generated: %{{customdata[2]:.3f}}<br>seed: %{{customdata[3]}}<br>phase: %{{customdata[4]:.3f}}<extra></extra>',
      marker:{{size:5.8,opacity:.86,symbol:condition==='aligned'?'circle':'diamond',color:rows.map(d=>d.omega_hat),colorscale:'Viridis',cmin:2.0,cmax:6.5,line:{{color:'rgba(20,35,31,.45)',width:.7}},showscale:target==='high'&&condition==='aligned',colorbar:{{title:{{text:'ω generated',side:'right'}},thickness:12,len:.55,x:1.02}}}}
    }});
  }}
}}
const scene={{xaxis:{{title:'PC1 score',showbackground:false,gridcolor:'#e5e9e7',zerolinecolor:'#c5ceca'}},yaxis:{{title:'PC2 score',showbackground:false,gridcolor:'#e5e9e7',zerolinecolor:'#c5ceca'}},zaxis:{{title:'PC3 score',showbackground:false,gridcolor:'#e5e9e7',zerolinecolor:'#c5ceca'}},bgcolor:'#fff',aspectmode:'data',camera:{{eye:{{x:1.45,y:1.4,z:1.05}}}}}};
const config={{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['toImage','sendDataToCloud']}};
Plotly.newPlot('endpoints',endpointTraces,{{paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{{l:0,r:20,t:12,b:0}},legend:{{orientation:'h',x:0,y:1.04,font:{{size:11}}}},scene}},config);
const diff=[];
for (const target of ['low','high']) {{
  const rows=DATA.differences.filter(d=>d.target_label===target);
  diff.push({{type:'scatter3d',mode:'markers',name:`${{target}} target`,x:rows.map(d=>d.pc1),y:rows.map(d=>d.pc2),z:rows.map(d=>d.pc3),customdata:rows.map(d=>[d.receiver_id,d.omega_true,d.omega_aligned,d.omega_conflict,d.generated_frequency_gap,d.generation_seed]),hovertemplate:'%{{customdata[0]}}<br>target: '+target+'<br>ω true: %{{customdata[1]:.3f}}<br>ω aligned: %{{customdata[2]:.3f}}<br>ω conflict: %{{customdata[3]:.3f}}<br>|Δω|: %{{customdata[4]:.3f}}<br>seed: %{{customdata[5]}}<extra></extra>',marker:{{size:6.5,opacity:.9,symbol:'circle',color:rows.map(d=>d.generated_frequency_gap),colorscale:'Viridis',cmin:2.0,cmax:4.0,line:{{color:'rgba(20,35,31,.5)',width:.7}},showscale:target==='high',colorbar:{{title:{{text:'|Δω|',side:'right'}},thickness:12,len:.55,x:1.02}}}}}});
}}
Plotly.newPlot('differences',diff,{{paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{{l:0,r:20,t:12,b:0}},legend:{{orientation:'h',x:0,y:1.04,font:{{size:11}}}},scene}},config);
</script>
</body>
</html>
"""


def _static_preview(data: Mapping[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14.2, 6.4), facecolor="white")
    endpoint_axis = figure.add_subplot(1, 2, 1, projection="3d")
    difference_axis = figure.add_subplot(1, 2, 2, projection="3d")
    endpoint_values = np.asarray([row["omega_hat"] for row in data["endpoints"]])
    norm = plt.Normalize(endpoint_values.min(), endpoint_values.max())
    for target in ("low", "high"):
        for condition, marker in (("aligned", "o"), ("conflict", "D")):
            rows = [row for row in data["endpoints"] if row["target_label"] == target and row["condition"] == condition]
            endpoint_axis.scatter(
                [row["pc1"] for row in rows], [row["pc2"] for row in rows], [row["pc3"] for row in rows],
                c=[row["omega_hat"] for row in rows], cmap="viridis", norm=norm,
                marker=marker, s=34, alpha=.86, edgecolors="#263631", linewidths=.35,
                label=f"{target} · {condition}",
            )
    difference_values = np.asarray([row["generated_frequency_gap"] for row in data["differences"]])
    diff_norm = plt.Normalize(difference_values.min(), difference_values.max())
    for target in ("low", "high"):
        rows = [row for row in data["differences"] if row["target_label"] == target]
        difference_axis.scatter(
            [row["pc1"] for row in rows], [row["pc2"] for row in rows], [row["pc3"] for row in rows],
            c=[row["generated_frequency_gap"] for row in rows], cmap="viridis", norm=diff_norm,
            marker="o", s=38, alpha=.9, edgecolors="#263631", linewidths=.4,
            label=f"{target} target",
        )
    for axis, title in ((endpoint_axis, "Held-out endpoint activations"), (difference_axis, "Held-out matched differences")):
        axis.set_title(title, loc="left", fontsize=14, pad=15)
        axis.set_xlabel("PC1 score", labelpad=7)
        axis.set_ylabel("PC2 score", labelpad=7)
        axis.set_zlabel("PC3 score", labelpad=7)
        axis.view_init(elev=22, azim=-52)
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        axis.grid(True, alpha=.18)
        axis.set_facecolor("white")
    endpoint_mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    difference_mappable = plt.cm.ScalarMappable(norm=diff_norm, cmap="viridis")
    figure.colorbar(endpoint_mappable, ax=endpoint_axis, shrink=.58, pad=.08, label="generated frequency")
    figure.colorbar(difference_mappable, ax=difference_axis, shrink=.58, pad=.08, label="|generated frequency gap|")
    figure.suptitle("Pendulum residual-state geometry · fit64 basis / held-out64 projection", x=.055, ha="left", fontsize=18, fontweight="semibold")
    figure.text(.055, .925, "Large–Short · seed 3407 · 50K · after DiT block 12 · 20 concatenated FM calls", color="#53635f", fontsize=10)
    figure.subplots_adjust(left=.03, right=.96, top=.84, bottom=.08, wspace=.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=180, facecolor="white", bbox_inches="tight")
    os.replace(temporary, path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    _validate(data)
    _atomic_text(args.html, _html_document(data))
    _static_preview(data, args.preview)


if __name__ == "__main__":
    main()
