#!/usr/bin/env python3
"""Build the standalone Spring causal-route 3-D interactive bundle.

The visualization uses frozen held-out top-four coordinates.  Its horizontal
axis is the common route-offset coordinate; the other two axes are the
direction-specific, fit-only phase-plane coordinates used by the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = [
    "#3A9688", "#D8ECE8", "#F8F6F1", "#899097", "#30343B",
    "#899097", "#F8F6F1", "#D8ECE8", "#3A9688",
]
BRANCHES = {
    "true_fast_red_conflict": ("fast", "Target fast", "circle"),
    "true_slow_blue_conflict": ("slow", "Target slow", "diamond"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cyclic_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list("paper_phase", PALETTE, N=256)


def build_payload(coordinates: Path, phase_csv: Path, summary_path: Path) -> dict:
    rows = read_jsonl(coordinates)
    summary = json.loads(summary_path.read_text())
    phase_rows = list(csv.DictReader(phase_csv.open(newline="")))
    phase_by_key = {
        (row["trajectory_id"], row["branch"]): row for row in phase_rows
    }
    if len(rows) != 128 or len(phase_rows) != 128:
        raise AssertionError("Expected 64 held-out trajectories per target direction")

    traces = []
    all_values = []
    for source_name, (key, label, marker) in BRANCHES.items():
        branch = summary["branches"][source_name]
        basis = np.asarray(branch["phase_plane_orthonormal_basis_4x2"], dtype=float)
        beta = np.asarray(branch["first_harmonic_beta_intercept_cos_sin_by_PC"], dtype=float)
        offset = np.asarray(branch["common_edit_offset_mu"], dtype=float)
        subset = [row for row in rows if row["direction"] == source_name]
        points = []
        for index, row in enumerate(subset, 1):
            z = np.asarray(row["z"], dtype=float)
            plane = (z - offset) @ basis
            archived = phase_by_key[(row["trajectory_id"], source_name)]
            expected = np.asarray(
                [float(archived["z_phase_1"]), float(archived["z_phase_2"])]
            )
            if not np.allclose(plane, expected, atol=2e-5, rtol=1e-7):
                raise AssertionError("Archived phase-plane coordinates do not match frozen top-four data")
            point = {
                "index": index,
                "x": float(z[0]),
                "y": float(plane[0]),
                "z": float(plane[1]),
                "theta": float(row["theta_boundary"]),
                "top4": [float(value) for value in z],
            }
            points.append(point)
            all_values.append([point["x"], point["y"], point["z"]])

        theta = np.linspace(0, 2 * np.pi, 241)
        fit = []
        for value in theta:
            predicted = beta[0] + beta[1] * np.cos(value) + beta[2] * np.sin(value)
            plane = (predicted - offset) @ basis
            fit.append({
                "x": float(predicted[0]), "y": float(plane[0]),
                "z": float(plane[1]), "theta": float(value),
            })
        traces.append({
            "key": key,
            "label": label,
            "marker": marker,
            "heldout_n": len(points),
            "phase_r2": float(branch["phase_plane_R2"]),
            "points": points,
            "fit": fit,
        })

    values = np.asarray(all_values)
    bounds = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }
    return {
        "schema_version": 1,
        "title": "A compact causal route is organized by boundary phase",
        "subtitle": "Target direction selects the route family; boundary phase moves the edit around it.",
        "axes": ["route offset coordinate", "phase coordinate 1", "phase coordinate 2"],
        "checkpoint": {"model": "Large Short", "seed": 3408, "step": "50K", "functional_block": "B6"},
        "contract": {
            "pca": "raw, uncentered top-four PCA fit only on disjoint fit trajectories",
            "phase_alignment": "direction-specific first-harmonic plane fit only on fit trajectories",
            "displayed_points": "64 disjoint held-out trajectories per target direction",
            "line": "fit-only first-harmonic prediction",
            "not_used": ["UMAP", "t-SNE", "held-out refitting", "decoded appearance color"],
        },
        "phase_palette": PALETTE,
        "bounds": bounds,
        "traces": traces,
        "sources": {
            "coordinates": coordinates.name,
            "phase_csv": phase_csv.name,
            "summary": summary_path.name,
        },
    }


def render_fallback(payload: dict, out: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "text.color": "#30343B",
        "axes.labelcolor": "#626970",
        "xtick.color": "#899097",
        "ytick.color": "#899097",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 5.2), dpi=180, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    cmap = cyclic_cmap()
    for trace in payload["traces"]:
        fit = trace["fit"]
        ax.plot([p["x"] for p in fit], [p["y"] for p in fit], [p["z"] for p in fit],
                color="#899097", lw=1.1, alpha=.7)
        points = trace["points"]
        ax.scatter([p["x"] for p in points], [p["y"] for p in points], [p["z"] for p in points],
                   c=[p["theta"] for p in points], cmap=cmap, vmin=0, vmax=2*np.pi,
                   marker="o" if trace["marker"] == "circle" else "D", s=28,
                   edgecolor="#FFFFFF", linewidth=.45, alpha=.92, label=trace["label"])
    ax.set_xlabel(payload["axes"][0], labelpad=8)
    ax.set_ylabel(payload["axes"][1], labelpad=8)
    ax.set_zlabel(payload["axes"][2], labelpad=8)
    ax.view_init(elev=20, azim=-58)
    ax.grid(True, color="#DDE0E2", linewidth=.45)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(payload["title"], loc="left", fontsize=11, fontweight="semibold", pad=12)
    fig.text(.125, .91, payload["subtitle"], fontsize=8, color="#626970")
    fig.subplots_adjust(left=.02, right=.92, top=.89, bottom=.03)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Interactive Spring causal-route geometry</title>
<style>
:root{color-scheme:light;--ink:#30343B;--muted:#899097;--dark:#626970;--grid:#DDE0E2;--teal:#3A9688;--paper:#FFFFFF;--wash:#F8F6F1}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.4 Inter,Arial,sans-serif}
.wrap{max-width:1080px;margin:auto;padding:18px 18px 12px}.head{display:flex;gap:16px;align-items:flex-start;justify-content:space-between}
h1{font-size:18px;line-height:1.2;margin:0 0 5px;font-weight:650}.sub{margin:0;color:var(--dark);font-size:13px}.controls{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
button{border:1px solid var(--grid);background:#fff;color:var(--dark);border-radius:999px;padding:6px 10px;font:600 12px/1 Inter,Arial,sans-serif;cursor:pointer}
button:hover,button:focus-visible{border-color:var(--teal);outline:none}button.active{color:#fff;background:var(--teal);border-color:var(--teal)}
.stage{position:relative;margin-top:10px;border-top:1px solid #EEF0F0;border-bottom:1px solid #EEF0F0;background:linear-gradient(#fff,#fff 76%,#FBFAF8)}
canvas{display:block;width:100%;height:min(66vw,640px);min-height:420px;cursor:grab;touch-action:none}canvas.drag{cursor:grabbing}
.tip{position:absolute;display:none;pointer-events:none;min-width:150px;padding:8px 10px;border:1px solid var(--grid);border-radius:8px;background:rgba(255,255,255,.96);box-shadow:0 8px 24px rgba(48,52,59,.10);font-size:12px;color:var(--dark)}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:9px;color:var(--dark);font-size:12px}.mark{display:inline-block;width:9px;height:9px;border:1.5px solid var(--teal);margin-right:5px;vertical-align:-1px}.circle{border-radius:50%}.diamond{transform:rotate(45deg)}
.phase{height:6px;width:150px;border-radius:99px;background:linear-gradient(90deg,#3A9688,#D8ECE8,#F8F6F1,#899097,#30343B,#899097,#F8F6F1,#D8ECE8,#3A9688)}
.note{margin:8px 0 0;color:var(--muted);font-size:11px}.noscript{max-width:100%;height:auto}
@media(max-width:700px){.head{display:block}.controls{justify-content:flex-start;margin-top:10px}canvas{height:520px;min-height:360px}.phase{width:110px}}
</style>
</head>
<body><main class="wrap">
<div class="head"><div><h1 id="title"></h1><p class="sub" id="subtitle"></p></div>
<div class="controls" aria-label="Visible target directions"><button data-view="both" class="active">Both</button><button data-view="fast">Target fast</button><button data-view="slow">Target slow</button><button id="reset">Reset view</button></div></div>
<div class="stage"><canvas id="plot" aria-label="Rotatable three-dimensional held-out causal-route coordinates"></canvas><div class="tip" id="tip"></div></div>
<div class="legend"><span><i class="mark circle"></i>Target fast</span><span><i class="mark diamond"></i>Target slow</span><span>boundary phase 0</span><i class="phase"></i><span>2π</span></div>
<p class="note">Dots are held-out trajectories. Thin curves are fit-only first-harmonic predictions. Drag to rotate; scroll or pinch to zoom. No UMAP or t-SNE.</p>
<noscript><img class="noscript" src="spring-route-geometry-fallback.png" alt="Static view of the two phase-organized causal-route families" /></noscript>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(()=>{"use strict";const D=JSON.parse(document.getElementById("payload").textContent),C=document.getElementById("plot"),X=C.getContext("2d"),tip=document.getElementById("tip");
document.getElementById("title").textContent=D.title;document.getElementById("subtitle").textContent=D.subtitle;
let yaw=-.72,pitch=-.28,zoom=1,view="both",drag=false,last=[0,0],screenPoints=[];
const pal=D.phase_palette.map(h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]);
const mix=(a,b,t)=>a.map((v,i)=>Math.round(v+(b[i]-v)*t));function phaseColor(theta){let u=((theta%(2*Math.PI))+2*Math.PI)%(2*Math.PI)/(2*Math.PI)*(pal.length-1),i=Math.floor(u),t=u-i,c=mix(pal[i],pal[Math.min(i+1,pal.length-1)],t);return `rgb(${c.join(",")})`}
const mins=D.bounds.min,maxs=D.bounds.max,ctr=mins.map((v,i)=>(v+maxs[i])/2),span=Math.max(...mins.map((v,i)=>maxs[i]-v));
function norm(p){return [(p.x-ctr[0])/span*2,(p.y-ctr[1])/span*2,(p.z-ctr[2])/span*2]}
function project(p,w,h){let [x,y,z]=norm(p),cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x-sy*y,y1=sy*x+cy*y,z1=z,y2=cp*y1-sp*z1,z2=sp*y1+cp*z1,f=3.2/(3.2-z2*.72),s=Math.min(w,h)*.34*zoom;return [w/2+x1*f*s,h/2-y2*f*s,z2]}
function line(a,b,color,width=1,dash=[]){X.save();X.strokeStyle=color;X.lineWidth=width;X.setLineDash(dash);X.beginPath();X.moveTo(a[0],a[1]);X.lineTo(b[0],b[1]);X.stroke();X.restore()}
function drawAxes(w,h){const o={x:ctr[0],y:ctr[1],z:ctr[2]},ends=[{x:maxs[0],y:ctr[1],z:ctr[2]},{x:ctr[0],y:maxs[1],z:ctr[2]},{x:ctr[0],y:ctr[1],z:maxs[2]}],labels=D.axes.map((s,i)=>i===0?"route offset":i===1?"phase 1":"phase 2"),po=project(o,w,h);ends.forEach((e,i)=>{let pe=project(e,w,h);line(po,pe,"#C8CBCB",1);X.fillStyle="#899097";X.font="12px Inter,Arial,sans-serif";X.fillText(labels[i],pe[0]+5,pe[1]-4)})}
function draw(){const r=C.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);C.width=Math.round(r.width*dpr);C.height=Math.round(r.height*dpr);X.setTransform(dpr,0,0,dpr,0,0);let w=r.width,h=r.height;X.clearRect(0,0,w,h);drawAxes(w,h);screenPoints=[];
for(const t of D.traces){if(view!=="both"&&view!==t.key)continue;let prev=null;for(const p of t.fit){let q=project(p,w,h);if(prev)line(prev,q,"rgba(98,105,112,.58)",1.2);prev=q}}
let pts=[];for(const t of D.traces){if(view!=="both"&&view!==t.key)continue;for(const p of t.points){let q=project(p,w,h);pts.push({q,p,t})}}pts.sort((a,b)=>a.q[2]-b.q[2]);for(const it of pts){let [x,y]=it.q,col=phaseColor(it.p.theta);X.save();X.fillStyle=col;X.strokeStyle="#FFFFFF";X.lineWidth=1.1;X.beginPath();if(it.t.marker==="diamond"){X.moveTo(x,y-5);X.lineTo(x+5,y);X.lineTo(x,y+5);X.lineTo(x-5,y);X.closePath()}else X.arc(x,y,4.4,0,Math.PI*2);X.fill();X.stroke();X.restore();screenPoints.push(it)}}
function setView(v){view=v;document.querySelectorAll("[data-view]").forEach(b=>b.classList.toggle("active",b.dataset.view===v));tip.style.display="none";draw()}
document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>setView(b.dataset.view));document.getElementById("reset").onclick=()=>{yaw=-.72;pitch=-.28;zoom=1;draw()};
C.addEventListener("pointerdown",e=>{drag=true;last=[e.clientX,e.clientY];C.classList.add("drag");C.setPointerCapture(e.pointerId)});C.addEventListener("pointerup",()=>{drag=false;C.classList.remove("drag")});C.addEventListener("pointermove",e=>{if(drag){yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.25,Math.min(1.25,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw();return}let r=C.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,best=null,bd=144;for(const it of screenPoints){let d=(it.q[0]-mx)**2+(it.q[1]-my)**2;if(d<bd){bd=d;best=it}}if(!best){tip.style.display="none";return}tip.innerHTML=`<strong>${best.t.label}</strong><br>held-out trajectory ${best.p.index}<br>boundary phase ${(best.p.theta/Math.PI).toFixed(2)}π`;tip.style.display="block";tip.style.left=Math.min(r.width-170,mx+14)+"px";tip.style.top=Math.max(6,my-48)+"px"});
C.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.65,Math.min(2.2,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});new ResizeObserver(draw).observe(C);draw();})();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--phase-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("website/public/interactives"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args.coordinates, args.phase_csv, args.summary)
    data_path = args.out_dir / "spring-route-geometry-data.json"
    html_path = args.out_dir / "spring-route-geometry.html"
    fallback_path = args.out_dir / "spring-route-geometry-fallback.png"
    data_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    html_path.write_text(HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))
    render_fallback(payload, fallback_path)
    manifest = {
        "status": "PASS",
        "interactive": html_path.name,
        "data": data_path.name,
        "fallback": fallback_path.name,
        "checkpoint": payload["checkpoint"],
        "heldout_points": sum(trace["heldout_n"] for trace in payload["traces"]),
        "phase_r2": {trace["key"]: trace["phase_r2"] for trace in payload["traces"]},
        "sources": payload["sources"],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
