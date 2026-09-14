"""Reconstruct archived fit-only PC1/2/3 and export held-out raw/difference HTML."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def html(points, difference, library):
    key = "aligned_g_E3" if difference else "g_E3"
    low, high = min(r[key] for r in points), max(r[key] for r in points)
    traces = []
    groups = [("high", "circle"), ("low", "diamond")] if difference else [("aligned", "circle"), ("conflict", "diamond")]
    for group, marker in groups:
        selected = [p for p in points if p["target_band" if difference else "condition"] == group]
        traces.append({"type": "scatter3d", "mode": "markers", "name": "Target " + group + " g" if difference else group,
            "x": [p["pc1"] for p in selected], "y": [p["pc2"] for p in selected], "z": [p["pc3"] for p in selected],
            "customdata": [[p["pair_id"], p.get("input_colour", "paired"), p["target_band"], p["target_g"], p[key]] for p in selected],
            "hovertemplate": "pair=%{customdata[0]}<br>input colour=%{customdata[1]}<br>target band=%{customdata[2]}<br>target g=%{customdata[3]:.6f}<br>" + ("aligned" if difference else "decoded") + " g=%{customdata[4]:.6f}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra>held-out</extra>",
            "marker": {"symbol": marker, "size": 6, "opacity": .86, "color": [p[key] for p in selected],
                       "colorscale": "Viridis", "cmin": low, "cmax": high, "showscale": not traces,
                       "colorbar": {"title": "Aligned g" if difference else "Decoded g"}, "line": {"color": "#202124", "width": .35}}})
    title = "Free Fall: matched-difference PCA" if difference else "Free Fall: raw PCA projections"
    layout = {"title": title, "template": "plotly_white",
        "scene": {"xaxis": {"title": "Projection on residual PC1"}, "yaxis": {"title": "Projection on residual PC2"},
                  "zaxis": {"title": "Projection on residual PC3"}, "aspectmode": "auto"},
        "legend": {"orientation": "h", "x": 0, "y": 1, "xanchor": "left", "yanchor": "top"},
        "margin": {"l": 0, "r": 0, "t": 55, "b": 0}}
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + title + '</title><script>' + library + '</script></head><body style="margin:0;background:#fff">'
            '<div id="plot" style="width:100vw;height:100vh"></div><script>const traces=' + json.dumps(traces) + ';const layout=' + json.dumps(layout) + ';'
            'if(innerWidth<600){layout.title={text:"Free Fall: '+ ('differences' if difference else 'raw PCA') +'",font:{size:15}};layout.scene.xaxis.title="PC1";layout.scene.yaxis.title="PC2";layout.scene.zaxis.title="PC3";layout.scene.camera={eye:{x:2,y:2,z:2}};layout.margin.t=75;}'
            "Plotly.newPlot('plot',traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});</script></body></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--basis-reference", type=Path, required=True)
    parser.add_argument("--plotly", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pca = np.load(args.pca, allow_pickle=False)
    fit = np.flatnonzero(pca["train_mask"])
    held = np.flatnonzero(pca["heldout_mask"])
    assert len(fit) == len(held) == 64 and not np.any(pca["train_mask"] & pca["heldout_mask"])
    weights = pca["train_vectors"][:, :3] / np.sqrt(pca["eigenvalues"][:3])[None, :]
    shape = tuple(int(x) for x in pca["delta_shape"])
    basis = torch.zeros((3, int(np.prod(shape))), dtype=torch.float64, device=args.device)
    for j, i in enumerate(fit):
        pair = str(pca["pair_ids"][i])
        delta = np.load(args.captures / "fit" / f"{pair}_delta.npy", mmap_mode="r")
        assert delta.shape == shape
        value = torch.as_tensor(np.array(delta, copy=True).reshape(-1), device=args.device, dtype=torch.float64)
        weight = torch.tensor(weights[j], device=args.device)
        basis.add_(weight[:, None] * value[None])
    gram = (basis @ basis.T).cpu().numpy()
    np.testing.assert_allclose(gram, np.eye(3), rtol=1e-5, atol=1e-5)
    reference = torch.as_tensor(np.array(np.load(args.basis_reference), copy=True).reshape(2, -1), device=args.device)
    basis_error = float(torch.linalg.vector_norm(basis[:2] - reference) / torch.linalg.vector_norm(reference))
    assert basis_error < 1e-5, basis_error
    points, differences, errors = [], [], []
    for i in held:
        pair = str(pca["pair_ids"][i])
        record = json.loads((args.captures / "heldout" / f"{pair}.json").read_text())
        coordinates = {}
        for condition in ("aligned", "conflict"):
            activation = np.load(args.captures / "heldout" / f"{pair}_{condition}.npy", mmap_mode="r")
            value = torch.as_tensor(np.array(activation, copy=True).reshape(-1), device=args.device, dtype=torch.float64)
            coordinates[condition] = (basis @ value).cpu().numpy()
            measured = record["measurements"][condition]
            assert measured["valid"] and np.isfinite(measured["g_E3"])
            points.append({"pair_id": pair, "condition": condition, "split": "heldout",
                           "target_band": record["target_band"], "target_g": record["target_g"],
                           **measured, **dict(zip(("pc1", "pc2", "pc3"), coordinates[condition].tolist()))})
        difference = coordinates["aligned"] - coordinates["conflict"]
        expected = pca["scores"][i, :3]
        np.testing.assert_allclose(difference, expected, rtol=1e-5, atol=1e-3)
        errors.append(float(np.max(np.abs(difference - expected))))
        differences.append({"pair_id": pair, "split": "heldout", "condition": "difference",
                            "target_band": record["target_band"], "target_g": record["target_g"],
                            "aligned_g_E3": record["measurements"]["aligned"]["g_E3"],
                            **dict(zip(("pc1", "pc2", "pc3"), difference.tolist()))})
    counts = {f"{colour}/{band}": sum(p["input_colour"] == colour and p["target_band"] == band for p in points)
              for colour in ("red", "blue") for band in ("low", "high")}
    assert set(counts.values()) == {32}
    metadata = {"task": "freefall", "block_zero_based": 1, "checkpoint_step": 100000,
                "observed_frames": 32, "fm_calls": 20, "fit_pairs": 64, "heldout_pairs": 64,
                "basis": "first three archived uncentered fit-only PCA directions, reconstructed from recaptured fit residuals",
                "controller_rank": 2, "display_rank": 3, "raw_points": 128, "difference_points": 64,
                "groups": counts, "basis_gram": gram.tolist(), "basis_pc12_relative_error": basis_error,
                "max_archived_score_absolute_error": max(errors), "centering": "none",
                "raw_colour": "newly decoded gravity (E3)", "difference_colour": "aligned decoded gravity (E3)"}
    args.out.mkdir(parents=True, exist_ok=True)
    library = args.plotly.read_text()
    assert "plotly.js v2.35.2" in library[:300]
    for filename, data, difference in (("freefall_raw_projection_interactive.html", points, False),
                                       ("freefall_matched_difference_interactive.html", differences, True)):
        (args.out / filename).write_text(html(data, difference, library))
    (args.out / "freefall-pca-data.json").write_text(json.dumps({"metadata": metadata, "raw_points": points, "difference_points": differences}, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
