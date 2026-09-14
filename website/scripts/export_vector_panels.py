"""Export final-paper vectors, not raster images wrapped inside SVG files."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pymupdf

SOURCES = {
    "f1": "figures/source/figure1_haomin.pdf",
    "f2": "figures/generated/figure2_solution_selection.pdf",
    "f3": "figures/generated/figure3_causal_route.pdf",
    "f4": "figures/drafts/figure4_writeability_draft.pdf",
    "f6": "figures/drafts/figure5_downstream_draft.pdf",
}
# PDF-point crops retain each chart's axes. Shared panel headings are supplied
# by the existing webpage; cropped subplots never split their shared cue label.
PANELS = {
    "fig1a-context": ("f1", (0, 0, 202, 135)),
    "fig1b-conflict": ("f1", (210, 0, 396, 135)),
    "fig1c-rewrite": ("f1", (0, 137, 203, 263.622)),
    "fig1d-commitment": ("f1", (211, 137, 396, 263.622)),
    "fig2a-slow": ("f2", (0, 22, 199, 152)),
    "fig2a-fast": ("f2", (212, 22, 396, 152)),
    "fig2b-examples": ("f2", (20, 163, 396, 250)),
    "fig2c-history": ("f2", (0, 260, 396, 352.8)),
    "fig3a-replacement": ("f3", (0, 0, 270, 105)),
    "fig3b-fast-phase": ("f3", (0, 129, 131, 263)),
    "fig3b-slow-phase": ("f3", (132, 123, 270, 263)),
    "fig3b-recovery": ("f3", (277, 147, 396, 259)),
    "fig3c-controller-model": ("f3", (0, 307, 146, 428.4)),
    "fig3c-controller-replay": ("f3", (164, 305, 396, 428.4)),
    "fig4a-training": ("f4", (0, 0, 173, 156)),
    "fig4b-solutions": ("f4", (176, 0, 396, 156)),
    "fig4c-fate": ("f4", (0, 158, 172, 302.4)),
    "fig4d-history": ("f4", (172, 158, 396, 302.4)),
    "fig5a-persistence": ("f6", (0, 0, 192, 127)),
    "fig5b-transfer": ("f3", (270, 0, 396, 108)),
    "fig5c-kv": ("f6", (223, 0, 396, 267.84)),
    "fig5d-head": ("f6", (0, 128, 223, 267.84)),
}
SUPPLEMENTS = {
    "fig6-multiseed-behavior": "figures/generated/appendix_figure2_multiseed_behavior.pdf",
    "fig7-controller-ablation": "figures/generated/appendix_short50k_controller_comparison.pdf",
    "fig14-cross-step-exact100k": "figures/generated/appendix_figure5_cross_step_exact100k.pdf",
    "fig15-implementation-multiplicity": "figures/generated/appendix_figure5_implementation_multiplicity.pdf",
    "fig17-fm-time-decoded": "figures/generated/appendix_figure5_fm_and_decoded.pdf",
}


def export(pdf, destination, crop=None):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    with pymupdf.open(pdf) as document:
        page = document[0]
        root = ET.fromstring(page.get_svg_image(text_as_path=True))
        box = pymupdf.Rect(crop) if crop else page.rect
        root.set("viewBox", " ".join(str(x) for x in (box.x0, box.y0, box.width, box.height)))
        root.set("width", str(box.width))
        root.set("height", str(box.height))
        root.set("overflow", "hidden")
        paths = sum(e.tag.endswith("}path") for e in root.iter())
        assert paths > 0, f"No vector paths in {pdf}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
        return {"source": str(pdf.name), "viewBox": list(box), "vector_paths": paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--pendulum", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    manifest = {}
    for name, (key, crop) in PANELS.items():
        manifest[name] = export(args.paper / SOURCES[key], args.public / "paper/vector" / f"{name}.svg", crop)
    for name, source in SUPPLEMENTS.items():
        manifest[name] = export(args.paper / source, args.public / "paper/vector" / f"{name}.svg")
    for source, name in [("pendulum_behavior", "behavior"), ("pendulum_decoded_recovery", "decoded-recovery"),
                         ("pendulum_state_geometry", "state-geometry"), ("pendulum_writeability", "writeability")]:
        manifest[name] = export(args.pendulum / f"{source}.pdf", args.public / "pendulum" / f"{name}.svg")
    (args.public / "paper/vector/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(manifest)} vector figures; embedded video/image pixels are unchanged.")


if __name__ == "__main__":
    main()
