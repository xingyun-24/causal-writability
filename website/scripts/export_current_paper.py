"""Export unchanged current-paper figure PDFs as browser PNGs."""
import argparse
import json
from pathlib import Path
import pymupdf

FIGURES = {1: "figures/source/figure1_haomin.pdf", 2: "figures/generated/figure2_solution_selection.pdf",
           3: "figures/generated/figure3_causal_route.pdf", 4: "figures/drafts/figure4_writeability_draft.pdf",
           5: "figures/generated/figure5_pretrained_main.pdf", 6: "figures/drafts/figure5_downstream_draft.pdf"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for number, filename in FIGURES.items():
        with pymupdf.open(args.source / filename) as doc:
            doc[0].get_pixmap(matrix=pymupdf.Matrix(2.6, 2.6), alpha=False).save(args.out / f"figure{number}.png")
    (args.out / "sources.json").write_text(json.dumps(FIGURES, indent=2) + "\n")


if __name__ == "__main__":
    main()
