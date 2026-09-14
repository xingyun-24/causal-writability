from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "pdfs" / "main_current_hi"
OUTPUT = ROOT / "public" / "paper"
BASE_WIDTH = 935
BASE_HEIGHT = 1210

# Crops are defined against the visually audited 110 dpi page renders.
# They preserve complete plot axes, labels, legends, and color bars.
CROPS = {
    "fig1-overview.png": (2, (160, 88, 780, 550)),
    "fig1a-context.png": (2, (175, 145, 440, 325)),
    "fig1b-conflict.png": (2, (470, 145, 760, 325)),
    "fig1c-rewrite.png": (2, (165, 350, 555, 510)),
    "fig1d-commitment.png": (2, (555, 350, 780, 515)),
    "fig2-solution-landscape.png": (4, (175, 88, 775, 360)),
    "fig2-history-control.png": (4, (175, 365, 775, 565)),
    "fig2a-slow.png": (4, (175, 180, 475, 382)),
    "fig2a-fast.png": (4, (490, 180, 760, 382)),
    "fig2b-examples.png": (4, (220, 410, 730, 625)),
    "fig2c-history.png": (4, (175, 620, 780, 760)),
    "fig3-compact-route.png": (5, (175, 255, 775, 500)),
    "fig3-controller.png": (5, (175, 500, 775, 750)),
    "fig3a-replacement.png": (5, (190, 155, 750, 250)),
    "fig3b-fast-phase.png": (5, (180, 290, 365, 495)),
    "fig3b-slow-phase.png": (5, (375, 290, 585, 495)),
    "fig3b-recovery.png": (5, (605, 300, 770, 490)),
    "fig3c-controller-model.png": (5, (180, 550, 405, 720)),
    "fig3c-controller-replay.png": (5, (470, 520, 775, 740)),
    "fig4-writeability.png": (7, (175, 82, 780, 690)),
    "fig4a-training.png": (7, (145, 165, 420, 410)),
    "fig4b-solutions.png": (7, (445, 165, 780, 410)),
    "fig4c-fate.png": (7, (150, 460, 425, 692)),
    "fig4d-history.png": (7, (450, 460, 780, 692)),
    "fig5-bottleneck.png": (8, (175, 82, 780, 680)),
    "fig5a-persistence.png": (8, (135, 145, 480, 295)),
    "fig5b-transfer.png": (8, (540, 155, 770, 270)),
    "fig5c-kv.png": (8, (190, 305, 780, 465)),
    "fig5d-head.png": (8, (155, 480, 780, 650)),
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, (page, box) in CROPS.items():
        source = SOURCE / f"page-{page}.png"
        image = Image.open(source)
        scale_x = image.width / BASE_WIDTH
        scale_y = image.height / BASE_HEIGHT
        scaled = tuple(
            round(value * (scale_x if index % 2 == 0 else scale_y))
            for index, value in enumerate(box)
        )
        crop = image.crop(scaled)
        crop.save(OUTPUT / filename, optimize=True)
        print(f"{filename}: {crop.width}x{crop.height}")


if __name__ == "__main__":
    main()
