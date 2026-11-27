import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
# Global Ultralytics settings point runs_dir outside the repo; keep runs here
RUNS_DIR = PROJECT_DIR / "runs" / "detect"

# Run with --color to train on data/custom_color (see prepare_dataset.py --color)
COLOR = "--color" in sys.argv
DATA = str(PROJECT_DIR / "data" / ("custom_color" if COLOR else "custom") / "data.yaml")
NAME = "blackjack_cards_color" if COLOR else "blackjack_cards"


def pick_device():
    """NVIDIA GPU, then Apple GPU, then CPU, so training runs on any machine."""
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # Pretrained COCO weights; downloaded automatically on first run
    model = YOLO(PROJECT_DIR / "yolo11n.pt")

    # Hue/saturation jitter does nothing on grayscale; keep the defaults for color
    color_aug = {} if COLOR else {"hsv_h": 0.0, "hsv_s": 0.0}

    model.train(
        data=DATA,
        epochs=100,
        patience=20,
        imgsz=640,
        batch=8,  # 16 swaps on an 8 GB Mac
        workers=2,
        device=pick_device(),
        fliplr=0.0,  # mirrored rank/suit glyphs never occur on real cards
        degrees=15,
        project=str(RUNS_DIR),
        name=NAME,
        exist_ok=True,  # reuse the run folder so best.pt is always at the same path
        **color_aug,
    )

    # Score the best checkpoint once on images never used during training
    best = YOLO(RUNS_DIR / NAME / "weights" / "best.pt")
    best.val(data=DATA, split="test", imgsz=640, device=pick_device(),
             project=str(RUNS_DIR), name=NAME + "_test", exist_ok=True)


if __name__ == "__main__":
    main()
