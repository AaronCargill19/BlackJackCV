"""
Scores every card detector on the test set: precision, recall, F1 and accuracy.

A detection is correct (TP) if it has the right class and overlaps an unused
ground-truth card with IoU >= 0.5. Detection has no true negatives, so
accuracy = TP / (TP + FP + FN).

Usage:
  python evaluate.py               # confidence threshold 0.50 for the YOLO models
  python evaluate.py 0.65          # another threshold (the classical model has none)
  python evaluate.py 0.5 --json    # machine-readable output
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "classical"))  # append: classical/train.py must not shadow ./train.py
import cards  # noqa: E402
from train import pick_device  # noqa: E402

IOU = 0.50
GRAY_TEST = ROOT / "data/custom/test"
COLOR_TEST = ROOT / "data/custom_color/test"


def yolo_detector(weights, conf):
    from ultralytics import YOLO

    model = YOLO(weights)
    if list(model.names.values()) != cards.CLASS_NAMES:
        raise ValueError(f"{weights} classes {list(model.names.values())} differ from {cards.CLASS_NAMES}")
    device = pick_device()

    def detect(image_path):
        res = model.predict(image_path, conf=conf, imgsz=640, device=device, verbose=False)[0]
        order = res.boxes.conf.argsort(descending=True)
        return [(int(c), tuple(b)) for c, b in zip(res.boxes.cls[order].tolist(), res.boxes.xyxy[order].cpu().numpy())]

    return detect


def classical_detector():
    svm = cards.load_model()
    return lambda image_path: cards.detect(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE), svm)


def score(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn}


assert abs(score(8, 2, 2)["f1"] - 0.8) < 1e-9 and abs(score(8, 2, 2)["accuracy"] - 8 / 12) < 1e-9


def evaluate(detect, test_dir):
    names = cards.CLASS_NAMES
    per = {c: [0, 0, 0] for c in range(len(names))}
    mixups = {}
    for image_path in sorted((test_dir / "images").iterdir()):
        h, w = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).shape
        gt = []
        for line in (test_dir / "labels" / (image_path.stem + ".txt")).read_text().split("\n"):
            if line.strip():
                c, x, y, bw, bh = map(float, line.split())
                gt.append((int(c), ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)))
        used = set()
        for c, box in detect(image_path):
            best, best_i = 0.0, None
            for i, (gc, gbox) in enumerate(gt):
                if i not in used and gc == c and (v := cards.box_iou(box, gbox)) > best:
                    best, best_i = v, i
            if best >= IOU:
                used.add(best_i)
                per[c][0] += 1
            else:
                per[c][1] += 1
                # if it sat on a different card, record the mix-up
                hit = max(((cards.box_iou(box, gb), gc) for gc, gb in gt), default=(0, None))
                if hit[0] >= IOU and hit[1] != c:
                    key = f"{names[hit[1]]}->{names[c]}"
                    mixups[key] = mixups.get(key, 0) + 1
        for i, (gc, _) in enumerate(gt):
            if i not in used:
                per[gc][2] += 1
    total = [sum(v[i] for v in per.values()) for i in range(3)]
    return {"overall": score(*total), "per_class": {names[c]: score(*v) for c, v in per.items()}, "mixups": mixups}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conf = float(args[0]) if args else 0.50
    detectors = {
        "yolo_grayscale": (yolo_detector(ROOT / "runs/detect/blackjack_cards/weights/best.pt", conf), GRAY_TEST),
        "yolo_color": (yolo_detector(ROOT / "runs/detect/blackjack_cards_color/weights/best.pt", conf), COLOR_TEST),
        "classical": (classical_detector(), GRAY_TEST),
    }
    results = {name: evaluate(detect, test_dir) for name, (detect, test_dir) in detectors.items()}

    if "--json" in sys.argv:
        json.dump(results, sys.stdout, indent=1)
        return
    print(f"Test set, YOLO confidence {conf:.2f}, IoU >= {IOU}")
    print(f"{'model':16} {'precision':>9} {'recall':>7} {'F1':>7} {'accuracy':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
    for name, r in results.items():
        o = r["overall"]
        print(f"{name:16} {o['precision']:9.1%} {o['recall']:7.1%} {o['f1']:7.1%} {o['accuracy']:8.1%} {o['tp']:4} {o['fp']:4} {o['fn']:4}")
    for name, r in results.items():
        print(f"{name} mix-ups (true->predicted): {r['mixups'] or 'none'}")


if __name__ == "__main__":
    main()
