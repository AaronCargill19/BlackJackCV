"""
Trains the HOG + SVM card classifier used by classical/cards.py.

Training samples come from running the same contour candidate finder on the training
images, so the SVM sees cards exactly as detect() will warp them:
  - a candidate overlapping a labeled card (IoU >= 0.5) is a sample of that class
  - a candidate overlapping no card (IoU < 0.3) is NOT_CARD
  - plus random card-sized background patches as extra NOT_CARD samples
Each sample is also added rotated 180 degrees (a card looks the same either way up).

C and gamma are chosen on the validation set; the test set is never touched here.
Run prepare_dataset.py first. Usage: python classical/train.py
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cards  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "custom"
BACKGROUND_PER_IMAGE = 3


def read_labels(label_path, w, h):
    boxes = []
    for line in label_path.read_text().split("\n"):
        if line.strip():
            c, x, y, bw, bh = map(float, line.split())
            boxes.append((int(c), ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)))
    return boxes


def random_background(gray, labels, rng):
    """A card-sized rotated patch that doesn't overlap any labeled card."""
    h, w = gray.shape
    for _ in range(50):
        bw, bh = rng.uniform(0.12, 0.3) * w, rng.uniform(0.2, 0.45) * h
        cx, cy = rng.uniform(bw / 2, w - bw / 2), rng.uniform(bh / 2, h - bh / 2)
        box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        if all(cards.box_iou(box, gt) < 0.1 for _, gt in labels):
            return ((cx, cy), (bw, bh), rng.uniform(-20, 20))
    return None


def collect(split, with_background):
    rng = random.Random(0)
    feats, targets = [], []
    for image_path in sorted((DATA / split / "images").iterdir()):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        labels = read_labels(DATA / split / "labels" / (image_path.stem + ".txt"), w, h)

        rects = []
        for rect, box in cards.find_candidates(gray):
            best_iou, best_class = max(((cards.box_iou(box, gt), c) for c, gt in labels), default=(0, None))
            if best_iou >= 0.5:
                rects.append((rect, best_class))
            elif best_iou < 0.3:
                rects.append((rect, cards.NOT_CARD))
        if with_background:
            for _ in range(BACKGROUND_PER_IMAGE):
                rect = random_background(gray, labels, rng)
                if rect is not None:
                    rects.append((rect, cards.NOT_CARD))

        for rect, target in rects:
            card = cards.warp_card(gray, rect)
            for view in (card, cv2.rotate(card, cv2.ROTATE_180)):
                feats.append(cards.card_features(view))
                targets.append(target)
    return np.stack(feats), np.array(targets, dtype=np.int32)


def train_svm(x, y, c, gamma):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(c)
    svm.setGamma(gamma)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    return svm


def main():
    x_train, y_train = collect("train", with_background=True)
    x_valid, y_valid = collect("valid", with_background=False)
    print(f"Training samples: {len(y_train)}  Validation samples: {len(y_valid)}")
    print("Per class (train):", {(cards.CLASS_NAMES + ["NOT_CARD"])[c]: int((y_train == c).sum()) for c in np.unique(y_train)})

    best = None
    for c in (3, 10, 100):
        for gamma in (0.002, 0.005, 0.01, 0.02):
            svm = train_svm(x_train, y_train, c, gamma)
            acc = float((svm.predict(x_valid)[1].ravel().astype(int) == y_valid).mean())
            print(f"  C={c:<4} gamma={gamma:<5} validation accuracy {acc:.1%}")
            if best is None or acc > best[0]:
                best = (acc, c, gamma, svm)

    acc, c, gamma, svm = best
    svm.save(str(cards.MODEL_PATH))
    print(f"Best: C={c} gamma={gamma}, validation accuracy {acc:.1%}. Saved {cards.MODEL_PATH}")


if __name__ == "__main__":
    main()
