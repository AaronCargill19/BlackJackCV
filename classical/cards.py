"""
Classical card detector: contour/edge detection finds card-shaped regions,
then an SVM on HOG features names each one (or rejects it as not a card).

Pipeline for one grayscale table image:
  1. find_candidates: several binary masks (adaptive / Otsu threshold, Canny edges),
     external contours, keep rotated rectangles shaped like a card.
  2. warp_card: perspective-warp each rectangle to an upright card image.
  3. card_features: HOG of the whole card (pip layout) + HOG of the corner index.
  4. An SVM (trained in classical/train.py) predicts a rank, Back, or NOT_CARD.
"""
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "Back"]
NOT_CARD = len(CLASS_NAMES)  # extra SVM class for candidates that aren't cards

MODEL_PATH = Path(__file__).resolve().parent / "card_svm.yml.gz"  # OpenCV reads/writes gzip directly; plain YAML is 111 MB

CARD_W, CARD_H = 120, 180  # upright warped card size (standard card ratio ~0.69)

# Candidate shape filters, tuned on the training images
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.012, 0.15  # rotated-rect area / image area
MIN_RATIO, MAX_RATIO = 0.5, 0.9  # short side / long side
MIN_FILL = 0.75  # contour area / rotated-rect area (cards are solid rectangles)
DUPLICATE_IOU = 0.5  # candidates from different masks overlapping more than this are one card


def _kernel(n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def _masks(gray):
    """Each mask catches cards on different backgrounds; together they cover ~95-99%."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -10)
    return [
        adaptive,  # bright cards on dark tables
        cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, _kernel(5)),  # cuts thin lines (overlay, fabric stripes)
        cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.dilate(cv2.Canny(blur, 30, 90), _kernel(3), iterations=2),  # card outlines
        # light backgrounds (whiteboard): cards are only slightly brighter, so a softer threshold
        cv2.morphologyEx(
            cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, -3),
            cv2.MORPH_OPEN,
            _kernel(5),
        ),
    ]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def find_candidates(gray):
    """Returns [(rotated_rect, (x1, y1, x2, y2))] for card-shaped regions."""
    h, w = gray.shape
    found = []
    for mask in _masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            area = rw * rh
            if not MIN_AREA_FRAC * w * h < area < MAX_AREA_FRAC * w * h:
                continue
            if not MIN_RATIO < min(rw, rh) / max(rw, rh) < MAX_RATIO:
                continue
            if cv2.contourArea(contour) / area < MIN_FILL:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            box = (x, y, x + bw, y + bh)
            if all(box_iou(box, kept) <= DUPLICATE_IOU for _, kept in found):
                found.append((rect, box))
    return found


def warp_card(gray, rect):
    """Perspective-warps a rotated rectangle to an upright CARD_W x CARD_H card."""
    pts = cv2.boxPoints(rect)
    # order clockwise from top-left so the warp never mirrors the card
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
    # No "long side vertical" rule: the table calibration warp stretches cards differently
    # per session (an upright card can be wider than tall), so the edge nearest horizontal
    # stays on top. shortcut: cards turned more than ~45 degrees come out sideways and
    # will likely be misread; add 90-degree-rotated training crops if that shows up.
    dst = np.float32([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]])
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(pts.astype(np.float32), dst), (CARD_W, CARD_H))


# Whole card at 64x96: pip count and layout. Corner at 32x64: the rank index.
_HOG_CARD = cv2.HOGDescriptor((64, 96), (16, 16), (8, 8), (8, 8), 9)
_HOG_CORNER = cv2.HOGDescriptor((32, 64), (16, 16), (8, 8), (8, 8), 9)
CORNER = (slice(0, 50), slice(0, 26))  # rows, cols of the index in a CARD_W x CARD_H card


def card_features(card):
    """HOG feature vector for an upright warped card. Cards are 180-degree symmetric,
    so the top-left corner always holds an index whichever way up the card lies."""
    card = cv2.equalizeHist(card)  # evens out lighting between sessions
    whole = _HOG_CARD.compute(cv2.resize(card, (64, 96))).ravel()
    corner = _HOG_CORNER.compute(cv2.resize(card[CORNER], (32, 64))).ravel()
    return np.concatenate([whole, corner]).astype(np.float32)


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run: python classical/train.py")
    return cv2.ml.SVM_load(str(path))


def detect(gray, svm):
    """Returns [(class_id, (x1, y1, x2, y2))] for every candidate the SVM calls a card."""
    candidates = find_candidates(gray)
    if not candidates:
        return []
    feats = np.stack([card_features(warp_card(gray, rect)) for rect, _ in candidates])
    preds = svm.predict(feats)[1].ravel().astype(int)
    return [(int(p), box) for p, (_, box) in zip(preds, candidates) if p != NOT_CARD]
