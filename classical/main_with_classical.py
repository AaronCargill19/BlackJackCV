"""
Runs the live app (main.py) with the classical detector instead of YOLO:
contour/edge card finder + HOG/SVM classifier from classical/card_svm.yml.gz.

Everything else (camera, calibration, buttons, hand display) is main.py's own code.
The only change is that main.py's YOLO(...) is replaced by ClassicalModel, which
answers predict() in the same shape YOLO does.

Usage (from anywhere): python classical/main_with_classical.py
"""
import sys
from pathlib import Path

import cv2

CLASSICAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLASSICAL_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CLASSICAL_DIR))

import cards  # noqa: E402
import main as app  # noqa: E402


class _Box:
    """One detection in the shape main.detect_cards reads: box.conf[0], box.cls[0], box.xyxy[0].tolist()."""

    def __init__(self, class_id, xyxy):
        # The SVM gives a class, not a probability, so every detection reports 100%.
        self.conf = [1.0]
        self.cls = [class_id]
        self.xyxy = [_List(xyxy)]


class _List(list):
    def tolist(self):
        return list(self)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class ClassicalModel:
    """Stands in for ultralytics.YOLO inside main.py."""

    def __init__(self, _weights_path=None):
        self.svm = cards.load_model()  # classical/card_svm.yml.gz
        self.names = dict(enumerate(cards.CLASS_NAMES))

    def predict(self, source, **_yolo_options):
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
        boxes = [_Box(class_id, box) for class_id, box in cards.detect(gray, self.svm)]
        return [_Result(boxes)]


if __name__ == "__main__":
    app.YOLO = ClassicalModel
    print(f"Classical detector: {cards.MODEL_PATH.name}")
    app.main()
