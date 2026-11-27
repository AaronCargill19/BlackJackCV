# BlackJackCV
Using computer vision to analyze blackjack tables and return probable plays.

Two card detectors are compared:
- **Deep learning:** YOLO11, in `best.pt`
- **Classical:** contour detection + HOG/SVM, in `classical/`

Results are in `results.md` and `classical/classical.md`.

## Setup (any machine)

**Requires Python 3.12 or newer.** The pinned versions in `requirements.txt` have no wheels for 3.11 or older. They install on Windows (x64), Linux (x86_64) and Apple Silicon Macs. **Intel Macs are not supported** by these pins, because `torch==2.15.0` has no Intel-Mac build. Tested with Python 3.14 on macOS (Apple M2).

The scripts pick the best available device automatically: an NVIDIA GPU, then an Apple GPU, then the CPU. On a minimal Linux install, OpenCV's windows need the system OpenGL library (`sudo apt install libgl1` on Ubuntu/Debian).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Every script finds its files relative to its own location, so they can be run from any folder.

## Run the live app

A webcam is needed. Click **CALIBRATE**, then click the table corners in order: top-left, top-right, bottom-right, bottom-left.

```bash
python main.py                              # YOLO detector
python main_with_stats.py                   # YOLO + card counting / odds panel
python classical/main_with_classical.py     # classical detector (contours + HOG/SVM)
```

These need only `best.pt` (YOLO) or `classical/card_svm.yml.gz` (classical), which are both in the repo.

## Retrain and evaluate

This needs the labeled dataset in `label_studio_combined/`, with `images/`, `labels/` and `classes.txt` from the Label Studio YOLO export. The folder is **not in git**, so copy it in first.

```bash
python greyscale.py                 # images -> images_grayscale
python prepare_dataset.py           # session split -> data/custom
python prepare_dataset.py --color   # same split, color images -> data/custom_color
python train.py                     # YOLO grayscale -> runs/detect/blackjack_cards (copy weights/best.pt to ./best.pt to use it)
python train.py --color             # YOLO color
python classical/train.py           # classical HOG/SVM -> classical/card_svm.yml.gz
python evaluate.py                  # precision / recall / F1 / accuracy for all models on the test set
```

The class order is fixed in `label_studio_combined/classes.txt`: A, 2–10, J, Q, K, Back. `prepare_dataset.py` and `evaluate.py` stop with an error if a dataset or model uses a different order.

## Other

```bash
python validate_stats.py            # checks the odds engine against simulation, saves figures to stats_figures/
```
