# Card Detector Results: Deep Learning (YOLO) vs Classical (Contours + HOG/SVM)

**Summary:** on 55 test images from unseen table setups, the YOLO grayscale model got 100% of cards right, and the classical pipeline got 94.9%. See [Classical approach](#classical-approach-contours--hog--svm) for the comparison.

**Recommendation:** use the **grayscale model at confidence 0.50**. It is the model `main.py` loads from `best.pt`. On 55 test images from table setups it never trained on, it got **all 267 cards right**: no wrong cards and no missed cards. Raising the threshold to 0.65 only caused misses.

## Setup

| | |
|---|---|
| Base model | `yolo11n.pt` (YOLO11 nano, COCO-pretrained, trained fresh, not from an older card model) |
| Data | `label_studio_combined`: 397 images, 1,593 labeled cards, 14 classes (A–K plus Back) |
| Boxes | Whole card, not just the corner |
| Split | **By capture session** (see below): 286 train / 56 valid / 55 test, identical images for both models |
| Training | 640px, batch 8, up to 100 epochs, early stopping after 20 epochs without improvement, Apple M2 (MPS) |
| Augmentation | Horizontal flips off (mirrored ranks never occur), rotation ±15°; hue/saturation jitter off for grayscale, YOLO defaults for color |

| Model | Input | Best epoch | Training time | Weights |
|---|---|---|---|---|
| Grayscale | `images_grayscale` | 24 of 44 | 13 min | `runs/detect/blackjack_cards/weights/best.pt` (copied to `best.pt`) |
| Color | `images` | 24 of 44 | 13 min | `runs/detect/blackjack_cards_color/weights/best.pt` |

## Session split

Photos are grouped into sessions using the capture time in each filename. A new session starts after a gap of more than 20 minutes. Within a session, photos are at most 6.5 minutes apart; between sessions the gap is at least 56 minutes. Every session goes entirely into one set, so the test set contains only table setups the model has never seen.

| Session | Images | Setup | Set |
|---|---|---|---|
| 2026-11-13 15:02 | 35 | Black table (app overlay visible) | Train |
| 2026-11-13 17:20 | 29 | Black table, yellowish light (overlay) | Train |
| 2026-11-14 13:16 | 13 | Black table (overlay) | **Test** |
| 2026-11-16 12:33 | 40 | Black table, bluish tint | Train |
| 2026-11-23 14:25 | 42 | Dark grey surface | **Test** |
| 2026-11-23 23:09 | 109 | Light grey / whiteboard | Train |
| 2026-11-25 13:13 | 1 | Wood table | Valid |
| 2026-11-25 16:51 | 55 | Wood table | Valid |
| 2026-11-25 17:59 | 73 | Wood and grey, close-up single cards | Train |

The validation and test sessions are listed in `VALID_SESSIONS` / `TEST_SESSIONS` in `prepare_dataset.py`. Any new session goes to training automatically.

## How the metrics were scored

Both models ran on the same 55 test images (267 cards). A detection is correct only if it has the right rank **and** its box overlaps the real card by at least 50% (IoU ≥ 0.5).

- **Precision:** of the cards the model reported, how many were right.
- **Recall:** of the cards on the table, how many it found.
- **F1 score:** a single score balancing precision and recall.
- **Accuracy:** correct ÷ (correct + wrong + missed). Detection has no "correctly found nothing" case, so this is the usual stand-in.

## Overall results

| | Gray 0.50 | Gray 0.65 | Color 0.50 | Color 0.65 |
|---|---|---|---|---|
| Precision | **100%** | 100% | 100% | 100% |
| Recall | **100%** | 98.9% | 99.6% | 98.5% |
| F1 score | **100%** | 99.4% | 99.8% | 99.2% |
| Accuracy | **100%** | 98.9% | 99.6% | 98.5% |
| Correct cards (of 267) | **267** | 264 | 266 | 263 |
| Wrong cards | 0 | 0 | 0 | 0 |
| Missed cards | 0 | 3 (4 ×2, 5 ×1) | 1 (5) | 4 (5 ×3, Q ×1) |

0.50 is the confidence threshold currently set in `main.py`.

## F1 score by class

Bold marks an F1 below 0.95.

| Class | Gray 0.50 | Gray 0.65 | Color 0.50 | Color 0.65 |
|---|---|---|---|---|
| A | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 | **0.94** | 1.00 | 1.00 |
| 5 | 1.00 | 0.97 | 0.97 | **0.90** |
| 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 | 1.00 | 1.00 |
| 9 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 | 1.00 | 1.00 | 1.00 | 1.00 |
| J | 1.00 | 1.00 | 1.00 | 1.00 |
| Q | 1.00 | 1.00 | 1.00 | 0.97 |
| K | 1.00 | 1.00 | 1.00 | 1.00 |
| Back | 1.00 | 1.00 | 1.00 | 1.00 |

## What each model did well and badly

### Grayscale
- **Good:** perfect at 0.50 on every class, on two table setups it never trained on.
- **Bad:** at 0.65 it missed two 4s and one 5. Those detections were correct but had confidence between 0.50 and 0.65.

### Color
- **Good:** it reported no wrong cards at either threshold. Its face cards and card backs were perfect at 0.50.
- **Bad:** it missed one 5 at 0.50, and three 5s plus a Queen at 0.65. 5 is its weakest class.

### Raising the threshold from 0.50 to 0.65
Neither model reported a wrong card at 0.50, so a higher threshold has nothing to filter out. It only removes correct but less confident detections. **Keep 0.50.**

### Grayscale vs color
The difference is one missed card, which is within noise for a 55-image test set. Grayscale is slightly ahead, and it is the model the app is built around (`grayscale_for_model` in `main.py`).

## Classical approach: contours + HOG + SVM

Code: `classical/cards.py` (detector) and `classical/train.py` (training). The model is in `classical/card_svm.yml.gz`.

### How it works
1. **Find card-shaped regions (contour/edge detection).** The grayscale image is turned into five binary masks: an adaptive threshold, the same mask with a morphological opening, an Otsu threshold, dilated Canny edges, and a softer adaptive threshold for light backgrounds. The outer contour of each white blob is fitted with a rotated rectangle (`minAreaRect`). The rectangle is kept if it looks like a card: 1.2–15% of the image, short/long side ratio 0.5–0.9, and the contour fills at least 75% of the rectangle. Duplicates found by more than one mask are merged.
2. **Straighten each region.** The rotated rectangle is perspective-warped to an upright 120×180 card.
3. **Describe it (HOG).** The features are a HOG of the whole card at 64×96, which captures pip count and layout, plus a HOG of the top-left corner index at 32×64, which captures the rank symbol. Histogram equalization comes first, to even out lighting.
4. **Classify it (SVM).** An RBF-kernel SVM predicts one of 15 classes: the 14 card classes plus **NOT_CARD**, which rejects regions that passed the shape filter but aren't cards.

**Why contours instead of a sliding window:** cards are high-contrast rectangles, so contours find them directly, at any rotation and in one pass. A sliding window would need many window sizes and angles, and HOG isn't rotation-invariant.

### How it was trained
- **Training samples:** the same candidate finder was run on the training images. Each candidate overlapping a labeled card (IoU ≥ 0.5) became a sample of that class. Candidates overlapping nothing became NOT_CARD, along with 3 random background patches per image. Every sample was also added rotated 180°, because a card looks the same either way up. That gave 3,946 samples, including 1,750 NOT_CARD.
- **Settings:** C and gamma were chosen on the validation set from a 3×4 grid. Best: C=10, gamma=0.005, 91.1% validation accuracy on candidates. Training takes about 2.5 minutes on the CPU.
- **Same split:** it uses exactly the same session split as YOLO. The test set was not used for any tuning.

### How often it finds the cards
Before classification, the candidate finder locates 95% of labeled cards in train, 96% in validation and 99% in test. Each mask alone did worse: the best single mask found 78% in train. Each one fails on a different background:
- **App overlay:** the divider line joins cards together. The opening step fixes this.
- **Light grey / whiteboard:** cards are barely brighter than the surface. The soft threshold fixes this.
- **Striped fabric:** the stripes merge with the cards. The opening step fixes this.

### Results (test set, 267 cards)

| | YOLO grayscale (0.50) | YOLO color (0.50) | Classical |
|---|---|---|---|
| Precision | **100%** | 100% | 98.1% |
| Recall | **100%** | 99.6% | 96.6% |
| F1 score | **100%** | 99.8% | 97.4% |
| Accuracy | **100%** | 99.6% | 94.9% |
| Correct / wrong / missed | 267 / 0 / 0 | 266 / 0 / 1 | 258 / 5 / 9 |
| Speed (Apple M2) | **14 ms/image** (GPU) | — | 64 ms/image (CPU) |

The classical model has no confidence score, so the 0.50/0.65 threshold doesn't apply to it.

| Class | A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J | Q | K | Back |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Classical F1 | **0.94** | 1.00 | **0.94** | 0.97 | 0.97 | 0.95 | 0.97 | 0.97 | 0.97 | 0.97 | 1.00 | 1.00 | **0.94** | 1.00 |

**Mix-ups (true → predicted):** 6→K, 8→4, A→3, 7→6, A→5. Each happened once.

### What it did well and badly
- **Good:**
  - Card backs, J, Q, K and 2 were nearly or fully perfect. Face cards and backs have very distinctive HOG patterns.
  - It's 95% accurate with no deep learning and only OpenCV, and it trains in minutes on a CPU.
  - Every step can be inspected: the masks, the contours, the warped card and the HOG features.
- **Bad:**
  - **Wrong ranks on number cards:** A→3, A→5, 7→6 and 8→4. HOG on a small warped card blurs the pip count, and the corner index is only about 26×50 pixels.
  - **Missed cards (4):** cards whose contour merged with a neighbor or the background, so no card-shaped region was found.
  - **Hand-tuned:** the thresholds and shape limits were tuned to these tables. A new surface may need new mask settings, whereas YOLO learns this from data.
  - **Known limits:** cards turned more than about 45°, or overlapping cards, aren't handled. An overlap makes one merged blob that fails the shape filter.
  - **Slower:** about 4.5× slower than YOLO, though still fast enough for about 15 frames per second.

### Deep learning vs classical
YOLO learns where cards are and what they are from data, so it handles new backgrounds and lighting without hand-tuning. The classical pipeline depends on hand-designed steps: thresholds, shape rules and HOG features. It gets most of the way, about 95%, but its errors are the kind that matter for blackjack, wrong number-card ranks. It also breaks in cases YOLO handles, such as overlapping cards.

## Caveats

- **Small test set:** 55 images from two sessions. Differences of one or two cards are not meaningful.
- **Narrow data:** every session appears to use the same Bicycle deck, the same webcam and a similar camera angle. Performance on a different deck, a steep angle or very different lighting is untested.
- **App overlay in old captures:** the three 2026-11-13/14 sessions were captured with the app's overlay drawn on them, including old detection boxes. `main.py` now saves clean frames.
- **Live error not covered:** in live testing a 9 was read as a K, a case that never appears in the test set. To fix it, capture failing frames with **CAPTURE** in `main.py`, label them, and retrain.

## Reproducing

```bash
python greyscale.py                 # label_studio_combined/images -> images_grayscale
python prepare_dataset.py           # data/custom (grayscale, session split)
python prepare_dataset.py --color   # data/custom_color (same split, color images)
python train.py                     # grayscale model, also scores the test set
python train.py --color             # color model
python classical/train.py           # classical HOG + SVM model (~2.5 min, CPU)
python evaluate.py                  # precision/recall/F1/accuracy for all three models
python evaluate.py 0.65             # same at YOLO confidence 0.65
```

YOLO's own test-set reports (curves and confusion matrices) are in `runs/detect/blackjack_cards_test/` and `runs/detect/blackjack_cards_color_test/`. They were regenerated after the label fixes. YOLO scores differently from the tables above: it averages over all confidence levels and matches each card to its best-overlapping box regardless of confidence. Its numbers therefore differ slightly:

| YOLO report | Grayscale | Color |
|---|---|---|
| Precision | 0.981 | 0.987 |
| Recall | 0.978 | 0.987 |
| mAP50 | 0.995 | 0.995 |
| mAP50-95 | 0.748 | 0.828 |

The two "4 → 2" entries in the grayscale confusion matrix come from this matching, not from real errors. Each of those 4s was detected correctly at 0.58–0.88 confidence. The model also made a faint second guess of "2" (at most 0.23 confidence) whose box overlapped slightly more, and YOLO matched the card to that guess. Both are far below the app's 0.50 threshold. The large "background" column is the same kind of low-confidence extra boxes.

The color model's higher mAP50-95 means its boxes fit the card edges a bit more tightly. That doesn't affect which card is detected.
