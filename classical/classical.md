# Classical Card Detector: Experiments and Results

This is the classical (non-deep-learning) half of the project: **contour/edge detection** finds the cards, and an **SVM on HOG features** identifies them. This file records every experiment tried, in order, with its numbers and what was decided. The comparison with YOLO is in `../results.md`.

| File | Purpose |
|---|---|
| `cards.py` | The detector: candidate finder, warp, HOG features, `detect()` |
| `train.py` | Builds training samples, searches C/gamma on validation, saves the model |
| `card_svm.yml.gz` | The trained SVM (39 MB, gzip-compressed) |
| `main_with_classical.py` | Live app (`main.py`) running the classical detector instead of YOLO |
| `../evaluate.py` | Scores YOLO and classical on the test set with the same rules |

## Final result

On the test set (55 images, 267 cards, two table setups never seen in training), a detection counts only if it has the right class **and** overlaps the labeled card by IoU ≥ 0.5:

| | Classical | YOLO grayscale (conf 0.50) |
|---|---|---|
| Precision | 98.1% | 100% |
| Recall | 96.6% | 100% |
| F1 score | 97.4% | 100% |
| Accuracy | **94.9%** | 100% |
| Correct / wrong / missed | 258 / 5 / 9 | 267 / 0 / 0 |
| Speed (Apple M2) | 64 ms/image (CPU) | 14 ms/image (GPU) |

Accuracy = correct ÷ (correct + wrong + missed), because detection has no true negatives.

## Data

The YOLO dataset is reused unchanged: grayscale images from `data/custom` (1200×700 warped table views), whole-card boxes, 14 classes (A–K plus Back), and the **session split**. That's 286 train / 56 validation / 55 test images. The validation set is the wood table; the test set is a black table and a dark grey surface. The test set was used only for the final score, never for tuning.

---

## Pipeline

**Contour-based localization + HOG/SVM classification:**

**Contour / edge detection → warp → HOG → SVM**

The first two steps find where the cards are. The last two decide what each card is. HOG comes before the SVM: HOG turns each card image into numbers, and the SVM classifies those numbers. The SVM never sees pixels.

1. **Contour / edge detection: where are the cards?** Five binary masks are made: three adaptive thresholds, an Otsu threshold and dilated Canny edges. External contours are traced around each white region and fitted with a rotated rectangle (`minAreaRect`). A card-shape filter keeps only card-like rectangles. Canny edge detection is only one of the five masks; thresholding, which separates bright cards from the background, does most of the work.
2. **Warp: straighten each card.** Each rotated rectangle is perspective-warped to an upright 120×180 card image.
3. **HOG: describe the card.** HOG (Histogram of Oriented Gradients) converts the card into numbers that describe its edge patterns. After histogram equalization, there are two parts:
   - HOG of the whole card at 64×96, which captures the suit-symbol layout (2,772 values)
   - HOG of the top-left corner index at 32×64, which captures the rank symbol (756 values)

   That is 3,528 features per card.
4. **SVM: decide what it is.** An RBF-kernel SVM reads the HOG features and outputs one of 15 classes: the 14 card classes (A–K, Back) or **NOT_CARD**. NOT_CARD throws away regions from step 1 that passed the shape filter but aren't cards.

**Card-shape filter** (applies to every experiment below): the rotated-rectangle area must be 1.2–15% of the image, the short/long side ratio 0.5–0.9, and the contour must fill at least 75% of its rectangle.

**Why contours, not a sliding window:** cards are high-contrast rectangles, so contours find them in one pass at any angle. A sliding window needs many window sizes and angles, and HOG isn't rotation-invariant.

---

## Experiment 1: Single mask for finding candidates

Can one binary mask locate the cards? The score is the share of labeled cards matched by a candidate box (IoU ≥ 0.5), ignoring class. "Extra" counts candidate boxes that match no card.

| Mask | Train | Valid | Test |
|---|---|---|---|
| Canny edges (30/90, dilated ×2) | 62% (+17 extra) | 31% (+0) | 90% (+1) |
| Otsu threshold | 65% (+2) | 80% (+7) | 97% (+1) |
| Adaptive threshold (block 151, C −10) | **78%** (+5) | **95%** (+0) | **98%** (+0) |

**Decision:** adaptive threshold is the best single mask, but 78% on train is too low.

## Experiment 2: Where does the adaptive mask fail?

Adaptive-mask hit rate per training session:

| Session | Background | Cards found |
|---|---|---|
| 2026-11-13 15:02 | Black, app overlay | 112/161 (70%) |
| 2026-11-13 17:20 | Black, app overlay | 92/141 (65%) |
| 2026-11-16 12:33 | Black | 208/215 (97%) |
| 2026-11-23 23:09 | Light grey / whiteboard | 403/537 (75%) |
| 2026-11-25 17:59 | Striped fabric / wood | 88/106 (83%) |

Looking at the failed images showed three causes:
- **App overlay:** the cyan dealer/player divider line joins neighboring cards into one blob.
- **Whiteboard:** cards are barely brighter than the surface, so suit symbols break each card's blob into fragments.
- **Striped fabric:** stripes pass the threshold and merge with the cards.

## Experiment 3: Morphology on the adaptive mask

Opening (erode then dilate) should cut thin lines and stripes. Closing (dilate then erode) should fill the gaps left by suit symbols. Each row is block size, C, opening kernel size, closing kernel size (0 = off).

| Block | C | Open | Close | Train | Valid | Test |
|---|---|---|---|---|---|---|
| 151 | −10 | 0 | 0 | **78%** (+5) | 95% (+0) | 98% (+0) |
| 151 | −10 | 5 | 0 | 75% (+21) | **96%** (+0) | **99%** (+0) |
| 151 | −10 | 5 | 15 | 70% (+17) | 87% (+0) | 91% (+2) |
| 151 | −10 | 7 | 21 | 63% (+27) | 82% (+0) | 92% (+3) |
| 101 | −5 | 5 | 15 | 72% (+5) | 69% (+0) | 89% (+3) |
| 201 | −10 | 5 | 15 | 73% (+9) | 85% (+0) | 91% (+2) |
| 151 | −5 | 5 | 21 | 66% (+15) | 58% (+1) | 83% (+3) |
| 151 | −15 | 5 | 15 | 64% (+10) | 90% (+0) | 91% (+2) |

**Decision:** closing always hurt, because it also joins neighboring cards. Opening helped validation and test but not train. No single setting fixed every background.

## Experiment 4: Combining several masks

Each mask fails on a different background, so all five masks are run and their candidates combined. Duplicates are merged: at IoU > 0.7 in this prototype, and at IoU > 0.5 in the final `cards.py`. The SVM's NOT_CARD class then removes the extras.

| Mask alone | Train | Valid | Test |
|---|---|---|---|
| Adaptive (151, −10) | 78% | 95% | 98% |
| Adaptive + opening 5 | 75% | 96% | 99% |
| Otsu | 65% | 80% | 97% |
| Canny | 62% | 31% | 90% |
| Soft adaptive (block 301, C −3) + opening 5, aimed at the whiteboard | 88% | 92% | 99% |
| **All five combined** | **95%** (+41 extra) | **96%** (+8) | **99%** (+1) |

**Decision:** use all five masks. This raised training coverage from 78% to 95%. The remaining misses are mostly cards touching or overlapping another card.

## Experiment 5: Card orientation in the warp (bug found and fixed)

**First version:** the warp made the rectangle's long side vertical, so every card came out portrait.

**Problem:** many warped cards had their rank index in the top-right corner, which looks like a mirrored card. Checking the raw crops showed the source images were fine. The cause is the app's table calibration, a perspective warp to 1200×700, which stretches cards differently in each session. One upright card measured **315 px wide × 242 px tall**, so the long-side rule rotated it 90°.

**Fix:** drop the long-side rule and keep the edge closest to horizontal on top. Synthetic tests at 0°, −3°, −88°, 30° and −60° all came out with the marked corner at the top-left. Real cards then came out upright, with the index inside the corner crop.

**Known limit:** a card turned more than about 45° comes out sideways. This is noted as a `shortcut:` comment in `cards.py`.

## Experiment 6: SVM hyperparameter search, round 1

**Training samples:**
- Candidates on training images that overlap a labeled card (IoU ≥ 0.5) take that card's class.
- Candidates with IoU < 0.3 against every card are NOT_CARD. Those between 0.3 and 0.5 are skipped as ambiguous.
- Three random card-sized background patches per image are added as NOT_CARD.
- Every sample is also added rotated 180°.

That gave **3,946 training samples**: about 130–190 per rank, 296 Back and 1,750 NOT_CARD. Validation has 336 samples. The score is the classification accuracy on validation candidates.

| C \ gamma | 0.01 | 0.05 | 0.2 |
|---|---|---|---|
| 1 | 78.6% | 82.1% | 28.0% |
| 10 | **90.8%** | 83.0% | 28.9% |
| 100 | 90.8% | 83.0% | 28.9% |

**Observations:**
- **gamma = 0.2 collapses (about 28%).** With 3,528 features, a large gamma makes the RBF kernel so narrow that every validation sample looks unlike every training sample.
- **Smaller gamma is better.** The best value, 0.01, was at the edge of the grid, so a second round searched lower.

## Experiment 7: SVM hyperparameter search, round 2

| C \ gamma | 0.002 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|
| 3 | 74.4% | 86.3% | 90.2% | 90.2% |
| 10 | 88.7% | **91.1%** | 90.8% | 89.9% |
| 100 | 90.8% | **91.1%** | 90.8% | 89.9% |

**Decision:** **C = 10, gamma = 0.005** (91.1%). Results level off around 90–91% for C ≥ 10 and gamma between 0.005 and 0.01, so the choice isn't sensitive. Each round of 9–12 SVMs took about 2.5 minutes on the CPU.

## Experiment 8: Model file size

The RBF SVM keeps **2,001 support vectors × 3,528 features**. Saved as OpenCV YAML text, that is **111 MB**, over GitHub's 100 MB file limit. OpenCV reads and writes gzip directly, so the model is saved as `card_svm.yml.gz` (**39 MB**). Test scores were identical after reloading.

## Experiment 9: Final test-set evaluation

`python evaluate.py` scores all three models with the same rules.

| Class | F1 | Wrong (FP) | Missed (FN) |
|---|---|---|---|
| A | **0.94** | 0 | 2 |
| 2 | 1.00 | 0 | 0 |
| 3 | **0.94** | 1 | 1 |
| 4 | 0.97 | 1 | 0 |
| 5 | 0.97 | 1 | 0 |
| 6 | 0.95 | 1 | 1 |
| 7 | 0.97 | 0 | 1 |
| 8 | 0.97 | 0 | 1 |
| 9 | 0.97 | 0 | 1 |
| 10 | 0.97 | 0 | 1 |
| J | 1.00 | 0 | 0 |
| Q | 1.00 | 0 | 0 |
| K | **0.94** | 1 | 1 |
| Back | 1.00 | 0 | 0 |

**Mix-ups (true → predicted), once each:** 6→K, 8→4, A→3, 7→6, A→5. These account for 5 of the 9 missed cards. The other 4 were never found as candidates, or were rejected as NOT_CARD.

**Speed:** 64 ms per image, against 14 ms for YOLO.

---

## What worked and what didn't

**Worked:**
- **Combining several masks.** It was the biggest single gain, raising training coverage from 78% to 95%.
- **The NOT_CARD class.** It lets the candidate finder be generous, because the SVM throws away the extras. Precision stays at 98%.
- **The corner-index HOG plus 180° augmentation.** Cards are point-symmetric, so the top-left corner always holds an index.
- **Face cards and backs:** J, Q, Back and 2 were perfect, because their HOG patterns are very distinctive.

**Didn't work, or limits:**
- **Morphological closing.** It joins neighboring cards.
- **"Long side vertical" orientation.** The table calibration distorts card proportions.
- **Number-card ranks:** A→3, A→5, 7→6 and 8→4. HOG on a small warp blurs the pip count, and the corner index is only about 26×50 pixels.
- **Overlapping cards.** They become one blob that fails the shape filter. The design doesn't handle them.
- **Cards turned more than about 45°.** They come out sideways.
- **Hand-tuned settings.** Thresholds, block sizes and shape limits were tuned to these backgrounds. A new surface may need retuning, whereas YOLO learns this from data.

## Ideas not tried

- **Larger warp or corner crop:** for example 180×270 with a 40×75 corner, to sharpen the rank index. This would likely help A, 3 and 4.
- **Rotation augmentation (90°):** so that cards turned more than 45° are read correctly.
- **A linear SVM:** about 100× smaller model and faster, probably a few points less accurate.
- **Separating touching cards:** watershed or a distance transform, to split merged blobs.

## Reproducing

```bash
python prepare_dataset.py                    # session split -> data/custom
python classical/train.py                    # experiments 6-7 grid search, saves card_svm.yml.gz
python evaluate.py                           # experiment 9 test scores (YOLO + classical)
python classical/main_with_classical.py      # live app with the classical detector
```

Experiments 1–4 were run with prototype scripts that weren't kept. Their settings are the ones now built into `cards.py` (`_masks`, `find_candidates`). `classical/train.py` currently searches the round-2 grid (C ∈ {3, 10, 100}, gamma ∈ {0.002, 0.005, 0.01, 0.02}).
