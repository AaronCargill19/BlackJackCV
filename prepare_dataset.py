from datetime import datetime
from pathlib import Path
import shutil
import sys


PROJECT_DIR = Path(__file__).resolve().parent

# Run with --color to build a dataset from the original color images
COLOR = "--color" in sys.argv

IMAGE_SOURCE = PROJECT_DIR / "label_studio_combined" / (
    "images" if COLOR else "images_grayscale"
)

if not IMAGE_SOURCE.exists():
    raise FileNotFoundError(
        f"Could not find the image folder: {IMAGE_SOURCE}"
    )

LABEL_SOURCE = PROJECT_DIR / "label_studio_combined" / "labels"

# Split by capture session (see split_by_session). Keys are session start times.
SESSION_GAP_MINUTES = 20
VALID_SESSIONS = {
    "20261125_1313",  # wood table, 1 image (same setup as the next session)
    "20261125_1651",  # wood table
}
TEST_SESSIONS = {
    "20261114_1316",  # black table
    "20261123_1425",  # dark grey surface
}

OUTPUT_ROOT = (
    PROJECT_DIR
    / "data"
    / ("custom_color" if COLOR else "custom")
)

TRAIN_IMAGES = OUTPUT_ROOT / "train" / "images"
TRAIN_LABELS = OUTPUT_ROOT / "train" / "labels"

VALID_IMAGES = OUTPUT_ROOT / "valid" / "images"
VALID_LABELS = OUTPUT_ROOT / "valid" / "labels"

TEST_IMAGES = OUTPUT_ROOT / "test" / "images"
TEST_LABELS = OUTPUT_ROOT / "test" / "labels"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

CLASS_NAMES = [
    "A",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "Back",
]

# Label Studio's export defines the class ids in the label files; they must match.
_exported = (PROJECT_DIR / "label_studio_combined" / "classes.txt").read_text().split()
if _exported != CLASS_NAMES:
    raise ValueError(f"label_studio_combined/classes.txt {_exported} does not match CLASS_NAMES {CLASS_NAMES}")


def create_directories():
    # Clear previous splits so images from an old split can't leak into another set
    for split in ["train", "valid", "test"]:
        shutil.rmtree(OUTPUT_ROOT / split, ignore_errors=True)

    for directory in [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VALID_IMAGES,
        VALID_LABELS,
        TEST_IMAGES,
        TEST_LABELS,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


def get_match_key(path):
    """
    Converts both naming styles into the same key.

    Example:
    0be24f36-table_20261114_131832.txt
    table_20261114_131832.jpg

    Both become:
    20261114_131832
    """

    filename = path.stem.lower()

    if "-table_" in filename:
        filename = filename.split(
            "-table_",
            1
        )[1]

    elif filename.startswith("table_"):
        filename = filename.replace(
            "table_",
            "",
            1
        )

    return filename


def find_images():
    image_lookup = {}

    for image_path in IMAGE_SOURCE.rglob("*"):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            match_key = get_match_key(image_path)
            image_lookup[match_key] = image_path

    return image_lookup


def find_pairs():
    image_lookup = find_images()
    pairs = []
    missing_images = []

    for label_path in LABEL_SOURCE.glob("*.txt"):
        match_key = get_match_key(label_path)

        image_path = image_lookup.get(match_key)

        if image_path is None:
            missing_images.append(label_path.name)
            continue

        pairs.append((image_path, label_path))

    print(f"Matching pairs found: {len(pairs)}")

    if missing_images:
        print(
            f"Labels without matching images: "
            f"{len(missing_images)}"
        )

        for filename in missing_images[:5]:
            print(f"Missing image for: {filename}")

    return pairs


def copy_pair(
    image_path,
    label_path,
    image_dir,
    label_dir
):
    # Copy the image using its original name
    shutil.copy2(
        image_path,
        image_dir / image_path.name
    )

    # Rename the label to match the image filename
    matching_label_name = (
        image_path.stem + ".txt"
    )

    shutil.copy2(
        label_path,
        label_dir / matching_label_name
    )


def write_data_yaml():
    yaml_text = "train: train/images\n"
    yaml_text += "val: valid/images\n"
    yaml_text += "test: test/images\n\n"
    yaml_text += f"nc: {len(CLASS_NAMES)}\n\n"
    yaml_text += "names:\n"

    for class_name in CLASS_NAMES:
        yaml_text += f"  - {class_name}\n"

    data_yaml_path = OUTPUT_ROOT / "data.yaml"

    data_yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )


def capture_time(path):
    return datetime.strptime(get_match_key(path), "%Y%m%d_%H%M%S")


def split_by_session(pairs):
    """
    Keeps every image from one capture session in the same set, so near-identical
    frames can't appear in both train and test. A new session starts after a
    gap of more than SESSION_GAP_MINUTES between captures. Sessions are named by
    their first capture time; any session not listed below goes to train.
    """
    pairs = sorted(pairs, key=lambda pair: capture_time(pair[0]))

    sessions = []
    previous = None
    for pair in pairs:
        current = capture_time(pair[0])
        if previous is None or (current - previous).total_seconds() > SESSION_GAP_MINUTES * 60:
            sessions.append((current.strftime("%Y%m%d_%H%M"), []))
        sessions[-1][1].append(pair)
        previous = current

    train_pairs, valid_pairs, test_pairs = [], [], []
    print("Sessions:")
    for start, session_pairs in sessions:
        if start in TEST_SESSIONS:
            target, name = test_pairs, "test"
        elif start in VALID_SESSIONS:
            target, name = valid_pairs, "valid"
        else:
            target, name = train_pairs, "train"
        target.extend(session_pairs)
        print(f"  {start}  {len(session_pairs):4d} images -> {name}")

    missing = (TEST_SESSIONS | VALID_SESSIONS) - {start for start, _ in sessions}
    if missing:
        raise RuntimeError(f"Listed sessions not found in the data: {sorted(missing)}")

    return train_pairs, valid_pairs, test_pairs


def main():
    create_directories()

    pairs = find_pairs()

    if not pairs:
        raise RuntimeError(
            "No matching image/label pairs were found."
        )

    train_pairs, valid_pairs, test_pairs = split_by_session(pairs)

    for image_path, label_path in train_pairs:
        copy_pair(
            image_path,
            label_path,
            TRAIN_IMAGES,
            TRAIN_LABELS
        )

    for image_path, label_path in valid_pairs:
        copy_pair(
            image_path,
            label_path,
            VALID_IMAGES,
            VALID_LABELS
        )

    for image_path, label_path in test_pairs:
        copy_pair(
            image_path,
            label_path,
            TEST_IMAGES,
            TEST_LABELS
        )

    write_data_yaml()

    print()
    print("Dataset preparation complete.")
    print(f"Images used: {len(pairs)}")
    print(f"Training images: {len(train_pairs)}")
    print(f"Validation images: {len(valid_pairs)}")
    print(f"Test images: {len(test_pairs)}")
    print(f"Image source: {IMAGE_SOURCE}")
    print(f"Labels source: {LABEL_SOURCE}")
    print(f"Dataset output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()