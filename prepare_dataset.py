from pathlib import Path
import random
import shutil


PROJECT_DIR = Path(__file__).resolve().parent

# Automatically check several possible grayscale folder names
possible_image_folders = [
    PROJECT_DIR / "data" / "custom" / "grayscale_images",
]

IMAGE_SOURCE = None

for folder in possible_image_folders:
    if folder.exists():
        IMAGE_SOURCE = folder
        break

if IMAGE_SOURCE is None:
    raise FileNotFoundError(
        "Could not find the grayscale image folder."
    )

LABEL_SOURCE = (
    PROJECT_DIR
    / "data"
    / "label_studio_import"
    / "labels"
)

OUTPUT_ROOT = (
    PROJECT_DIR
    / "data"
    / "custom"
)

TRAIN_IMAGES = OUTPUT_ROOT / "train" / "images"
TRAIN_LABELS = OUTPUT_ROOT / "train" / "labels"

VALID_IMAGES = OUTPUT_ROOT / "valid" / "images"
VALID_LABELS = OUTPUT_ROOT / "valid" / "labels"

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


def create_directories():
    for directory in [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VALID_IMAGES,
        VALID_LABELS,
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
    yaml_text += "val: valid/images\n\n"
    yaml_text += f"nc: {len(CLASS_NAMES)}\n\n"
    yaml_text += "names:\n"

    for class_name in CLASS_NAMES:
        yaml_text += f"  - {class_name}\n"

    data_yaml_path = OUTPUT_ROOT / "data.yaml"

    data_yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )


def main():
    create_directories()

    pairs = find_pairs()

    if not pairs:
        raise RuntimeError(
            "No matching image/label pairs were found."
        )

    # Reproducible random split
    random.seed(42)
    random.shuffle(pairs)

    split_index = int(len(pairs) * 0.80)

    train_pairs = pairs[:split_index]
    valid_pairs = pairs[split_index:]

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

    write_data_yaml()

    print()
    print("Dataset preparation complete.")
    print(f"Images used: {len(pairs)}")
    print(f"Training images: {len(train_pairs)}")
    print(f"Validation images: {len(valid_pairs)}")
    print(f"Image source: {IMAGE_SOURCE}")
    print(f"Labels source: {LABEL_SOURCE}")
    print(f"Dataset output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()