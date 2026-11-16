from pathlib import Path
import cv2


SOURCE_DIR = Path(
    r"C:\Users\abcar\Desktop\BlackJackCV\data\custom\raw_images"
)

OUTPUT_DIR = Path(
    r"C:\Users\abcar\Desktop\BlackJackCV\data\custom\grayscale_images"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

converted = 0
failed = 0

for image_path in SOURCE_DIR.rglob("*"):
    if image_path.suffix.lower() not in valid_extensions:
        continue

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print(f"Could not read: {image_path}")
        failed += 1
        continue

    # Convert to grayscale
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Convert back to 3 channels so YOLO can use it
    grayscale_3channel = cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2BGR
    )

    relative_path = image_path.relative_to(SOURCE_DIR)
    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        grayscale_3channel
    )

    converted += 1
    print(f"Converted: {image_path.name}")


print()
print(f"Finished. Converted: {converted}")
print(f"Failed: {failed}")
print(f"Saved to: {OUTPUT_DIR}")