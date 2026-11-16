import cv2
from ultralytics import YOLO
import numpy as np
import sys
import time
import webbrowser
import subprocess
from pathlib import Path
from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------

DISPLAY_CLASS_CORRECTIONS = {
    "10": "A",
    "A": "10",
    "J": "Back",
    "Q": "J",
    "Back": "Q",
}

MODEL_PATH = (
    "runs/detect/custom_blackjack_cards-clean/weights/best.pt"
)
CONFIDENCE_THRESHOLD = 0.50

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 675

TABLE_WIDTH = 1200
TABLE_HEIGHT = 700

BOTTOM_BAR_HEIGHT = 70

USE_LIGHTING_CORRECTION = False

CAPTURE_DIR = Path("data/custom/raw_images")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Global state
# -----------------------------

latest_frame = None
status_message = ""

labelstudio_started = False
training_started = False

zoom_level = 1.0
running = True

button_locations = {}

calibrating = False
calibration_points = []
perspective_matrix = None

current_view_height = DISPLAY_HEIGHT


button_definitions = [
    ("calibrate", "CALIBRATE", 120, (120, 70, 0)),
    ("capture", "CAPTURE", 105, (0, 120, 0)),
    ("labelstudio", "LABEL", 90, (100, 70, 0)),
    ("train", "TRAIN", 80, (120, 70, 0)),
    ("minus", "-", 50, (0, 120, 0)),
    ("plus", "+", 50, (0, 120, 0)),
    ("quit", "QUIT", 85, (0, 0, 180)),
]


# -----------------------------
# Camera/image functions
# -----------------------------

def grayscale_for_model(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert back to 3 channels because YOLO expects 3-channel input
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def improve_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gamma = 1.25

    lookup_table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(enhanced, lookup_table)


def apply_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame

    height, width = frame.shape[:2]

    crop_width = int(width / zoom)
    crop_height = int(height / zoom)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2

    cropped = frame[
        y1:y1 + crop_height,
        x1:x1 + crop_width
    ]

    return cv2.resize(cropped, (width, height))


def calculate_perspective_transform(points):
    source_points = np.float32(points)

    destination_points = np.float32([
        [0, 0],
        [TABLE_WIDTH - 1, 0],
        [TABLE_WIDTH - 1, TABLE_HEIGHT - 1],
        [0, TABLE_HEIGHT - 1]
    ])

    return cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )


# -----------------------------
# Button actions
# -----------------------------

def start_calibration():
    global calibrating
    global calibration_points
    global status_message

    calibrating = True
    calibration_points = []

    status_message = (
        "Click table corners: "
        "TL, TR, BR, BL"
    )


def capture_image():
    global latest_frame
    global status_message

    if latest_frame is None:
        status_message = "No table image available"
        return

    filename = datetime.now().strftime(
        "table_%Y%m%d_%H%M%S.jpg"
    )

    save_path = CAPTURE_DIR / filename

    cv2.imwrite(str(save_path), latest_frame)

    status_message = f"Saved {filename}"


def open_label_studio():
    global labelstudio_started
    global status_message

    try:
        if not labelstudio_started:
            creation_flags = getattr(
                subprocess,
                "CREATE_NEW_CONSOLE",
                0
            )

            subprocess.Popen(
                [
                    "label-studio",
                    "start",
                    "--port",
                    "8080"
                ],
                creationflags=creation_flags
            )

            labelstudio_started = True
            time.sleep(3)

        webbrowser.open("http://localhost:8080")
        status_message = "Label Studio opened"

    except FileNotFoundError:
        status_message = (
            "Label Studio not found. "
            "Run: pip install label-studio"
        )


def start_training():
    global training_started
    global status_message

    if training_started:
        status_message = "Training is already running"
        return

    if not Path("data/custom/data.yaml").exists():
        status_message = "Missing data/custom/data.yaml"
        return

    try:
        creation_flags = getattr(
            subprocess,
            "CREATE_NEW_CONSOLE",
            0
        )

        subprocess.Popen(
            [sys.executable, "train_custom.py"],
            creationflags=creation_flags
        )

        training_started = True
        status_message = "Training started"

    except Exception as error:
        status_message = f"Training error: {error}"


# -----------------------------
# Button drawing
# -----------------------------

def draw_button(frame, text, location, color):
    x1, y1, x2, y2 = location

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2
    )[0]

    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


def draw_bottom_bar(frame):
    global button_locations

    height, width = frame.shape[:2]

    canvas = np.zeros(
        (
            height + BOTTOM_BAR_HEIGHT,
            width,
            3
        ),
        dtype=np.uint8
    )

    canvas[:height] = frame

    cv2.rectangle(
        canvas,
        (0, height),
        (width, height + BOTTOM_BAR_HEIGHT),
        (35, 35, 35),
        -1
    )

    button_locations = {}

    button_height = 45
    spacing = 10
    top = height + 12
    current_x = width - 15

    # Reverse so buttons appear left-to-right properly
    for key, text, button_width, color in reversed(
        button_definitions
    ):
        x2 = current_x
        x1 = x2 - button_width

        button_locations[key] = (
            x1,
            top,
            x2,
            top + button_height
        )

        draw_button(
            canvas,
            text,
            button_locations[key],
            color
        )

        current_x = x1 - spacing

    return canvas


# -----------------------------
# Mouse interaction
# -----------------------------

def mouse_callback(event, x, y, flags, parameter):
    global zoom_level
    global running
    global calibration_points
    global perspective_matrix
    global calibrating
    global status_message

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Handle buttons first
    for key, location in button_locations.items():
        x1, y1, x2, y2 = location

        if x1 <= x <= x2 and y1 <= y <= y2:

            if key == "calibrate":
                start_calibration()

            elif key == "capture":
                capture_image()

            elif key == "labelstudio":
                open_label_studio()

            elif key == "train":
                start_training()

            elif key == "minus":
                zoom_level = max(
                    1.0,
                    zoom_level - 0.25
                )

            elif key == "plus":
                zoom_level = min(
                    2.5,
                    zoom_level + 0.25
                )

            elif key == "quit":
                running = False

            return

    # During calibration, record clicks on the camera image
    if calibrating and y < current_view_height:

        if len(calibration_points) < 4:
            calibration_points.append((x, y))

            status_message = (
                f"Selected point "
                f"{len(calibration_points)}/4"
            )

        if len(calibration_points) == 4:
            perspective_matrix = (
                calculate_perspective_transform(
                    calibration_points
                )
            )

            calibrating = False
            status_message = (
                "Table calibrated successfully"
            )


# -----------------------------
# Drawing detections
# -----------------------------

def detect_cards(model, model_frame, display_frame):
    results = model.predict(
        source=model_frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    dealer_cards = []
    player_cards = []

    frame_height, frame_width = display_frame.shape[:2]
    divider_y = frame_height // 2

    # Dealer/player divider
    cv2.line(
        display_frame,
        (0, divider_y),
        (frame_width, divider_y),
        (255, 255, 0),
        3
    )

    cv2.putText(
        display_frame,
        "DEALER",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "PLAYER",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            card_name = model.names[class_id]

            card_name = DISPLAY_CLASS_CORRECTIONS.get(
                card_name,
                card_name
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center_y = (y1 + y2) // 2

            if center_y < divider_y:
                dealer_cards.append(card_name)
                color = (0, 165, 255)  # Orange
            else:
                player_cards.append(card_name)
                color = (0, 255, 0)  # Green

            label = (
                f"{card_name} "
                f"{confidence * 100:.1f}%"
            )

            # Draw on the color display frame
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                display_frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    dealer_text = (
        f"Dealer cards: {len(dealer_cards)}"
    )

    if len(dealer_cards) == 1:
        dealer_text += " | Face-down assumed"

    player_text = (
        f"Player cards: {len(player_cards)}"
    )

    cv2.putText(
        display_frame,
        dealer_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2
    )

    cv2.putText(
        display_frame,
        player_text,
        (20, frame_height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    return display_frame


def add_raw_camera_hud(table_frame, raw_frame):
    hud_width = 260
    hud_height = 146

    hud = cv2.resize(
        raw_frame,
        (hud_width, hud_height)
    )

    cv2.rectangle(
        hud,
        (0, 0),
        (hud_width - 1, hud_height - 1),
        (255, 255, 255),
        2
    )

    cv2.putText(
        hud,
        "RAW CAMERA",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y1 = 10
    x1 = table_frame.shape[1] - hud_width - 10

    table_frame[
        y1:y1 + hud_height,
        x1:x1 + hud_width
    ] = hud

    return table_frame


# -----------------------------
# Main program
# -----------------------------

def main():
    global latest_frame
    global running
    global zoom_level
    global calibrating
    global calibration_points
    global perspective_matrix
    global current_view_height
    global status_message

    model = YOLO(MODEL_PATH)

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    print(
        "Camera resolution:",
        camera.get(cv2.CAP_PROP_FRAME_WIDTH),
        "x",
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    window_name = "Blackjack Card Detection"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT + BOTTOM_BAR_HEIGHT
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print("Camera started.")
    print("Click CALIBRATE to select the table corners.")
    print("Press Q to quit.")

    while running:
        success, original_frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Resize raw frame for consistent calibration coordinates
        raw_display = cv2.resize(
            original_frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        # Calibration mode
        if calibrating:
            view = raw_display.copy()

            cv2.putText(
                view,
                "Click: TL, TR, BR, BL",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            for index, point in enumerate(
                calibration_points
            ):
                cv2.circle(
                    view,
                    point,
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    view,
                    str(index + 1),
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            current_view_height = view.shape[0]

            view = draw_bottom_bar(view)

            cv2.imshow(
                window_name,
                view
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                running = False

            continue

        # If table has not been calibrated yet
        if perspective_matrix is None:
            view = raw_display.copy()

            cv2.putText(
                view,
                "Click CALIBRATE to select the table",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            current_view_height = view.shape[0]

            view = draw_bottom_bar(view)

            cv2.imshow(
                window_name,
                view
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                running = False

            continue

        # Warp the selected table into a rectangle
        table_frame = cv2.warpPerspective(
            raw_display,
            perspective_matrix,
            (TABLE_WIDTH, TABLE_HEIGHT)
        )

        # Apply optional lighting correction
        if USE_LIGHTING_CORRECTION:
            table_frame = improve_lighting(
                table_frame
            )

        # Apply zoom after calibration
        table_frame = apply_zoom(
            table_frame,
            zoom_level
        )

        # Save clean transformed image for capture
        latest_frame = table_frame.copy()

        # Create grayscale input for the model
        model_frame = grayscale_for_model(table_frame)

        # Detect using grayscale, but draw on the color table frame
        table_frame = detect_cards(
            model,
            model_frame,
            table_frame
        )

        # Add raw camera preview
        table_frame = add_raw_camera_hud(
            table_frame,
            raw_display
        )

        # Add status text
        if status_message:
            cv2.putText(
                table_frame,
                status_message,
                (20, TABLE_HEIGHT - 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.putText(
            table_frame,
            f"Zoom: {zoom_level:.2f}x",
            (20, TABLE_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        current_view_height = table_frame.shape[0]

        # Add bottom buttons
        display_frame = draw_bottom_bar(
            table_frame
        )

        cv2.imshow(
            window_name,
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            running = False

        elif key == ord("c"):
            start_calibration()

        elif key in [ord("+"), ord("=")]:
            zoom_level = min(
                2.5,
                zoom_level + 0.25
            )

        elif key == ord("-"):
            zoom_level = max(
                1.0,
                zoom_level - 0.25
            )

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()