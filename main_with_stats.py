import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import webbrowser

import cv2
import numpy as np
from ultralytics import YOLO

from blackjack_stats import RANKS, Shoe, build_snapshot
from card_tracker import CardTracker
from stats_panel import render_stats_panel

try:
    from gui import render_dashboard, update_hand_memory
    HAS_LEGACY_GUI = True
except ImportError:
    HAS_LEGACY_GUI = False


# -----------------------------
# Configuration
# -----------------------------

# Paths are relative to this file, so the app runs from any working directory
PROJECT_DIR = Path(__file__).resolve().parent

# Empty: the previous model mislabeled some classes and needed swaps here; the
# current best.pt (trained from yolo11n.pt on label_studio_combined) outputs correct names
DISPLAY_CLASS_CORRECTIONS = {}

SUIT_LETTER_TO_NAME = {
    "C": "clubs",
    "D": "diamonds",
    "H": "hearts",
    "S": "spades",
}

MODEL_PATH = PROJECT_DIR / "best.pt"
CONFIDENCE_THRESHOLD = 0.50

SHOE_DECKS = 1
COUNT_SYSTEM = "hi_lo"  # "hi_lo", "ko", or "omega_ii"

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 675

TABLE_WIDTH = 1200
TABLE_HEIGHT = 700

STATS_PANEL_WIDTH = 380
BOTTOM_BAR_HEIGHT = 70

USE_LIGHTING_CORRECTION = False
USE_LEGACY_GUI = False
SHOW_STATS_PANEL = True

CAPTURE_DIR = PROJECT_DIR / "data/custom/raw_images"
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
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def improve_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    gamma = 1.25
    lookup_table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")
    return cv2.LUT(enhanced, lookup_table)


def apply_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame

    height, width = frame.shape[:2]
    crop_width = int(width / zoom)
    crop_height = int(height / zoom)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2

    cropped = frame[y1:y1 + crop_height, x1:x1 + crop_width]
    return cv2.resize(cropped, (width, height))


def calculate_perspective_transform(points):
    source_points = np.float32(points)
    destination_points = np.float32([
        [0, 0],
        [TABLE_WIDTH - 1, 0],
        [TABLE_WIDTH - 1, TABLE_HEIGHT - 1],
        [0, TABLE_HEIGHT - 1],
    ])
    return cv2.getPerspectiveTransform(source_points, destination_points)


# -----------------------------
# Button actions
# -----------------------------

def start_calibration():
    global calibrating, calibration_points, status_message
    calibrating = True
    calibration_points = []
    status_message = "Click table corners: TL, TR, BR, BL"


def capture_image():
    global latest_frame, status_message
    if latest_frame is None:
        status_message = "No table image available"
        return

    filename = datetime.now().strftime("table_%Y%m%d_%H%M%S.jpg")
    save_path = CAPTURE_DIR / filename
    cv2.imwrite(str(save_path), latest_frame)
    status_message = f"Saved {filename}"


def open_label_studio():
    global labelstudio_started, status_message
    try:
        if not labelstudio_started:
            creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen(
                ["label-studio", "start", "--port", "8080"],
                creationflags=creation_flags,
            )
            labelstudio_started = True
            time.sleep(3)

        webbrowser.open("http://localhost:8080")
        status_message = "Label Studio opened"
    except FileNotFoundError:
        status_message = "Label Studio not found. Run: pip install label-studio"


def start_training():
    global training_started, status_message
    if training_started:
        status_message = "Training is already running"
        return

    if not (PROJECT_DIR / "data/custom/data.yaml").exists():
        status_message = "Missing data/custom/data.yaml (run prepare_dataset.py)"
        return

    try:
        creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "train.py")],
            creationflags=creation_flags,
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
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def draw_bottom_bar(frame):
    global button_locations
    height, width = frame.shape[:2]

    canvas = np.zeros((height + BOTTOM_BAR_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = frame

    cv2.rectangle(
        canvas,
        (0, height),
        (width, height + BOTTOM_BAR_HEIGHT),
        (35, 35, 35),
        -1,
    )

    button_locations = {}
    button_height = 45
    spacing = 10
    top = height + 12
    current_x = width - 15

    for key, text, button_width, color in reversed(button_definitions):
        x2 = current_x
        x1 = x2 - button_width
        button_locations[key] = (x1, top, x2, top + button_height)
        draw_button(canvas, text, button_locations[key], color)
        current_x = x1 - spacing

    return canvas


# -----------------------------
# Mouse interaction
# -----------------------------

def mouse_callback(event, x, y, flags, parameter):
    global zoom_level, running, calibration_points, perspective_matrix, calibrating, status_message

    if event != cv2.EVENT_LBUTTONDOWN:
        return

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
                zoom_level = max(1.0, zoom_level - 0.25)
            elif key == "plus":
                zoom_level = min(2.5, zoom_level + 0.25)
            elif key == "quit":
                running = False
            return

    if calibrating and y < current_view_height and x < TABLE_WIDTH:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            status_message = f"Selected point {len(calibration_points)}/4"

        if len(calibration_points) == 4:
            perspective_matrix = calculate_perspective_transform(calibration_points)
            calibrating = False
            status_message = "Table calibrated successfully"


# -----------------------------
# Detection & tracking
# -----------------------------

def collect_detections(model, model_frame, color_frame):
    results = model.predict(
        source=model_frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,
        agnostic_nms=False,
        max_det=20,
        verbose=False,
    )

    boxes = [box for result in results for box in result.boxes if float(box.conf[0]) >= CONFIDENCE_THRESHOLD]
    

    detections = []
    for box in boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        rank = DISPLAY_CLASS_CORRECTIONS.get(model.names[class_id], model.names[class_id])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        detections.append({
            "rank": rank,
            "conf": confidence,
            "box": (x1, y1, x2, y2),
        })

    return detections


def draw_zones_and_boxes(display_frame, detections, tracker):
    frame_height, frame_width = display_frame.shape[:2]
    divider_y = frame_height // 2

    cv2.line(display_frame, (0, divider_y), (frame_width, divider_y), (255, 255, 0), 3)
    cv2.putText(display_frame, "DEALER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(display_frame, "PLAYER", (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        center_y = (y1 + y2) // 2
        color = (0, 165, 255) if center_y < divider_y else (0, 255, 0)

        
        label = f"{det['rank']} {det['conf'] * 100:.1f}%"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display_frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    tracker.draw(display_frame)

    dealer_ranks, player_ranks = tracker.hands()
    dealer_text = f"Dealer cards (active): {len(dealer_ranks)}"
    if len(dealer_ranks) == 1:
        dealer_text += " | Face-down assumed"
    player_text = f"Player cards (active): {len(player_ranks)}"

    cv2.putText(display_frame, dealer_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
    cv2.putText(display_frame, player_text, (20, frame_height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return display_frame, dealer_ranks, player_ranks


def add_raw_camera_hud(table_frame, raw_frame):
    hud_width = 260
    hud_height = 146

    hud = cv2.resize(raw_frame, (hud_width, hud_height))
    cv2.rectangle(hud, (0, 0), (hud_width - 1, hud_height - 1), (255, 255, 255), 2)
    cv2.putText(hud, "RAW CAMERA", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y1 = 10
    x1 = table_frame.shape[1] - hud_width - 10
    table_frame[y1:y1 + hud_height, x1:x1 + hud_width] = hud
    return table_frame


def refresh_stats_panel(shoe, tracker, system=COUNT_SYSTEM, width=STATS_PANEL_WIDTH, height=TABLE_HEIGHT):
    dealer_ranks, player_ranks = tracker.hands()
    valid_dealer = [r for r in dealer_ranks if r in RANKS]
    valid_player = [r for r in player_ranks if r in RANKS]

    snapshot = build_snapshot(
        shoe,
        dealer_up=valid_dealer if valid_dealer else None,
        player_ranks=valid_player if valid_player else None,
        system=system,
    )
    return render_stats_panel(snapshot, height=height, width=width)


def parse_args():
    parser = argparse.ArgumentParser(description="Blackjack Card Counting Object Detection")
    parser.add_argument("--decks", type=int, default=SHOE_DECKS, help="Number of decks in shoe")
    parser.add_argument("--system", choices=["hi_lo", "ko", "omega_ii"], default=COUNT_SYSTEM)
    parser.add_argument("--legacy-gui", action="store_true", help="Also render legacy gui.py dashboard")
    parser.add_argument("--no-stats", action="store_true", help="Start with stats panel hidden")
    return parser.parse_args()


# -----------------------------
# Main program
# -----------------------------

def main():
    global latest_frame, running, zoom_level, calibrating, calibration_points
    global perspective_matrix, current_view_height, status_message, SHOW_STATS_PANEL

    args = parse_args()
    if args.no_stats:
        SHOW_STATS_PANEL = False

    model = YOLO(MODEL_PATH)

    shoe = Shoe(decks=args.decks)
    tracker = CardTracker(
        shoe=shoe
    )
    stats_img = refresh_stats_panel(shoe, tracker, system=args.system)
    last_hand_state = None

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    window_name = "Blackjack Card Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_w = DISPLAY_WIDTH + (STATS_PANEL_WIDTH if SHOW_STATS_PANEL else 0)
    cv2.resizeWindow(window_name, initial_w, DISPLAY_HEIGHT + BOTTOM_BAR_HEIGHT)
    cv2.setMouseCallback(window_name, mouse_callback)

    dealer_memory = []
    player_memory = []

    print("Camera started. Click CALIBRATE to select table corners.")
    print("Keys: Q quit | C calibrate | R new shoe | U undo card | P toggle stats panel")

    while running:
        success, original_frame = camera.read()
        if not success:
            print("Could not read webcam frame.")
            break

        raw_display = cv2.resize(original_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if calibrating or perspective_matrix is None:
            view = raw_display.copy()
            prompt = "Click: TL, TR, BR, BL" if calibrating else "Click CALIBRATE to select the table"
            cv2.putText(view, prompt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if calibrating:
                for index, point in enumerate(calibration_points):
                    cv2.circle(view, point, 8, (0, 0, 255), -1)
                    cv2.putText(view, str(index + 1), (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            current_view_height = view.shape[0]
            display_frame = draw_bottom_bar(view)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
            elif key == ord("c"):
                start_calibration()
            continue

        table_frame = cv2.warpPerspective(raw_display, perspective_matrix, (TABLE_WIDTH, TABLE_HEIGHT))

        if USE_LIGHTING_CORRECTION:
            table_frame = improve_lighting(table_frame)

        table_frame = apply_zoom(table_frame, zoom_level)
        latest_frame = table_frame.copy()

        model_frame = grayscale_for_model(table_frame)
        detections = collect_detections(model, model_frame, latest_frame)

        divider_y = table_frame.shape[0] // 2
        newly_counted = tracker.update(detections, divider_y)

        table_frame, dealer_ranks, player_ranks = draw_zones_and_boxes(table_frame, detections, tracker)

        hand_state = (tuple(dealer_ranks), tuple(player_ranks))
        if newly_counted or hand_state != last_hand_state:
            stats_img = refresh_stats_panel(
                shoe,
                tracker,
                system=args.system,
                height=table_frame.shape[0],
            )
            last_hand_state = hand_state

        if (USE_LEGACY_GUI or args.legacy_gui) and HAS_LEGACY_GUI:
            current_time = time.time()
            gui_dealer = update_hand_memory(dealer_ranks, dealer_memory, current_time)
            gui_player = update_hand_memory(player_ranks, player_memory, current_time)
            cv2.imshow("Blackjack Dashboard", render_dashboard(gui_dealer, gui_player))

        table_frame = add_raw_camera_hud(table_frame, raw_display)

        if status_message:
            cv2.putText(table_frame, status_message, (20, TABLE_HEIGHT - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(
            table_frame,
            f"Zoom: {zoom_level:.2f}x | Shoe #{tracker.shoe_number} | RC: {shoe.running_count(args.system):+d}",
            (20, TABLE_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        current_view_height = table_frame.shape[0]

        if SHOW_STATS_PANEL:
            panel_resized = cv2.resize(stats_img, (STATS_PANEL_WIDTH, table_frame.shape[0]))
            combined_view = np.hstack([table_frame, panel_resized])
        else:
            combined_view = table_frame

        display_frame = draw_bottom_bar(combined_view)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
        elif key == ord("c"):
            start_calibration()
        elif key == ord("r"):
            tracker.new_shoe()
            stats_img = refresh_stats_panel(shoe, tracker, system=args.system, height=table_frame.shape[0])
            status_message = f"Started new shoe #{tracker.shoe_number}"
        elif key == ord("u"):
            undone = shoe.undo()
            stats_img = refresh_stats_panel(shoe, tracker, system=args.system, height=table_frame.shape[0])
            status_message = f"Undid {undone}" if undone else "Nothing to undo"
        elif key == ord("p"):
            SHOW_STATS_PANEL = not SHOW_STATS_PANEL
            status_message = "Stats panel shown" if SHOW_STATS_PANEL else "Stats panel hidden"
        elif key in [ord("+"), ord("=")]:
            zoom_level = min(2.5, zoom_level + 0.25)
        elif key == ord("-"):
            zoom_level = max(1.0, zoom_level - 0.25)

    tracker.save_log()
    print(tracker.summary())
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()