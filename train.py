from ultralytics import YOLO


def main():
    model = YOLO(
        "runs/detect/blackjack_cards/weights/best.pt"
    )

    model.train(
        data="Cards.yolov11/data.yaml",
        epochs=5,
        imgsz=416,
        batch=8,
        workers=2,
        cache="disk",
        name="blackjack_cards_fast"
    )


if __name__ == "__main__":
    main()