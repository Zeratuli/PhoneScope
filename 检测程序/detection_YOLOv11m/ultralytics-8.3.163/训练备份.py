from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
        data = r"coco8.yaml",
        epochs = 30,
        batch = 16,
        imgsz = 640,
        workers = 0,
        cache = False,
        # amp = False,
    )