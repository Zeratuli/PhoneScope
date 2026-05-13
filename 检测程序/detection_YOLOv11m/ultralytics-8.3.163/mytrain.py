import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    (ROOT / "yolo11n.pt", 40),
    (ROOT / "yolo11s.pt", 30),
    (ROOT / "yolo11m.pt", 16),
]


if __name__ == "__main__":
    for model_path, batch_size in MODEL_CANDIDATES:
        if not model_path.exists():
            print(f"[SKIP] 模型不存在: {model_path}")
            continue

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"[INFO] train {model_name} | batch={batch_size}")
        model = YOLO(str(model_path))
        model.train(
            data="iphone13.yaml",
            epochs=100,
            batch=batch_size,
            imgsz=640,
            workers=0,
            project="results",
            name=model_name,
            cache=False,
        )
