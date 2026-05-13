from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
BASE_WEIGHTS = ROOT / "results" / "yolo11m_phone_2" / "weights" / "best.pt"


if __name__ == "__main__":
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError(f"找不到基础权重: {BASE_WEIGHTS}")

    model = YOLO(str(BASE_WEIGHTS))

    model.train(
        data="phone.yaml",
        epochs=80,
        imgsz=960,
        batch=-1,
        workers=1,
        cache=False,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.05,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.45,
        degrees=8.0,
        translate=0.08,
        scale=0.40,
        shear=1.5,
        perspective=0.000,
        fliplr=0.50,
        flipud=0.0,
        patience=30,
        lr0=0.002,
        optimizer="auto",
        amp=True,
        project="results",
        name="yolo11m_phone_ft960_2",
    )
