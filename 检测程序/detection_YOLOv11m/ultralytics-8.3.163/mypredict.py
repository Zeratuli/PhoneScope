from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "results" / "yolo11m_phone_ft960_2" / "weights" / "best.pt"
SOURCE = ROOT / "datasets" / "Phone" / "images" / "val"


if __name__ == "__main__":
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"找不到检测权重: {WEIGHTS}")
    if not SOURCE.exists():
        raise FileNotFoundError(f"找不到预测输入: {SOURCE}")

    model = YOLO(str(WEIGHTS))
    model.predict(
        source=str(SOURCE),
        conf=0.7,
        save=False,
        show=True,
        save_txt=False,
    )
