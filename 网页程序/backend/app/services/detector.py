import random
import numpy as np
from PIL import Image

from app.config import YOLO_CONFIDENCE
from app.schemas.models import DetectionItem, BoundingBox


class DetectorService:
    def __init__(self, model_path: str = ""):
        self._model = None
        self._mock = not model_path
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._mock = False
        except Exception:
            self._mock = True

    def is_loaded(self) -> bool:
        return self._model is not None or self._mock

    def detect(self, image: Image.Image, conf: float | None = None) -> list[DetectionItem]:
        conf = YOLO_CONFIDENCE if conf is None else conf
        if self._mock:
            return self._mock_detect(image)
        return self._real_detect(image, conf)

    def _real_detect(self, image: Image.Image, conf: float) -> list[DetectionItem]:
        img_array = np.array(image)
        results = self._model(img_array, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    DetectionItem(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(box.conf[0]),
                        label=r.names[int(box.cls[0])],
                        crop_base64=None,
                    )
                )
        return detections

    def _mock_detect(self, image: Image.Image) -> list[DetectionItem]:
        w, h = image.size
        num_detections = random.randint(1, 3)
        detections = []
        for _ in range(num_detections):
            cx = random.uniform(0.2, 0.8) * w
            cy = random.uniform(0.2, 0.8) * h
            bw = random.uniform(0.08, 0.25) * w
            bh = random.uniform(0.15, 0.4) * h
            detections.append(
                DetectionItem(
                    bbox=BoundingBox(
                        x1=max(0, cx - bw / 2),
                        y1=max(0, cy - bh / 2),
                        x2=min(w, cx + bw / 2),
                        y2=min(h, cy + bh / 2),
                    ),
                    confidence=round(random.uniform(0.75, 0.99), 4),
                    label="phone",
                    crop_base64=None,
                )
            )
        return detections
