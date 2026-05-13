import io
import logging
import time
import uuid
from datetime import datetime, timezone

from PIL import Image

from app.schemas.models import FusionResult, ImageResult, TaskStatusResponse
from app.services.detector import DetectorService
from app.services.classifier import ClassifierService
from app.services.image_utils import crop_base64, draw_annotations
from app.services.db_logger import FrameRecord, persist_session_with_logs

logger = logging.getLogger(__name__)

task_store: dict[str, TaskStatusResponse] = {}
# 融合识别结果缓存（key: session_id）—— 供 /export/* 接口二次读取
fusion_store: dict[str, FusionResult] = {}


class PipelineService:
    def __init__(self, detector: DetectorService, classifier: ClassifierService):
        self.detector = detector
        self.classifier = classifier

    def process_image(self, image_bytes: bytes, filename: str) -> ImageResult:
        start = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        detections = self.detector.detect(image)

        classifications = []
        if detections:
            for det in detections:
                crop = image.crop((
                    int(det.bbox.x1), int(det.bbox.y1),
                    int(det.bbox.x2), int(det.bbox.y2),
                ))
                classifications.append(self.classifier.classify(crop))
                det.crop_base64 = crop_base64(image, det.bbox)
        else:
            # 若 YOLO 未检出手机，退化为整图分类，避免直接无结果
            classifications.append(self.classifier.classify(image.copy()))

        result = ImageResult(
            image_id=uuid.uuid4().hex[:12],
            filename=filename,
            width=w, height=h,
            detections=detections, classifications=classifications,
            annotated_image_base64="", processing_time_ms=0,
        )

        labels = [
            f"{cls.display_name or cls.model_name} {det.confidence:.0%}"
            for det, cls in zip(detections, classifications)
        ]
        result.annotated_image_base64 = draw_annotations(
            image.copy(), detections, labels=labels,
            color_strategy="traffic_light",
        )
        result.processing_time_ms = round((time.time() - start) * 1000, 2)

        task_store[result.image_id] = TaskStatusResponse(
            task_id=result.image_id, status="completed", progress=1.0,
            current_file=None, results=[result],
            created_at=datetime.now(timezone.utc), error=None,
        )

        _persist_single_image_result(result, mode="single")
        return result

    def process_batch(self, files_data: list[tuple[bytes, str]], task_id: str) -> None:
        total = len(files_data)
        task_store[task_id] = TaskStatusResponse(
            task_id=task_id, status="processing", progress=0.0,
            current_file=None, results=[],
            created_at=datetime.now(timezone.utc), error=None,
        )
        results = []
        try:
            for i, (content, filename) in enumerate(files_data):
                task_store[task_id].current_file = filename
                task_store[task_id].progress = i / total
                results.append(self.process_image(content, filename))

            task_store[task_id].status = "completed"
            task_store[task_id].progress = 1.0
            task_store[task_id].results = results
            task_store[task_id].current_file = None
        except Exception as e:
            logger.exception("process_batch failed: %s", e)
            task_store[task_id].status = "failed"
            task_store[task_id].error = str(e)


def create_task() -> str:
    task_id = uuid.uuid4().hex[:16]
    task_store[task_id] = TaskStatusResponse(
        task_id=task_id, status="pending", progress=0.0,
        results=None, created_at=datetime.now(timezone.utc),
    )
    return task_id


def store_single_result(result: ImageResult) -> None:
    task_store[result.image_id] = TaskStatusResponse(
        task_id=result.image_id, status="completed", progress=1.0,
        current_file=None, results=[result],
        created_at=datetime.now(timezone.utc), error=None,
    )


def get_task(task_id: str) -> TaskStatusResponse | None:
    return task_store.get(task_id)


def _persist_single_image_result(result: ImageResult, mode: str = "single") -> None:
    """将单图识别结果写入 MySQL（通过统一入库函数）。"""
    best_cls = result.classifications[0] if result.classifications else None
    frame = FrameRecord(
        frame_index=0,
        filename=result.filename,
        width=result.width, height=result.height,
        detections=result.detections,
        quality_score=None,
        is_best_frame=True,
        processing_time_ms=result.processing_time_ms,
        classification_model_name=best_cls.model_name if best_cls else None,
        classification_confidence=best_cls.confidence if best_cls else None,
        classification_topk=best_cls.top_k if best_cls else None,
    )
    persist_session_with_logs(
        session_id=result.image_id,
        mode=mode,
        frames=[frame],
        total_processing_ms=result.processing_time_ms,
        final_model_name=best_cls.model_name if best_cls else None,
        final_confidence=best_cls.confidence if best_cls else None,
        best_frame_index=0,
    )
