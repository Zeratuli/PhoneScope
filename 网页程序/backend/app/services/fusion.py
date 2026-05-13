import io
import time

from PIL import Image

from app.config import (
    FUSION_EDGE_SHRINK_RATIO,
    FUSION_MIN_CONFIDENCE,
    FUSION_MIN_AREA_RATIO,
)
from app.schemas.models import (
    BoundingBox,
    DetectionItem,
    FrameEvidence,
    FusionResult,
    TopKItem,
)
from app.services.classifier import ClassifierService
from app.services.detector import DetectorService
from app.services.image_utils import crop_base64, draw_annotations


def compute_quality_score(det: DetectionItem, image_w: int, image_h: int) -> float:
    b = det.bbox
    box_w, box_h = b.x2 - b.x1, b.y2 - b.y1
    if box_w <= 0 or box_h <= 0:
        return 0.0
    area_ratio = (box_w * box_h) / (image_w * image_h)
    if area_ratio < FUSION_MIN_AREA_RATIO:
        return 0.0
    if det.confidence < FUSION_MIN_CONFIDENCE:
        return 0.0
    aspect = box_w / box_h
    if aspect > 5.0 or aspect < 0.2:
        return 0.0
    return area_ratio * det.confidence


def shrink_crop(image: Image.Image, bbox: BoundingBox, ratio: float) -> Image.Image:
    bw, bh = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
    dx, dy = bw * ratio, bh * ratio
    x1 = max(0, bbox.x1 + dx)
    y1 = max(0, bbox.y1 + dy)
    x2 = min(image.width, bbox.x2 - dx)
    y2 = min(image.height, bbox.y2 - dy)
    if x2 <= x1 or y2 <= y1:
        return image.crop((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)))
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


class FusionService:
    def __init__(self, detector: DetectorService, classifier: ClassifierService):
        self.detector = detector
        self.classifier = classifier

    def process_fusion(
        self,
        evidences: list[tuple[bytes, str]],
        session_id: str,
        mode: str = "images",
    ) -> FusionResult:
        start = time.time()
        frames: list[FrameEvidence] = []
        candidates: list[tuple[int, int, float, DetectionItem, Image.Image]] = []

        for fi, (img_bytes, filename) in enumerate(evidences):
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = image.size
            dets = self.detector.detect(image)

            best_det_idx: int | None = None
            best_score = 0.0
            for di, det in enumerate(dets):
                score = compute_quality_score(det, w, h)
                if score > best_score:
                    best_score, best_det_idx = score, di

            is_valid = best_det_idx is not None and best_score > 0
            if is_valid:
                candidates.append(
                    (fi, best_det_idx, best_score, dets[best_det_idx], image))

            annotated_b64 = draw_annotations(
                image.copy(),
                dets,
                labels=[f"{d.confidence:.0%}" for d in dets],
                highlight_indices={best_det_idx} if best_det_idx is not None else set(),
                color_strategy="highlight",
            )

            frames.append(FrameEvidence(
                frame_index=fi,
                filename=filename,
                width=w, height=h,
                detections=dets,
                best_detection_index=best_det_idx,
                quality_score=round(best_score, 6),
                is_valid=is_valid,
                annotated_image_base64=annotated_b64,
            ))

        final_model: str | None = None
        final_brand: str | None = None
        final_series: str | None = None
        final_display: str | None = None
        final_conf: float | None = None
        final_topk: list[TopKItem] | None = None
        final_spec = None
        best_frame_idx: int | None = None
        best_crop_b64: str | None = None

        if candidates:
            candidates.sort(key=lambda c: c[2], reverse=True)
            best_fi, _, _, best_det, best_img = candidates[0]
            best_frame_idx = best_fi
            frames[best_fi].is_best = True

            cls_result = self.classifier.classify(
                shrink_crop(best_img, best_det.bbox, FUSION_EDGE_SHRINK_RATIO))
            final_model = cls_result.model_name
            final_brand = cls_result.brand_name
            final_series = cls_result.series_name
            final_display = cls_result.display_name
            final_conf = cls_result.confidence
            final_topk = cls_result.top_k
            final_spec = cls_result.phone_spec
            best_crop_b64 = crop_base64(best_img, best_det.bbox)

            if len(candidates) >= 2:
                _, _, _, sec_det, sec_img = candidates[1]
                sec_cls = self.classifier.classify(
                    shrink_crop(sec_img, sec_det.bbox, FUSION_EDGE_SHRINK_RATIO))
                if sec_cls.model_name == final_model:
                    final_conf = max(final_conf, sec_cls.confidence)

        elapsed = (time.time() - start) * 1000

        return FusionResult(
            session_id=session_id, mode=mode,
            total_frames=len(evidences), valid_frames=len(candidates),
            best_frame_index=best_frame_idx, frames=frames,
            final_model_name=final_model,
            final_brand_name=final_brand,
            final_series_name=final_series,
            final_display_name=final_display,
            final_confidence=round(final_conf, 4) if final_conf is not None else None,
            final_top_k=final_topk, final_phone_spec=final_spec,
            best_crop_base64=best_crop_b64,
            processing_time_ms=round(elapsed, 2),
        )
