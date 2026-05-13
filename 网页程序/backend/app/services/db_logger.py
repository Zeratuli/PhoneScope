"""统一的会话 + 日志入库逻辑。

合并原 ``pipeline._log_to_db`` 与 ``fusion._save_to_db``，
消除两处对 ``RecognitionSession`` / ``DetectionLog`` 的重复构造代码。
"""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from typing import Optional

from app.database import get_db
from app.models.db_models import DetectionLog, RecognitionSession
from app.schemas.models import DetectionItem, TopKItem

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """入库单帧记录所需的最小信息。"""
    frame_index: int
    filename: str
    width: int
    height: int
    detections: list[DetectionItem]
    quality_score: Optional[float] = None
    is_best_frame: bool = False
    processing_time_ms: float = 0.0
    classification_model_name: Optional[str] = None
    classification_confidence: Optional[float] = None
    classification_topk: Optional[list[TopKItem]] = None


def _bbox_list(detections: list[DetectionItem]) -> list[dict]:
    return [
        {"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2,
         "conf": d.confidence}
        for d in detections
    ]


def _best_detection_conf(detections: list[DetectionItem]) -> Optional[float]:
    if not detections:
        return None
    return max(d.confidence for d in detections)


def _topk_json(topk: Optional[list[TopKItem]]) -> Optional[str]:
    if not topk:
        return None
    return json.dumps(
        [{"name": t.name, "confidence": t.confidence} for t in topk],
        ensure_ascii=False,
    )


def persist_session_with_logs(
    session_id: str,
    mode: str,
    frames: list[FrameRecord],
    total_processing_ms: float,
    final_model_name: Optional[str] = None,
    final_confidence: Optional[float] = None,
    best_frame_index: Optional[int] = None,
) -> bool:
    """写入一条 RecognitionSession + 多条 DetectionLog。

    Returns:
        True 表示写入成功，False 表示失败（异常已记录到日志，调用方不必自行处理）。
    """
    total_frames = max(1, len(frames))
    try:
        with get_db() as db:
            session_row = RecognitionSession(
                session_id=session_id,
                mode=mode,
                total_frames=total_frames,
                final_model_name=final_model_name,
                final_confidence=final_confidence,
                best_frame_index=best_frame_index,
                total_processing_ms=total_processing_ms,
            )
            db.add(session_row)
            db.flush()

            for f in frames:
                log = DetectionLog(
                    session_id=session_id,
                    frame_index=f.frame_index,
                    filename=f.filename,
                    image_width=f.width,
                    image_height=f.height,
                    detection_count=len(f.detections),
                    detection_bbox_json=json.dumps(
                        _bbox_list(f.detections), ensure_ascii=False),
                    detection_confidence=_best_detection_conf(f.detections),
                    classification_model_name=f.classification_model_name,
                    classification_confidence=f.classification_confidence,
                    classification_topk_json=_topk_json(f.classification_topk),
                    quality_score=f.quality_score,
                    is_best_frame=f.is_best_frame,
                    processing_time_ms=f.processing_time_ms,
                )
                db.add(log)
        return True
    except Exception as e:
        logger.error(
            "[DB] persist_session_with_logs failed for session=%s mode=%s: %s\n%s",
            session_id, mode, e, traceback.format_exc(),
        )
        return False
