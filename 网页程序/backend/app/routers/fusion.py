import logging
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.config import DEFAULT_FRAME_INTERVAL
from app.schemas.models import FusionResponse, FusionResult
from app.services.db_logger import FrameRecord, persist_session_with_logs
from app.services.fusion import FusionService
from app.services.pipeline import fusion_store
from app.services.video_extractor import extract_frames

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_fusion_service(model_type: str = "swin") -> FusionService:
    from app.main import detector, classifier_swin, classifier_mobilenet
    classifier = classifier_mobilenet if model_type == "mobilenet" else classifier_swin
    return FusionService(detector, classifier)


@router.post("/detect/fusion", response_model=FusionResponse)
async def detect_fusion(
    files: Optional[list[UploadFile]] = File(None),
    video: Optional[UploadFile] = File(None),
    frame_interval: int = Form(default=DEFAULT_FRAME_INTERVAL),
    model_type: Optional[str] = Form("swin"),
):
    session_id = uuid.uuid4().hex[:16]
    svc = _get_fusion_service(model_type or "swin")

    if video and video.filename:
        ct = video.content_type or ""
        if ct not in ("video/mp4", "video/webm", "video/avi", "video/quicktime"):
            raise HTTPException(400, f"不支持的视频格式: {ct}")
        video_bytes = await video.read()
        if len(video_bytes) > 200 * 1024 * 1024:
            raise HTTPException(400, "视频大小超过 200MB 限制")
        interval = max(1, min(frame_interval, 300))
        evidences = extract_frames(video_bytes, interval)
        if not evidences:
            raise HTTPException(400, "无法从视频中提取帧")
        mode = "video"

    elif files:
        if len(files) > 5:
            raise HTTPException(400, "融合模式最多上传 5 张图片")
        evidences = []
        for f in files:
            ct = f.content_type or ""
            if not ct.startswith("image/"):
                raise HTTPException(400, f"不支持的文件类型: {f.filename}")
            content = await f.read()
            evidences.append((content, f.filename or "unknown.jpg"))
        mode = "images"
    else:
        raise HTTPException(400, "请上传图片或视频")

    result = svc.process_fusion(evidences, session_id, mode=mode)
    fusion_store[session_id] = result   # 缓存供后续 PDF/图片导出
    _persist_fusion_result(result)
    return FusionResponse(success=True, result=result)


def _persist_fusion_result(result: FusionResult) -> None:
    """将融合识别的会话与每帧明细写入数据库。"""
    per_frame_ms = result.processing_time_ms / max(result.total_frames, 1)
    frames: list[FrameRecord] = []
    for frame in result.frames:
        is_best = (frame.frame_index == result.best_frame_index)
        frames.append(FrameRecord(
            frame_index=frame.frame_index,
            filename=frame.filename,
            width=frame.width, height=frame.height,
            detections=frame.detections,
            quality_score=frame.quality_score,
            is_best_frame=is_best,
            processing_time_ms=per_frame_ms,
            classification_model_name=result.final_model_name if is_best else None,
            classification_confidence=result.final_confidence if is_best else None,
            classification_topk=result.final_top_k if is_best else None,
        ))
    persist_session_with_logs(
        session_id=result.session_id,
        mode="fusion",
        frames=frames,
        total_processing_ms=result.processing_time_ms,
        final_model_name=result.final_model_name,
        final_confidence=result.final_confidence,
        best_frame_index=result.best_frame_index,
    )
