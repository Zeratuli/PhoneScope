from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from sqlalchemy import func as sa_func

from app.database import get_db
from app.models.db_models import DetectionLog, RecognitionSession

router = APIRouter()


# ---------- Response schemas ----------

class SessionItem(BaseModel):
    id: int
    session_id: str
    mode: str
    total_frames: int
    final_model_name: Optional[str]
    final_confidence: Optional[float]
    best_frame_index: Optional[int]
    total_processing_ms: float
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class LogItem(BaseModel):
    id: int
    session_id: str
    frame_index: int
    filename: str
    image_width: int
    image_height: int
    detection_count: int
    detection_confidence: Optional[float]
    classification_model_name: Optional[str]
    classification_confidence: Optional[float]
    quality_score: Optional[float]
    is_best_frame: bool
    processing_time_ms: float
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class SessionsResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[SessionItem]


class LogsResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[LogItem]


class StatsResponse(BaseModel):
    total_detections: int
    total_sessions: int
    avg_detection_count: float
    top_models: list[dict]
    mode_counts: dict


# ---------- Session-level endpoints ----------

@router.get("/sessions", response_model=SessionsResponse)
def get_sessions(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    try:
        with get_db() as db:
            q = db.query(RecognitionSession).filter(
                RecognitionSession.is_deleted == False  # noqa: E712
            )
            if mode:
                q = q.filter(RecognitionSession.mode == mode)
            if model_name:
                q = q.filter(RecognitionSession.final_model_name.contains(model_name))
            if date_from:
                try:
                    q = q.filter(RecognitionSession.created_at >= datetime.fromisoformat(date_from))
                except ValueError:
                    pass
            if date_to:
                try:
                    q = q.filter(RecognitionSession.created_at <= datetime.fromisoformat(date_to))
                except ValueError:
                    pass
            total = q.count()
            items = (
                q.order_by(RecognitionSession.created_at.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            return SessionsResponse(
                total=total, page=page, size=size,
                items=[SessionItem.model_validate(item) for item in items],
            )
    except Exception:
        return SessionsResponse(total=0, page=page, size=size, items=[])


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Soft-delete a session and its logs are hidden from queries."""
    with get_db() as db:
        row = db.query(RecognitionSession).filter(
            RecognitionSession.session_id == session_id
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="会话不存在")
        row.is_deleted = True
        row.deleted_at = datetime.utcnow()
    return {"success": True}


# ---------- Legacy log-level endpoints (kept for backward compatibility) ----------

@router.get("/logs", response_model=LogsResponse)
def get_logs(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
):
    try:
        with get_db() as db:
            active_sessions = (
                db.query(RecognitionSession.session_id)
                .filter(RecognitionSession.is_deleted == False)  # noqa: E712
                .subquery()
            )
            q = db.query(DetectionLog).filter(
                DetectionLog.session_id.in_(active_sessions)
            )
            if session_id:
                q = q.filter(DetectionLog.session_id == session_id)
            if mode:
                q = q.join(
                    RecognitionSession,
                    DetectionLog.session_id == RecognitionSession.session_id,
                ).filter(RecognitionSession.mode == mode)
            if model_name:
                q = q.filter(DetectionLog.classification_model_name.contains(model_name))
            if date_from:
                try:
                    q = q.filter(DetectionLog.created_at >= datetime.fromisoformat(date_from))
                except ValueError:
                    pass
            if date_to:
                try:
                    q = q.filter(DetectionLog.created_at <= datetime.fromisoformat(date_to))
                except ValueError:
                    pass
            total = q.count()
            items = (
                q.order_by(DetectionLog.created_at.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            return LogsResponse(
                total=total, page=page, size=size,
                items=[LogItem.model_validate(item) for item in items],
            )
    except Exception:
        return LogsResponse(total=0, page=page, size=size, items=[])


@router.get("/logs/stats", response_model=StatsResponse)
def get_stats():
    try:
        with get_db() as db:
            active_sessions = (
                db.query(RecognitionSession.session_id)
                .filter(RecognitionSession.is_deleted == False)  # noqa: E712
                .subquery()
            )
            base_q = db.query(DetectionLog).filter(
                DetectionLog.session_id.in_(active_sessions)
            )
            total = base_q.count()
            sessions = (
                db.query(RecognitionSession)
                .filter(RecognitionSession.is_deleted == False)  # noqa: E712
                .count()
            )
            avg_det = base_q.with_entities(
                sa_func.avg(DetectionLog.detection_count)
            ).scalar() or 0
            model_counts = (
                base_q.with_entities(
                    DetectionLog.classification_model_name,
                    sa_func.count(DetectionLog.id).label("cnt"),
                )
                .filter(DetectionLog.classification_model_name.isnot(None))
                .group_by(DetectionLog.classification_model_name)
                .order_by(sa_func.count(DetectionLog.id).desc())
                .limit(10)
                .all()
            )
            mode_rows = (
                db.query(
                    RecognitionSession.mode,
                    sa_func.count(RecognitionSession.id).label("cnt"),
                )
                .filter(RecognitionSession.is_deleted == False)  # noqa: E712
                .group_by(RecognitionSession.mode)
                .all()
            )
            return StatsResponse(
                total_detections=total,
                total_sessions=sessions,
                avg_detection_count=round(float(avg_det), 2),
                top_models=[{"name": r[0], "count": r[1]} for r in model_counts],
                mode_counts={r[0]: r[1] for r in mode_rows},
            )
    except Exception:
        return StatsResponse(
            total_detections=0, total_sessions=0, avg_detection_count=0,
            top_models=[], mode_counts={},
        )


@router.delete("/logs/{log_id}")
def delete_log(log_id: int):
    with get_db() as db:
        item = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="记录不存在")
        db.delete(item)
    return {"success": True}


@router.delete("/logs")
def delete_logs_by_session(session_id: str = Query(...)):
    """Soft-delete the session; logs are hidden but preserved."""
    with get_db() as db:
        row = db.query(RecognitionSession).filter(
            RecognitionSession.session_id == session_id
        ).first()
        if row:
            row.is_deleted = True
            row.deleted_at = datetime.utcnow()
        else:
            db.query(DetectionLog).filter(
                DetectionLog.session_id == session_id
            ).delete()
    return {"success": True}
