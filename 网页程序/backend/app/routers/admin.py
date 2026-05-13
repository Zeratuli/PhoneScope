from fastapi import APIRouter, HTTPException, Query

from sqlalchemy import func as sa_func

from app.config import (
    ADMIN_SECRET_KEY, ADMIN_PAGE_PASSWORD, APP_NAME, APP_VERSION, DEBUG,
    DUALHEAD_ENABLED, DUALHEAD_MOBILENET_LATEST_RUN_DIR, DUALHEAD_SWIN_LATEST_RUN_DIR,
    USE_MOCK_SERVICES, MYSQL_URL,
    YOLO_MODEL_PATH, SWIN_MODEL_PATH, MOBILENET_MODEL_PATH,
)
from app.database import get_db
from app.middleware.security import get_service_status, set_service_status
from app.models.db_models import DetectionLog, PhoneModel, RecognitionSession
from app.schemas.models import AdminRequest, AdminResponse

router = APIRouter()


def _require_admin(key: str) -> None:
    """Admin 页面接口统一鉴权。"""
    if key != ADMIN_PAGE_PASSWORD:
        raise HTTPException(status_code=403, detail="密码错误")


@router.post("/admin/toggle", response_model=AdminResponse)
async def toggle_service(req: AdminRequest):
    if req.secret_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="密钥错误")

    if req.action == "enable":
        set_service_status(True)
    elif req.action == "disable":
        set_service_status(False)
    else:
        raise HTTPException(status_code=400, detail="action 必须为 enable 或 disable")

    return AdminResponse(
        success=True,
        service_enabled=get_service_status(),
        message=f"服务已{'启用' if get_service_status() else '禁用'}",
    )


@router.get("/admin/debug")
async def get_debug_info(key: str = Query(...)):
    _require_admin(key)
    from app.main import classifier_mobilenet, classifier_swin, detector

    table_counts = {}
    try:
        with get_db() as db:
            table_counts["recognition_sessions"] = db.query(
                sa_func.count(RecognitionSession.id)
            ).scalar() or 0
            table_counts["recognition_sessions_active"] = db.query(
                sa_func.count(RecognitionSession.id)
            ).filter(RecognitionSession.is_deleted == False).scalar() or 0  # noqa: E712
            table_counts["detection_logs"] = db.query(
                sa_func.count(DetectionLog.id)
            ).scalar() or 0
            table_counts["phone_models"] = db.query(
                sa_func.count(PhoneModel.id)
            ).scalar() or 0
    except Exception as e:
        table_counts["error"] = str(e)

    db_url_safe = MYSQL_URL.split("@")[-1] if "@" in MYSQL_URL else MYSQL_URL

    return {
        "health": {
            "status": "healthy" if get_service_status() else "disabled",
            "models_loaded": (
                detector.is_loaded()
                and classifier_mobilenet.is_loaded()
                and classifier_swin.is_loaded()
            ),
            "detector_loaded": detector.is_loaded(),
            "mobilenet_loaded": classifier_mobilenet.is_loaded(),
            "swin_loaded": classifier_swin.is_loaded(),
            "mobilenet_mock": getattr(classifier_mobilenet, "_mock", None),
            "swin_mock": getattr(classifier_swin, "_mock", None),
            "version": APP_VERSION,
            "service_enabled": get_service_status(),
        },
        "config": {
            "app_name": APP_NAME,
            "version": APP_VERSION,
            "debug": str(DEBUG),
            "use_mock": str(USE_MOCK_SERVICES),
            "dualhead_enabled": str(DUALHEAD_ENABLED),
            "db_host": db_url_safe,
            "yolo_model": YOLO_MODEL_PATH or "(mock)",
            "swin_model": SWIN_MODEL_PATH or "(mock)",
            "mobilenet_model": MOBILENET_MODEL_PATH or "(mock)",
            "mobilenet_run": DUALHEAD_MOBILENET_LATEST_RUN_DIR or "(none)",
            "swin_run": DUALHEAD_SWIN_LATEST_RUN_DIR or "(none)",
        },
        "tables": table_counts,
    }


@router.get("/admin/sessions")
async def admin_get_all_sessions(
    key: str = Query(...),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    search: str = Query(""),
    include_deleted: bool = Query(False),
):
    _require_admin(key)

    with get_db() as db:
        q = db.query(RecognitionSession)
        if not include_deleted:
            q = q.filter(RecognitionSession.is_deleted == False)  # noqa: E712
        if search:
            q = q.filter(
                (RecognitionSession.session_id.contains(search))
                | (RecognitionSession.final_model_name.contains(search))
                | (RecognitionSession.mode.contains(search))
            )
        total = q.count()
        items = (
            q.order_by(RecognitionSession.created_at.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
        return {
            "total": total,
            "page": page,
            "size": size,
            "items": [
                {
                    "id": s.id,
                    "session_id": s.session_id,
                    "mode": s.mode,
                    "total_frames": s.total_frames,
                    "final_model_name": s.final_model_name,
                    "final_confidence": s.final_confidence,
                    "best_frame_index": s.best_frame_index,
                    "total_processing_ms": s.total_processing_ms,
                    "is_deleted": s.is_deleted,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "deleted_at": s.deleted_at.isoformat() if s.deleted_at else None,
                }
                for s in items
            ],
        }


@router.get("/admin/logs")
async def admin_get_all_logs(
    key: str = Query(...),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    session_id: str = Query(""),
    search: str = Query(""),
):
    _require_admin(key)

    with get_db() as db:
        q = db.query(DetectionLog)
        if session_id:
            q = q.filter(DetectionLog.session_id == session_id)
        if search:
            q = q.filter(
                (DetectionLog.filename.contains(search))
                | (DetectionLog.classification_model_name.contains(search))
                | (DetectionLog.session_id.contains(search))
            )
        total = q.count()
        items = (
            q.order_by(DetectionLog.created_at.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
        return {
            "total": total,
            "page": page,
            "size": size,
            "items": [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "frame_index": r.frame_index,
                    "filename": r.filename,
                    "image_width": r.image_width,
                    "image_height": r.image_height,
                    "detection_count": r.detection_count,
                    "detection_confidence": r.detection_confidence,
                    "classification_model_name": r.classification_model_name,
                    "classification_confidence": r.classification_confidence,
                    "quality_score": r.quality_score,
                    "is_best_frame": r.is_best_frame,
                    "processing_time_ms": r.processing_time_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in items
            ],
        }


@router.get("/admin/phone-models")
async def admin_get_phone_models(key: str = Query(...)):
    _require_admin(key)

    with get_db() as db:
        items = db.query(PhoneModel).order_by(PhoneModel.model_key).all()
        return [
            {
                "id": p.id,
                "model_key": p.model_key,
                "manufacturer": p.manufacturer,
                "brand": p.brand,
                "model_name": p.model_name,
                "is_active": p.is_active,
                "specs_json": p.specs_json,
            }
            for p in items
        ]


# ----------------------------------------------------------------------
# 数据库写操作：硬删除 / 软删除恢复 / 永久清空 / 型号启停
# ----------------------------------------------------------------------


@router.delete("/admin/sessions/{session_id}")
async def admin_delete_session(session_id: str, key: str = Query(...)):
    """物理删除一条会话及其关联的所有日志明细。"""
    _require_admin(key)
    with get_db() as db:
        row = db.query(RecognitionSession).filter(
            RecognitionSession.session_id == session_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="会话不存在")
        deleted_logs = db.query(DetectionLog).filter(
            DetectionLog.session_id == session_id).delete(
            synchronize_session=False)
        db.delete(row)
    return {
        "success": True,
        "session_id": session_id,
        "deleted_logs": int(deleted_logs or 0),
    }


@router.post("/admin/sessions/{session_id}/restore")
async def admin_restore_session(session_id: str, key: str = Query(...)):
    """恢复一条被软删除的会话。"""
    _require_admin(key)
    with get_db() as db:
        row = db.query(RecognitionSession).filter(
            RecognitionSession.session_id == session_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="会话不存在")
        if not row.is_deleted:
            return {"success": True, "message": "该会话未被删除，无需恢复"}
        row.is_deleted = False
        row.deleted_at = None
    return {"success": True, "session_id": session_id}


@router.post("/admin/sessions/purge-deleted")
async def admin_purge_deleted_sessions(key: str = Query(...)):
    """永久删除所有 is_deleted=True 的会话及其日志。"""
    _require_admin(key)
    with get_db() as db:
        deleted_ids = [
            r.session_id
            for r in db.query(RecognitionSession.session_id).filter(
                RecognitionSession.is_deleted == True).all()  # noqa: E712
        ]
        if not deleted_ids:
            return {"success": True, "purged_sessions": 0, "purged_logs": 0}
        purged_logs = db.query(DetectionLog).filter(
            DetectionLog.session_id.in_(deleted_ids)).delete(
            synchronize_session=False)
        purged_sessions = db.query(RecognitionSession).filter(
            RecognitionSession.is_deleted == True).delete(  # noqa: E712
            synchronize_session=False)
    return {
        "success": True,
        "purged_sessions": int(purged_sessions or 0),
        "purged_logs": int(purged_logs or 0),
    }


@router.delete("/admin/logs/{log_id}")
async def admin_delete_log(log_id: int, key: str = Query(...)):
    """物理删除单条日志明细。"""
    _require_admin(key)
    with get_db() as db:
        row = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="日志不存在")
        db.delete(row)
    return {"success": True, "log_id": log_id}


@router.post("/admin/phone-models/{model_id}/toggle")
async def admin_toggle_phone_model(model_id: int, key: str = Query(...)):
    """切换 phone_models 某行 is_active。"""
    _require_admin(key)
    with get_db() as db:
        row = db.query(PhoneModel).filter(PhoneModel.id == model_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="型号不存在")
        row.is_active = not bool(row.is_active)
        new_state = row.is_active
    return {"success": True, "model_id": model_id, "is_active": new_state}
