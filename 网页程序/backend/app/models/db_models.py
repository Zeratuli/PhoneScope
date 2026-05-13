from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime, Index, func,
)

from app.database import Base


class RecognitionSession(Base):
    """识别会话表 — 一次识别请求对应一行，前端数据管理页的操作主体"""
    __tablename__ = "recognition_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, unique=True, index=True)
    mode = Column(String(16), nullable=False)
    total_frames = Column(Integer, nullable=False, default=1)
    final_model_name = Column(String(128), nullable=True)
    final_confidence = Column(Float, nullable=True)
    best_frame_index = Column(Integer, nullable=True)
    total_processing_ms = Column(Float, nullable=False, default=0.0)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    deleted_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_sessions_created", "created_at"),
        Index("ix_sessions_deleted", "is_deleted"),
    )


class DetectionLog(Base):
    """识别日志明细表 — 每帧一行，通过 session_id 关联会话表"""
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False, default=0)
    filename = Column(String(255), nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    detection_count = Column(Integer, nullable=False, default=0)
    detection_bbox_json = Column(Text, nullable=True)
    detection_confidence = Column(Float, nullable=True)
    classification_model_name = Column(String(128), nullable=True)
    classification_confidence = Column(Float, nullable=True)
    classification_topk_json = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    is_best_frame = Column(Boolean, nullable=False, default=False)
    processing_time_ms = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_detection_logs_session", "session_id"),
        Index("ix_detection_logs_created", "created_at"),
    )


class PhoneModel(Base):
    """手机型号参数表 — 存储分类标签对应的手机规格信息"""
    __tablename__ = "phone_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_key = Column(String(64), nullable=False, unique=True, index=True)
    manufacturer = Column(String(64), nullable=True)
    brand = Column(String(64), nullable=True)
    model_name = Column(String(128), nullable=True)
    specs_json = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())
