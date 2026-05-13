import base64
import io
import re
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.schemas.models import ExportRequest
from app.services.pipeline import fusion_store, get_task

router = APIRouter()


def _get_exporter():
    from app.main import exporter
    return exporter


def _safe_filename(name: str) -> str:
    """去除文件名中的非法字符，保证可在 zip / Windows 下使用。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "image.jpg"


@router.post("/export/pdf")
async def export_pdf(req: ExportRequest):
    """根据 task_id 导出 PDF。支持单图/批量任务与融合会话两类。"""
    exporter_svc = _get_exporter()

    # 1. 单图 / 批量任务
    task = get_task(req.task_id)
    if task is not None:
        if task.status != "completed" or not task.results:
            raise HTTPException(400, "任务尚未完成或无结果")
        pdf_bytes = exporter_svc.generate_pdf(
            task.results, title="PhoneScope 检测报告")
        filename = f"report_{req.task_id}.pdf"
    else:
        # 2. 融合会话
        fusion_result = fusion_store.get(req.task_id)
        if fusion_result is None:
            raise HTTPException(404, "任务不存在或已过期")
        pdf_bytes = exporter_svc.generate_pdf_fusion(
            fusion_result, title="PhoneScope 融合识别报告")
        filename = f"fusion_report_{req.task_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _build_zip_for_results(results, task_id: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, result in enumerate(results, 1):
            if result.annotated_image_base64:
                try:
                    data = base64.b64decode(result.annotated_image_base64)
                    stem = _safe_filename(result.filename).rsplit(".", 1)[0]
                    zf.writestr(f"{idx:02d}_{stem}_annotated.jpg", data)
                except Exception:
                    pass
        for idx, result in enumerate(results, 1):
            stem = _safe_filename(result.filename).rsplit(".", 1)[0]
            for ci, det in enumerate(result.detections, 1):
                if not det.crop_base64:
                    continue
                try:
                    data = base64.b64decode(det.crop_base64)
                    zf.writestr(f"crops/{idx:02d}_{stem}_crop{ci}.jpg", data)
                except Exception:
                    pass
    return buf.getvalue()


def _build_zip_for_fusion(fusion_result) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for frame in fusion_result.frames:
            if not frame.annotated_image_base64:
                continue
            try:
                data = base64.b64decode(frame.annotated_image_base64)
            except Exception:
                continue
            stem = _safe_filename(frame.filename).rsplit(".", 1)[0]
            is_best = (frame.frame_index == fusion_result.best_frame_index)
            prefix = "best_" if is_best else ""
            zf.writestr(
                f"{prefix}frame_{frame.frame_index:02d}_{stem}.jpg", data)

        # 最优帧裁剪图
        if fusion_result.best_crop_base64:
            try:
                data = base64.b64decode(fusion_result.best_crop_base64)
                zf.writestr("best_crop.jpg", data)
            except Exception:
                pass
    return buf.getvalue()


@router.post("/export/images")
async def export_images(req: ExportRequest):
    """将任务中每张图的标注图打包为 ZIP 下载。支持单图/批量/融合。"""
    task = get_task(req.task_id)
    if task is not None:
        if task.status != "completed" or not task.results:
            raise HTTPException(400, "任务尚未完成或无结果")
        data = _build_zip_for_results(task.results, req.task_id)
        filename = f"images_{req.task_id}.zip"
    else:
        fusion_result = fusion_store.get(req.task_id)
        if fusion_result is None:
            raise HTTPException(404, "任务不存在或已过期")
        data = _build_zip_for_fusion(fusion_result)
        filename = f"fusion_images_{req.task_id}.zip"

    if not data or len(data) <= 22:   # 空 zip 约 22 字节
        raise HTTPException(400, "任务结果中没有可导出的图像")

    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
