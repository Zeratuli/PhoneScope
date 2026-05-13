from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Form
from typing import Optional

from app.schemas.models import AnalysisResponse, BatchTaskResponse
from app.services.pipeline import create_task, get_task, PipelineService
from app.utils.file_utils import is_image_file

router = APIRouter()


def _get_classifier(model_type: str = "swin"):
    from app.main import classifier_swin, classifier_mobilenet
    return classifier_mobilenet if model_type == "mobilenet" else classifier_swin


def _get_detector():
    from app.main import detector
    return detector


@router.post("/detect/image", response_model=AnalysisResponse)
async def detect_image(
    file: UploadFile = File(...),
    model_type: Optional[str] = Form("swin"),
):
    if not file.content_type or file.content_type not in (
        "image/jpeg", "image/png", "image/webp", "image/bmp"
    ):
        raise HTTPException(status_code=400, detail="仅支持 JPEG/PNG/WebP/BMP 图片")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小超过 50MB 限制")

    classifier = _get_classifier(model_type or "swin")
    pipeline = PipelineService(_get_detector(), classifier)
    result = pipeline.process_image(content, file.filename or "unknown.jpg")
    return AnalysisResponse(success=True, result=result)


@router.post("/detect/batch", response_model=BatchTaskResponse)
async def detect_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    model_type: Optional[str] = Form("swin"),
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="批量上传最多 20 个文件")

    files_data = []
    for f in files:
        if not is_image_file(f.content_type):
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {f.filename}")
        content = await f.read()
        files_data.append((content, f.filename or "unknown.jpg"))

    task_id = create_task()
    classifier = _get_classifier(model_type or "swin")
    pipeline = PipelineService(_get_detector(), classifier)
    background_tasks.add_task(pipeline.process_batch, files_data, task_id)

    return BatchTaskResponse(
        success=True,
        task_id=task_id,
        total_files=len(files_data),
        message="批量任务已提交",
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task
