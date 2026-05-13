import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException

from app.config import (
    ALLOWED_TYPES,
    ALLOWED_IMAGE_TYPES,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR,
)

MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",
    b"BM": "image/bmp",
    b"\x00\x00\x00": "video/mp4",
    b"\x1a\x45\xdf\xa3": "video/webm",
}


def validate_file_type(content_type: str | None, header_bytes: bytes) -> str:
    if content_type and content_type in ALLOWED_TYPES:
        return content_type
    for sig, mime in MAGIC_SIGNATURES.items():
        if header_bytes[: len(sig)] == sig:
            return mime
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {content_type}")


def validate_file_size(size: int) -> None:
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {size / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_MB}MB",
        )


async def save_upload_file(file: UploadFile, sub_dir: str = "") -> Path:
    content = await file.read()
    validate_file_size(len(content))
    validate_file_type(file.content_type, content[:16])

    ext = Path(file.filename or "file").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_dir = UPLOAD_DIR / sub_dir if sub_dir else UPLOAD_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    save_path.write_bytes(content)
    await file.seek(0)
    return save_path


def is_image_file(content_type: str | None) -> bool:
    return content_type in ALLOWED_IMAGE_TYPES if content_type else False


def cleanup_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
