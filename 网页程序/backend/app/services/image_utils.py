"""共用图像工具：裁剪 Base64、绘制检测框标注。

合并原 ``pipeline._crop_base64`` / ``pipeline._draw_annotations`` 与
``fusion._crop_base64`` / ``fusion._annotate_frame``，避免重复实现。
"""
from __future__ import annotations

import base64
import io
from typing import Callable, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

from app.schemas.models import BoundingBox, DetectionItem

_TRAFFIC_LIGHT = [
    (0.85, (16, 185, 129)),   # 绿
    (0.70, (245, 158, 11)),   # 琥珀
    (0.00, (239, 68, 68)),    # 红
]


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def traffic_light_color(confidence: float) -> tuple[int, int, int]:
    """按置信度返回三色 RGB，与前端 ImageCanvas 保持一致。"""
    for threshold, color in _TRAFFIC_LIGHT:
        if confidence > threshold:
            return color
    return _TRAFFIC_LIGHT[-1][1]


def crop_base64(image: Image.Image, bbox: BoundingBox, quality: int = 85) -> str:
    """裁剪检测框区域并编码为 JPEG Base64。"""
    crop = image.crop((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_annotations(
    image: Image.Image,
    detections: Iterable[DetectionItem],
    labels: Optional[list[str]] = None,
    highlight_indices: Optional[set[int]] = None,
    color_strategy: str = "traffic_light",
    quality: int = 85,
) -> str:
    """通用检测框绘制函数。

    Args:
        image: 源图像（调用方应传入 copy 以避免修改原图）。
        detections: 待绘制的检测框列表。
        labels: 可选，每个框对应的文字标签；为 None 时使用置信度百分比。
        highlight_indices: 可选，指定哪些框以强调方式绘制（加粗边框）。
        color_strategy: "traffic_light" 按置信度三色；"highlight" 按是否在
            highlight_indices 中使用强调色或灰色。
        quality: 输出 JPEG 质量。

    Returns:
        Base64 编码的 JPEG 字节字符串。
    """
    draw = ImageDraw.Draw(image)
    font = _load_font(16)
    highlight_indices = highlight_indices or set()

    for i, det in enumerate(detections):
        b = det.bbox
        is_highlight = i in highlight_indices

        if color_strategy == "highlight":
            color = (16, 185, 129) if is_highlight else (100, 116, 139)
            width = 4 if is_highlight else 2
        else:
            color = traffic_light_color(det.confidence)
            width = 3

        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)

        label = labels[i] if labels and i < len(labels) else f"{det.confidence:.0%}"
        text_bbox = draw.textbbox((b.x1, b.y1 - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((b.x1, b.y1 - 22), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_DrawCallable = Callable[..., str]
