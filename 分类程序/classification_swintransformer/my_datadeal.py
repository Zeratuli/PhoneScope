
"""
my_datadeal.py

功能：
- 从指定文件夹读取图片/视频
- 使用 phone_detect.pt 检测手机（类别 phone）
- 取每张图/每帧最高置信度的手机框并裁剪
- 裁剪结果保存到 ./orimage
- 输出 ./orimage/crops.csv 记录检测信息
- 带进度条

运行目录要求：当前文件所在目录（classification_swintransformer）
"""

from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# 可修改配置区（按你第5条）
# =========================

ROOT = Path(__file__).resolve().parent

# 输入来源目录：把你要处理的图片/视频丢到这里（可改）
INPUT_DIR = str(ROOT / "oriData")

# 输出裁剪目录：固定为 \orimage（按你第4条）
OUT_DIR = str(ROOT / "orimage" / "REDMI_K80_Pro")

# YOLO 检测模型路径（按你第3条）
MODEL_PATH = str(ROOT / "phone_detect.pt")

# 文件命名：是否加“手机型号”前缀（按你第6条）
# True: HUAWEI_NOVA_10_00001.jpg
# False: 00001.jpg
USE_MODEL_PREFIX = True

# 你想写入前缀的“手机型号”（可选，不写就用空字符串）
# 注意：如果 USE_MODEL_PREFIX=True 且这里为空，会自动只用编号（不加前缀）
PHONE_MODEL_PREFIX = "REDMI_K80_Pro"   # 例如："iPhone_13"；不想要就设为 ""

# 图片：每张图只裁一个（最高置信度框）
# 视频：每隔 N 帧取一帧做检测（提高速度），比如 5 表示每5帧处理一次
VIDEO_FRAME_STRIDE = 5

# 检测阈值
CONF_THRES = 0.4
IOU_THRES = 0.5
IMGSZ = 960

# 裁剪时给 bbox 四周留边比例（避免裁太紧）
PAD_RATIO = 0.02

# 只保留指定类别（phone）。如果你的检测模型只有一个类别，通常 id=0。
# 不确定就先用 None（表示不限制类别）
TARGET_CLS_ID: Optional[int] = 0

# 输出 CSV 路径（按你第7条）
CSV_PATH = str(ROOT / "orimage" / "crops.csv")


# =========================
# 代码实现
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


@dataclass
class CropRecord:
    src_type: str          # image / video
    src_path: str          # 原文件路径
    frame_index: int       # 图片=-1；视频为帧号
    out_path: str          # 输出裁剪文件路径
    conf: float            # 检测置信度
    cls_id: int            # 类别id
    x1: int
    y1: int
    x2: int
    y2: int
    w: int                 # 原图宽
    h: int                 # 原图高


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_media_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in IMG_EXTS or suf in VID_EXTS:
            files.append(p)
    return sorted(files)


def safe_crop(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def pick_best_box(result, target_cls: Optional[int]) -> Optional[Tuple[float, float, float, float, float, int]]:
    """
    返回 best box: (x1,y1,x2,y2, conf, cls_id)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    best_i = None
    best_score = -1.0
    for i in range(len(xyxy)):
        if target_cls is not None and cls[i] != target_cls:
            continue
        score = float(conf[i])
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None

    x1, y1, x2, y2 = xyxy[best_i]
    return float(x1), float(y1), float(x2), float(y2), float(conf[best_i]), int(cls[best_i])


def make_out_name(index: int) -> str:
    """
    按你第6条：手机型号（可选）+编号 或 仅编号
    """
    num = f"{index:05d}"
    if USE_MODEL_PREFIX and PHONE_MODEL_PREFIX.strip():
        return f"{PHONE_MODEL_PREFIX}_{num}.jpg"
    return f"{num}.jpg"


def write_csv(records: List[CropRecord], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    header = [
        "src_type", "src_path", "frame_index",
        "out_path", "conf", "cls_id",
        "x1", "y1", "x2", "y2",
        "img_w", "img_h"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            w.writerow([
                r.src_type, r.src_path, r.frame_index,
                r.out_path, f"{r.conf:.6f}", r.cls_id,
                r.x1, r.y1, r.x2, r.y2,
                r.w, r.h
            ])


def process_image(model: YOLO, img_path: Path, out_dir: Path, start_index: int,
                  records: List[CropRecord]) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        return start_index

    h, w = img.shape[:2]
    results = model.predict(
        source=img,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )
    r = results[0]
    best = pick_best_box(r, TARGET_CLS_ID)
    if best is None:
        return start_index

    x1, y1, x2, y2, conf, cls_id = best

    # padding
    bw = x2 - x1
    bh = y2 - y1
    px = bw * PAD_RATIO
    py = bh * PAD_RATIO
    crop = safe_crop(img, x1 - px, y1 - py, x2 + px, y2 + py)
    if crop is None:
        return start_index

    out_name = make_out_name(start_index)
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), crop)

    records.append(CropRecord(
        src_type="image",
        src_path=str(img_path),
        frame_index=-1,
        out_path=str(out_path),
        conf=float(conf),
        cls_id=int(cls_id),
        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
        w=int(w), h=int(h),
    ))

    return start_index + 1


def video_frame_count(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        return n
    # 有些编码拿不到帧数，返回 -1
    return -1


def process_video(model: YOLO, vid_path: Path, out_dir: Path, start_index: int,
                  records: List[CropRecord]) -> int:
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return start_index

    total_frames = video_frame_count(cap)
    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                desc=f"Video {vid_path.name}", ncols=100)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % VIDEO_FRAME_STRIDE != 0:
            frame_idx += 1
            if total_frames > 0:
                pbar.update(1)
            continue

        h, w = frame.shape[:2]
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )
        r = results[0]
        best = pick_best_box(r, TARGET_CLS_ID)
        if best is not None:
            x1, y1, x2, y2, conf, cls_id = best
            bw = x2 - x1
            bh = y2 - y1
            px = bw * PAD_RATIO
            py = bh * PAD_RATIO
            crop = safe_crop(frame, x1 - px, y1 - py, x2 + px, y2 + py)
            if crop is not None:
                out_name = make_out_name(start_index)
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), crop)

                records.append(CropRecord(
                    src_type="video",
                    src_path=str(vid_path),
                    frame_index=int(frame_idx),
                    out_path=str(out_path),
                    conf=float(conf),
                    cls_id=int(cls_id),
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    w=int(w), h=int(h),
                ))
                start_index += 1

        frame_idx += 1
        if total_frames > 0:
            pbar.update(1)

    cap.release()
    pbar.close()
    return start_index


def main():
    input_dir = Path(INPUT_DIR)
    out_dir = Path(OUT_DIR)
    csv_path = Path(CSV_PATH)

    ensure_dir(out_dir)
    ensure_dir(input_dir)

    media = list_media_files(input_dir)
    if not media:
        print(f"[WARN] 输入目录为空：{input_dir.resolve()}")
        print("请把图片/视频放进去，然后重新运行。")
        print("支持图片:", sorted(IMG_EXTS))
        print("支持视频:", sorted(VID_EXTS))
        return

    print("[INFO] Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    records: List[CropRecord] = []
    idx = 1  # 编号从 00001 开始

    # 总进度条（按文件）
    file_pbar = tqdm(media, desc="Processing files", ncols=100)

    for path in file_pbar:
        suf = path.suffix.lower()
        file_pbar.set_postfix(file=path.name)

        if suf in IMG_EXTS:
            idx = process_image(model, path, out_dir, idx, records)
        elif suf in VID_EXTS:
            idx = process_video(model, path, out_dir, idx, records)
        else:
            continue

    # 输出 CSV
    write_csv(records, csv_path)

    print("\n==================== DONE ====================")
    print(f"Input dir   : {input_dir.resolve()}")
    print(f"Output dir  : {out_dir.resolve()}")
    print(f"CSV path    : {csv_path.resolve()}")
    print(f"Crops saved : {len(records)}")
    if len(records) > 0:
        # 简单统计
        avg_conf = sum(r.conf for r in records) / len(records)
        print(f"Avg conf    : {avg_conf:.4f}")
    print("=============================================\n")
    print("下一步：把 orimage 里的 crop 图按型号放进 data/train/ 与 data/val/ 再训练 Swin。")


if __name__ == "__main__":
    main()
