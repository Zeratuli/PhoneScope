import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

import cv2
import numpy as np


# =========================
# 配置区（只改这里）
# =========================
WEIGHTS_PATH = r"runs\swin\best.pt"
CLASSES_PATH = r"runs\swin\classes.txt"
INPUT_DIR = r"data\test"  # 可换成任意图片目录
SHOW_SECONDS = 3
TOPK = 5
IMG_SIZE = 224
MODEL_NAME = "swin_tiny_patch4_window7_224"
SAVE_CSV = r"runs\swin\predict_results.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes(classes_path: str):
    p = Path(classes_path)
    if not p.exists():
        raise FileNotFoundError(f"找不到 classes.txt：{p.resolve()}")
    classes = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise RuntimeError("classes.txt 为空")
    return classes


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def iter_images(input_dir: str):
    root = Path(input_dir)
    if root.is_file():
        if root.suffix.lower() in IMG_EXTS:
            yield root
        return

    if not root.exists():
        raise FileNotFoundError(f"输入路径不存在：{root.resolve()}")

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


@torch.no_grad()
def predict_one(model, tf, img_path: Path, device, classes, topk=5):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = F.softmax(logits, dim=1)[0]

    k = min(topk, prob.numel())
    scores, idxs = torch.topk(prob, k=k, dim=0)

    results = []
    for s, i in zip(scores.tolist(), idxs.tolist()):
        results.append((classes[i], float(s), int(i)))

    best_name, best_score, _ = results[0]
    return best_name, best_score, results, img


def pil_to_cv2(pil_img: Image.Image, max_h=720):
    """PIL(RGB) -> OpenCV(BGR)，并按高度缩放以适应屏幕"""
    rgb = np.array(pil_img)  # HWC RGB uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    if h > max_h:
        scale = max_h / h
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def make_text_panel(lines, h, w=520):
    """右侧文字面板"""
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    y = 30
    for i, line in enumerate(lines):
        font_scale = 0.65 if i == 0 else 0.55
        thickness = 2 if i == 0 else 1
        cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (10, 10, 10), thickness, cv2.LINE_AA)
        y += 30
    return panel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    classes = load_classes(CLASSES_PATH)
    tf = build_transform(IMG_SIZE)

    # 模型
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(classes))
    w = Path(WEIGHTS_PATH)
    if not w.exists():
        raise FileNotFoundError(f"找不到权重文件：{w.resolve()}")
    state = torch.load(str(w), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    images = list(iter_images(INPUT_DIR))
    if not images:
        raise RuntimeError(f"在 {Path(INPUT_DIR).resolve()} 中没有找到图片")
    print(f"[INFO] found images = {len(images)}")

    Path(SAVE_CSV).parent.mkdir(parents=True, exist_ok=True)
    csv_header = ["image_path", "pred_top1", "conf_top1"]
    for i in range(2, TOPK + 1):
        csv_header += [f"pred_top{i}", f"conf_top{i}"]
    csv_rows = []

    window_name = "Swin Predict (ESC=quit, SPACE=pause, n=next)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False

    for n, img_path in enumerate(images, start=1):
        while True:
            # 暂停状态下等待按键
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):  # SPACE
                    paused = False
                elif key == ord('n'):
                    break
                continue
            else:
                break

        try:
            top1_name, top1_conf, topk_list, pil_img = predict_one(
                model, tf, img_path, device, classes, topk=TOPK
            )
        except Exception as e:
            print(f"[SKIP] 失败：{img_path} | {e}")
            continue

        print(f"[{n}/{len(images)}] {img_path.name} -> {top1_name} ({top1_conf:.4f})")

        # CSV
        row = [str(img_path), top1_name, f"{top1_conf:.6f}"]
        for (name, conf, _) in topk_list[1:]:
            row += [name, f"{conf:.6f}"]
        while len(row) < len(csv_header):
            row += ["", ""]
        csv_rows.append(row)

        # 画面拼接
        img_bgr = pil_to_cv2(pil_img, max_h=720)
        h = img_bgr.shape[0]

        lines = [
            f"{n}/{len(images)}  {img_path.name}",
            f"Top-1: {top1_name}  ({top1_conf:.4f})",
            "",
        ]
        for rank, (name, conf, _) in enumerate(topk_list, start=1):
            lines.append(f"Top-{rank}: {name}  {conf:.4f}")
        lines.append("")
        lines.append("Keys: ESC quit | SPACE pause/resume | n next")

        panel = make_text_panel(lines, h=h, w=520)

        canvas = np.hstack([img_bgr, panel])
        cv2.imshow(window_name, canvas)

        # 等待 SHOW_SECONDS 秒或按键
        t0 = time.time()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                # 写 CSV 再退出
                with open(SAVE_CSV, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_header)
                    writer.writerows(csv_rows)
                print(f"[DONE] results saved to: {Path(SAVE_CSV).resolve()}")
                return
            elif key == ord(' '):  # pause
                paused = True
                break
            elif key == ord('n'):  # next
                break

            if (time.time() - t0) >= SHOW_SECONDS:
                break

            if paused:
                break

        # 如果暂停，会回到外层 while 等待按键

    # 完成：写 CSV
    with open(SAVE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    print(f"[DONE] results saved to: {Path(SAVE_CSV).resolve()}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
