import random
import shutil
from pathlib import Path

# =======================
# 配置区（一般不用改）
# =======================
BASE_DIR = Path(__file__).resolve().parent / "data"

SRC_DIR = BASE_DIR / "test"   # 当前所有图片所在目录
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

RATIO_TRAIN = 0.7
RATIO_VAL = 0.2
RATIO_TEST = 0.1

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42  # 固定随机种子，保证可复现


# =======================
# 工具函数
# =======================
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# =======================
# 主逻辑
# =======================
def main():
    random.seed(RANDOM_SEED)

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"找不到源目录：{SRC_DIR}")

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)

    class_dirs = [d for d in SRC_DIR.iterdir() if d.is_dir()]

    if not class_dirs:
        raise RuntimeError(f"{SRC_DIR} 下没有任何类别文件夹")

    print(f"[INFO] 发现 {len(class_dirs)} 个类别")
    print("===================================")

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n[CLASS] {class_name}")

        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]

        if len(images) < 3:
            print(f"  ⚠️ 图片数量过少（{len(images)}），跳过")
            continue

        random.shuffle(images)

        total = len(images)
        n_train = int(total * RATIO_TRAIN)
        n_val = int(total * RATIO_VAL)
        n_test = total - n_train - n_val  # 保证总数不丢

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f"  总数: {total}")
        print(f"  train: {len(train_imgs)} | val: {len(val_imgs)} | test: {len(test_imgs)}")

        # 目标目录
        train_cls_dir = TRAIN_DIR / class_name
        val_cls_dir = VAL_DIR / class_name
        test_cls_dir = TEST_DIR / class_name

        ensure_dir(train_cls_dir)
        ensure_dir(val_cls_dir)
        ensure_dir(test_cls_dir)

        # 移动文件
        for p in train_imgs:
            shutil.move(str(p), str(train_cls_dir / p.name))

        for p in val_imgs:
            shutil.move(str(p), str(val_cls_dir / p.name))

        # test 中的图片可以选择“保留不动”，也可以统一移动到 test 目录
        for p in test_imgs:
            shutil.move(str(p), str(test_cls_dir / p.name))

        # 清理空目录
        try:
            class_dir.rmdir()
        except OSError:
            pass

    print("\n==================== DONE ====================")
    print("数据已按 7:2:1 随机划分完成")
    print("目录结构：data/train | data/val | data/test")
    print("=============================================")


if __name__ == "__main__":
    main()
