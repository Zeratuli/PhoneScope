from pathlib import Path
import numpy as np
import cv2

def make_class_images(out_dir: Path, n: int, kind: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        # 两个类别用不同“明显特征”，让模型能学到（方便验证流程）
        if kind == "class_A":
            # 画一个大圆 + 少量噪声
            center = (rng.integers(80, 176), rng.integers(80, 176))
            radius = int(rng.integers(40, 70))
            cv2.circle(img, center, radius, (255, 255, 255), -1)
        else:
            # 画一个大方块 + 少量噪声
            x1 = int(rng.integers(60, 120))
            y1 = int(rng.integers(60, 120))
            size = int(rng.integers(80, 120))
            cv2.rectangle(img, (x1, y1), (x1 + size, y1 + size), (255, 255, 255), -1)

        noise = rng.integers(0, 35, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        cv2.imwrite(str(out_dir / f"{kind}_{i:04d}.jpg"), img)

def main():
    root = Path(__file__).parent.resolve() / "data"
    # train/val 结构（ImageFolder 要求）
    trainA = root / "train" / "class_A"
    trainB = root / "train" / "class_B"
    valA   = root / "val"   / "class_A"
    valB   = root / "val"   / "class_B"

    # 生成数据：训练每类 200 张，验证每类 50 张（很快）
    make_class_images(trainA, n=200, kind="class_A", seed=1)
    make_class_images(trainB, n=200, kind="class_B", seed=2)
    make_class_images(valA,   n=50,  kind="class_A", seed=3)
    make_class_images(valB,   n=50,  kind="class_B", seed=4)

    print("[DONE] Dummy dataset created at:", root)
    print("Structure:")
    print("data/train/class_A/*.jpg")
    print("data/train/class_B/*.jpg")
    print("data/val/class_A/*.jpg")
    print("data/val/class_B/*.jpg")

if __name__ == "__main__":
    main()
