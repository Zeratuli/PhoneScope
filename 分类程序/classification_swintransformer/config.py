from pathlib import Path

HERE = Path(__file__).resolve().parent

# =========================
# Swin Trainer Config
# =========================

# 数据
DATA_DIR = str(HERE / "data")  # 包含 train / val / test
IMG_SIZE = 224

# 数据增强：none / light / medium
AUG_LEVEL = "medium"

# 关键：数据读取/缓存策略
#  - "pil": 默认PIL读取（兼容最好）
#  - "tv": torchvision.io.read_image 读取（通常更快）
READ_BACKEND = "tv"

#  - "none": 不缓存（省内存）
#  - "ram": 训练集缓存到内存（占用大，但CPU/IO压力会明显下降）
CACHE_MODE = "none"

# 模型
MODEL_NAME = "swin_tiny_patch4_window7_224"
PRETRAINED = True

# 训练
EPOCHS = 15
BATCH_SIZE = 16       # 你5060ti 16G，48大概率OK；不行就降到32
LR = 3e-4
WEIGHT_DECAY = 0.05

GRAD_ACCUM = 2
USE_AMP = True
MAX_GRAD_NORM = 1.0

# 设备
USE_CPU = False

# 关键：Windows 下想提高吞吐就得 workers>0（可能会卡死，卡死就改回0/2）
WORKERS = 2

# 输出
OUT_DIR = "runs/swin"
# 若 OUT_DIR 已存在且非空，自动创建 OUT_DIR_2/_3/... 避免覆盖历史实验
AUTO_INCREMENT_OUT = True

# 评估
TEST_EVERY = 5
