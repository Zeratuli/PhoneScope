# ============================================================
# 手机分类系统 - 全局配置
# 支持 MobileNetV3-Large / Swin-Tiny 双头训练
# ============================================================

# ---- 数据 ----
DATA_DIR = "data"                       # 包含 train/val/test，每层为 厂商/型号/图片
# 目录层级仍按 brand/model 组织；内部标签统一转换为 canonical key
IMG_SIZE = 224
AUG_LEVEL = "light"                     # none / light / medium
READ_BACKEND = "pil"                    # "pil" 或 "tv"(torchvision.io)

# ---- 模型 ----
# 切换模型只需改这一行:
#   "mobilenetv3_large_100"         (~5.5M params, 轻量部署主干)
#   "swin_tiny_patch4_window7_224"  (~27.5M params, 精度对比主干)
MODEL_NAME = "mobilenetv3_large_100"
PRETRAINED = True
# "auto" = timm 自动下载预训练权重 (需联网)
# 也可填本地 .pt 路径, 例如 r"weights/mobilenetv3_large_100.pth"
PRETRAINED_SOURCE = "auto"

# ---- 双头损失权重 ----
BRAND_LOSS_WEIGHT = 0.3                 # 厂商分类损失权重 α
MODEL_LOSS_WEIGHT = 0.7                 # 型号分类损失权重 β

# ---- 训练 ----
EPOCHS = 30
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.05
FREEZE_BACKBONE_RATIO = 0.0            # 冻结 backbone 前 N% 的层 (0.0=不冻结)

GRAD_ACCUM = 1
USE_AMP = True
MAX_GRAD_NORM = 1.0

# ---- 设备 ----
USE_CPU = False
WORKERS = 2                             # Windows 下建议 0 或 2

# ---- 输出 ----
# 为空时按 backbone 自动选择:
#   MobileNetV3 -> runs/mobilenetv3
#   Swin-Tiny   -> runs/swin
OUT_DIR = ""

# ---- 评估 ----
TEST_EVERY = 5                          # 每 N 轮跑一次 test 集
