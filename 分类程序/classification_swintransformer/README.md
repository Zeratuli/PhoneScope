# Swin Transformer 手机型号分类

## 目录结构

classification/
├─ data/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ runs/
│  └─ swin/
├─ train_swin.py
└─ init_project.py

## 训练方式

python train_swin.py --data_dir data --epochs 30 --batch_size 32
