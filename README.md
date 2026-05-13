# PhoneScope

基于改进 YOLO 与双头细粒度分类的智能手机识别系统。

本项目包含三部分：

- `网页程序/`：面向展示与使用的 Web 系统，支持图片、批量图片、视频和摄像头识别
- `分类程序/`：当前主训练工程，使用双头分类器同时预测品牌与型号，支持 `MobileNetV3` 与 `Swin`
- `检测程序/`：YOLO 手机检测训练工程及实验脚本

## 功能概览

- 单张图片检测：上传 1 张图片时执行单图识别
- 批量单张检测：上传多张图片时自动逐张独立识别
- 多图融合识别：上传 2~5 张图片综合输出一个最终结论
- 视频融合识别：上传 1 个视频后抽帧融合识别
- 摄像头识别：浏览器实时拍照 / 连拍识别
- 识别结果展示：标注框、Top-K 预测、手机规格参数
- 识别记录管理：会话日志、统计、软删除、管理员后台
- 报告导出：PDF 报告与标注图 ZIP

## 项目结构

```text
程序/
├── 网页程序/
│   ├── backend/         # FastAPI 后端
│   ├── frontend/        # React + Vite 前端
│   ├── docker-compose.yml
│   └── PhoneScope启动.bat
├── 分类程序/
│   ├── run_train.py     # 双头训练入口
│   ├── train_panel.py   # ttkbootstrap 训练面板
│   ├── models.py        # DualHeadClassifier
│   ├── preprocess.py    # 去背景 + 划分数据集
│   ├── runs/            # 本地产生的训练产物（Git 忽略）
│   └── classification_swintransformer/
│       └── ...          # 历史单头 Swin 基线 / 论文保留工程
├── 检测程序/
│   └── detection_YOLOv11m/
│       └── ultralytics-8.3.163/
│           └── ...      # YOLO 检测训练与实验脚本
└── 技术规格.md
```

## 技术路线

### 识别链路

当前网页系统识别流程为：

1. YOLO 检测手机位置
2. 按检测框裁剪 ROI
3. 双头分类器预测品牌与型号
4. 返回结构化结果并渲染页面

说明：

- 若单图中 YOLO 未检出手机，后端会退化为整图分类，尽量避免完全无结果
- 网页当前支持 `MobileNetV3` 与 `Swin` 两套双头分类权重切换

### 双头分类器

`分类程序/` 当前主训练工程使用共享 backbone + 双输出头：

- `brand_head`：预测品牌
- `model_head`：预测型号

支持两个 backbone：

- `mobilenetv3_large_100`
- `swin_tiny_patch4_window7_224`

## 运行 Web 系统

### 1. 后端配置

请先在 `网页程序/backend/` 下准备本地 `.env`。

仓库中提供了示例配置：

- `网页程序/backend/.env.example`

你可以复制为：

```text
网页程序/backend/.env
```

然后根据你本地的模型、数据库与目录修改实际值。

### 2. 一键启动

Windows 下可直接运行：

```bat
cd /d D:\Graduation_Thesis\程序\网页程序
PhoneScope启动.bat
```

它会：

- 启动 MySQL Docker 容器
- 启动 FastAPI 后端
- 启动 Vite 前端
- 健康检查通过后打开浏览器

### 3. 手动启动

数据库：

```bat
cd /d D:\Graduation_Thesis\程序\网页程序
docker compose -p phonescope up -d
```

后端：

```bat
cd /d D:\Graduation_Thesis\程序\网页程序\backend
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

前端：

```bat
cd /d D:\Graduation_Thesis\程序\网页程序\frontend
npm install
npm run dev
```

默认访问地址：

- 前端：<http://localhost:5173>
- 后端：<http://localhost:8000>
- Swagger：<http://localhost:8000/docs>

## 训练分类模型

### Conda 环境

推荐使用本地 `classification` 环境。

### 命令行训练

进入主训练工程：

```bat
cd /d D:\Graduation_Thesis\程序\分类程序
```

训练 `MobileNetV3`：

```bat
python run_train.py --backbone mobilenetv3
```

训练 `Swin`：

```bat
python run_train.py --backbone swin
```

也支持指定常见参数，例如：

```bat
python run_train.py --backbone mobilenetv3 --epochs 40 --batch-size 16 --lr 1e-4
```

### 可视化训练面板

如果环境已安装 `ttkbootstrap`：

```bat
python train_panel.py
```

训练面板支持：

- 选择 backbone
- 设置 batch、epoch、学习率等常见参数
- 自动按 backbone 选择输出目录
- 实时查看训练状态和日志

### 训练输出目录

训练产物按模型目录与实验编号组织：

```text
分类程序/runs/
├── mobilenetv3/
│   ├── run_001/
│   └── run_002/
└── swin/
    ├── run_001/
    └── run_002/
```

每次训练通常会生成：

- `best.pt`
- `last.pt`
- `brand_classes.txt`
- `model_classes.txt`
- `brand_to_models.json`
- `train_history.csv`
- `train_curves.png`
- `brand_confusion_matrix.png`
- `model_confusion_matrix.png`
- `brand_classification_report.txt`
- `model_classification_report.txt`
- `test_summary.json`
- `gradcam_grid.png`
- `gradcam_samples/`

## 分类数据准备

分类训练数据目录约定为：

```text
raw_data/<品牌>/<型号>/*.jpg
```

如果原图背景噪音较多，可以使用：

```bat
cd /d D:\Graduation_Thesis\程序\分类程序
python preprocess.py --input raw_data --output data
```

这个脚本会：

- 调用 `rembg` 去背景
- 自动切分 `train / val / test`
- 输出到 `data/`

## 检测模型训练

YOLO 检测训练代码位于：

```text
检测程序/detection_YOLOv11m/ultralytics-8.3.163/
```

常用脚本：

- `yolo-phone-train.py`
- `mypredict.py`
- `mytrain.py`

## GitHub 版本说明

为了保证仓库干净和安全，以下内容默认不提交到 Git：

- 本地 `.env`
- 虚拟环境
- `node_modules`
- 训练权重 `*.pt`
- `runs/`、`results/`
- 上传缓存与导出缓存
- 数据集与原始视频

也就是说，GitHub 版本是：

- 源码
- 配置模板
- 文档

而不是你本机完整运行现场。

## 注意事项

- 当前仓库中的真实模型权重和数据集默认不随 Git 提交
- 如果你重新训练出新的 `run_XXX`，Web 后端会自动解析并加载最新一轮训练产物
- 推送到 GitHub 前，建议重新检查 `.env`、数据集和大模型文件是否被忽略

## License

本仓库根目录附带 `LICENSE` 文件。若项目中引用第三方代码或模型，请遵守其各自许可证要求。
