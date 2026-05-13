from __future__ import annotations

from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import timm


class DualHeadClassifier(nn.Module):
    """
    双头分类器: 共享 backbone + 厂商分类头 + 型号分类头。

    backbone 使用 timm 创建 (mobilenetv3_large_100 / swin_tiny_patch4_window7_224 等),
    去掉原始分类头后接两个独立的 Linear 层。
    """

    def __init__(
        self,
        backbone_name: str,
        num_brands: int,
        num_models: int,
        pretrained: bool = True,
        pretrained_source: str = "auto",
    ):
        super().__init__()

        if pretrained_source == "auto":
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0
            )
        else:
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, num_classes=0
            )
            state = torch.load(pretrained_source, map_location="cpu")
            # timm 预训练权重可能包含 classifier 键, 过滤掉即可
            filtered = {
                k: v for k, v in state.items()
                if not k.startswith("classifier") and not k.startswith("head")
            }
            self.backbone.load_state_dict(filtered, strict=False)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.feature_dim = self.backbone(dummy).shape[1]
        self.brand_head = nn.Linear(self.feature_dim, num_brands)
        self.model_head = nn.Linear(self.feature_dim, num_models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (brand_logits, model_logits)。"""
        features = self.backbone(x)
        brand_logits = self.brand_head(features)
        model_logits = self.model_head(features)
        return brand_logits, model_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        [3D匹配接口 - 特征提取]

        提取 backbone 输出的特征向量, 形状 (B, feature_dim)。
        MobileNetV3-Large: 常见为 960 维
        Swin-Tiny:         常见为 768 维

        此特征可用于:
          - 与 3D 模型渲染图的特征做余弦相似度匹配
          - 构建检索数据库 (向量数据库)
          - 度量学习 / 对比学习的嵌入空间
        """
        return self.backbone(x)


def freeze_backbone(model: DualHeadClassifier, ratio: float = 0.8) -> int:
    """
    冻结 backbone 前 ratio 比例的参数。

    返回被冻结的参数数量。
    """
    if ratio <= 0.0:
        return 0

    params = list(model.backbone.parameters())
    num_freeze = int(len(params) * ratio)

    frozen_count = 0
    for p in params[:num_freeze]:
        p.requires_grad = False
        frozen_count += p.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[FREEZE] frozen={frozen_count:,} | "
        f"trainable={trainable:,} / total={total:,}"
    )
    return frozen_count


def match_3d(
    features: torch.Tensor,
    database: Optional[Dict] = None,
) -> Dict:
    """
    [3D匹配预留接口]

    将分类模型提取的特征向量与 3D 模型数据库进行匹配。
    当前为占位实现, 返回空结果。

    ---- 接口位置 ----
    文件: models.py
    调用方式: match_3d(model.extract_features(x), database)

    ---- 参数 ----
    features : torch.Tensor, 形状 (1, feature_dim)
        由 DualHeadClassifier.extract_features() 产生。
        维度由所选 backbone 决定，例如 MobileNetV3-Large 常见为 960，
        Swin-Tiny 常见为 768。

    database : dict, 预期格式:
        {
            "iPhone_13": {
                "features": torch.Tensor,   # 该型号多角度渲染图的平均特征
                "mesh_path": "meshes/iphone13.obj",
            },
            ...
        }

    ---- 返回 ----
    dict: {"matched_model": str, "similarity": float}
          或空 dict (当前占位实现)

    ---- 后续实现思路 ----
    1. 为每款手机创建 3D 模型 (Blender / 网络下载)
    2. 从多角度渲染 2D 图片 (正面/背面/侧面/45度等)
    3. 用同一 backbone 提取每张渲染图的特征
    4. 对每款手机的所有渲染图特征取平均, 存入 database
    5. 推理时对输入图片提取特征, 与 database 中所有型号做余弦相似度
    6. 返回相似度最高的匹配结果
    """
    if database is None:
        return {}

    if len(database) == 0:
        return {}

    features = features.detach().float()
    if features.dim() == 2:
        features = features[0]

    best_name = ""
    best_sim = -1.0

    for name, entry in database.items():
        db_feat = entry["features"].detach().float()
        if db_feat.dim() == 2:
            db_feat = db_feat[0]
        sim = torch.nn.functional.cosine_similarity(
            features.unsqueeze(0), db_feat.unsqueeze(0)
        ).item()
        if sim > best_sim:
            best_sim = sim
            best_name = name

    return {"matched_model": best_name, "similarity": round(best_sim, 4)}
