import json
import importlib.util
import logging
import random
from pathlib import Path

from PIL import Image

from app.schemas.models import ClassificationItem, TopKItem, PhoneSpec
from app.services.phone_data import (
    get_phone_identity,
    get_phone_info,
    normalize_model_key,
)

PHONE_CLASSES = ["HUAWEI_NOVA_10", "REDMI_K80_Pro", "iPhone_13"]
BRAND_CLASSES = ["Apple", "Huawei", "Xiaomi"]
logger = logging.getLogger(__name__)


class ClassifierService:
    def __init__(
        self,
        weights_path: str = "",
        classes_path: str = "",
        model_name: str = "",
        *,
        dualhead: bool = False,
        brand_classes_path: str = "",
        model_classes_path: str = "",
        brand_to_models_path: str = "",
        backbone_name: str = "",
    ):
        self._model = None
        self._classes = PHONE_CLASSES
        self._brand_classes = BRAND_CLASSES
        self._brand_to_models = {}
        self._mock = not weights_path
        self._dualhead = dualhead
        self._backbone_name = backbone_name or model_name or "mobilenetv3_large_100"
        if dualhead and weights_path and brand_classes_path and model_classes_path:
            self._load_dualhead_model(
                weights_path,
                brand_classes_path,
                model_classes_path,
                brand_to_models_path,
                self._backbone_name,
            )
        elif weights_path and classes_path:
            self._load_singlehead_model(weights_path, classes_path, model_name)

    def _load_singlehead_model(self, weights_path: str, classes_path: str, model_name: str) -> None:
        try:
            import torch
            import timm

            classes_file = Path(classes_path)
            if classes_file.exists():
                self._classes = [
                    line.strip()
                    for line in classes_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            self._model = timm.create_model(
                model_name or "mobilenetv3_large_100",
                pretrained=False,
                num_classes=len(self._classes),
            )
            state = torch.load(weights_path, map_location="cpu")
            self._model.load_state_dict(state, strict=True)
            self._model.eval()
            self._mock = False
        except Exception as exc:
            logger.exception("single-head classifier load failed: %s", exc)
            self._mock = True

    def _load_dualhead_model(
        self,
        weights_path: str,
        brand_classes_path: str,
        model_classes_path: str,
        brand_to_models_path: str,
        backbone_name: str,
    ) -> None:
        try:
            import torch

            classification_root = (
                Path(__file__).resolve().parents[4] / "分类程序"
            )
            models_file = classification_root / "models.py"
            if not models_file.exists():
                raise FileNotFoundError(f"找不到双头模型定义文件: {models_file}")
            spec = importlib.util.spec_from_file_location(
                "phonescope_dualhead_models",
                models_file,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载双头模型模块: {models_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DualHeadClassifier = module.DualHeadClassifier

            brand_file = Path(brand_classes_path)
            model_file = Path(model_classes_path)
            if brand_file.exists():
                self._brand_classes = [
                    line.strip()
                    for line in brand_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            if model_file.exists():
                self._classes = [
                    line.strip()
                    for line in model_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            mapping_file = Path(brand_to_models_path)
            if mapping_file.exists():
                self._brand_to_models = json.loads(mapping_file.read_text(encoding="utf-8"))

            self._model = DualHeadClassifier(
                backbone_name=backbone_name,
                num_brands=len(self._brand_classes),
                num_models=len(self._classes),
                pretrained=False,
            )
            state = torch.load(weights_path, map_location="cpu")
            self._model.load_state_dict(state, strict=True)
            self._model.eval()
            self._mock = False
            self._dualhead = True
        except Exception as exc:
            logger.exception("dual-head classifier load failed: %s", exc)
            self._mock = True

    def is_loaded(self) -> bool:
        return self._model is not None or self._mock

    def classify(self, crop: Image.Image, topk: int = 3) -> ClassificationItem:
        if self._mock:
            return self._mock_classify(topk)
        if self._dualhead:
            return self._real_dualhead_classify(crop, topk)
        return self._real_singlehead_classify(crop, topk)

    def _build_transform(self):
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize(int(224 * 1.15)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _real_singlehead_classify(self, crop: Image.Image, topk: int) -> ClassificationItem:
        import torch
        import torch.nn.functional as F

        tf = self._build_transform()
        x = tf(crop.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
            prob = F.softmax(logits, dim=1)[0]

        k = min(topk, prob.numel())
        scores, idxs = torch.topk(prob, k=k)
        top_k_list = [
            TopKItem(
                name=normalize_model_key(self._classes[i]) or self._classes[i],
                confidence=round(float(s), 4),
            )
            for s, i in zip(scores.tolist(), idxs.tolist())
        ]
        return self._build_result(top_k_list)

    def _real_dualhead_classify(self, crop: Image.Image, topk: int) -> ClassificationItem:
        import torch
        import torch.nn.functional as F

        tf = self._build_transform()
        x = tf(crop.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            brand_logits, model_logits = self._model(x)
            brand_prob = F.softmax(brand_logits, dim=1)[0]
            model_prob = F.softmax(model_logits, dim=1)[0]

        bk = min(topk, brand_prob.numel())
        mk = min(topk, model_prob.numel())
        _brand_scores, brand_idxs = torch.topk(brand_prob, k=bk)
        model_scores, model_idxs = torch.topk(model_prob, k=mk)
        top_k_list = [
            TopKItem(
                name=normalize_model_key(self._classes[i]) or self._classes[i],
                confidence=round(float(s), 4),
            )
            for s, i in zip(model_scores.tolist(), model_idxs.tolist())
        ]

        result = self._build_result(top_k_list)
        if brand_idxs.numel() > 0:
            brand_name = self._brand_classes[brand_idxs[0].item()]
            result.brand_name = brand_name
        return result

    def _mock_classify(self, topk: int) -> ClassificationItem:
        weights = [random.random() for _ in self._classes]
        total = sum(weights)
        probs = sorted(
            [(cls, w / total) for cls, w in zip(self._classes, weights)],
            key=lambda x: x[1],
            reverse=True,
        )
        top_k_list = [
            TopKItem(
                name=normalize_model_key(name) or name,
                confidence=round(conf, 4),
            )
            for name, conf in probs[:topk]
        ]
        return self._build_result(top_k_list)

    def _build_result(self, top_k_list: list[TopKItem]) -> ClassificationItem:
        canonical_name = top_k_list[0].name
        info = get_phone_info(canonical_name)
        identity = get_phone_identity(canonical_name)
        spec = PhoneSpec(**info) if info else None
        return ClassificationItem(
            model_name=canonical_name,
            brand_name=identity.get("brand_name") if identity else None,
            series_name=identity.get("series_name") if identity else None,
            display_name=identity.get("display_name") if identity else None,
            confidence=top_k_list[0].confidence,
            top_k=top_k_list,
            phone_spec=spec,
        )
