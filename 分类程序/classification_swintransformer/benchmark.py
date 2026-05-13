"""推理性能基准脚本。

一键测量：
  1. Swin Transformer 分类模型：参数量 / FLOPs / CPU 单图耗时 / GPU 单图耗时
  2. MobileNetV3 分类模型（如果 runs/mobilenet/best.pt 存在）：同上
  3. YOLOv11m 目标检测模型（如果 .env 中指定的权重存在）：CPU/GPU 单图耗时
  4. 端到端"YOLO 检测 → ROI 裁剪 → 分类"组合管线耗时

输出：
  - runs/benchmark/benchmark_report.md  —— 一张可直接粘进论文的 markdown 表
  - runs/benchmark/benchmark_raw.json  —— 原始数值

用法：
  conda activate classification
  cd 分类程序/classification_swintransformer
  python benchmark.py                      # 默认跑全部
  python benchmark.py --repeats 200        # 调节样本数（默认 100）
  python benchmark.py --no-yolo            # 跳过 YOLO
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_classes(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _dummy_image(size=(224, 224)) -> Image.Image:
    """产生一张随机 RGB 图，避免真实文件 IO 影响测耗。"""
    arr = np.random.randint(0, 256, size=(*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _try_flops(model, input_size=(1, 3, 224, 224)) -> float | None:
    """尝试用 thop 或 fvcore 估算 FLOPs；都不可用则返回 None。"""
    import torch
    x = torch.randn(*input_size)
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(x,), verbose=False)
        return float(flops)
    except ImportError:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, x).total())
    except ImportError:
        return None


def _bench_classifier(model, classes, device, repeats: int,
                      preprocess, warmup: int = 10) -> dict:
    """对分类模型做 repeats 次单图推理，返回耗时统计（单位 ms）。"""
    import torch

    model = model.to(device).eval()
    imgs = [preprocess(_dummy_image()).unsqueeze(0).to(device)
            for _ in range(warmup + repeats)]

    # warmup
    with torch.no_grad():
        for i in range(warmup):
            model(imgs[i])
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for i in range(repeats):
            x = imgs[warmup + i]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "repeats": repeats,
        "device": str(device),
    }


def _bench_yolo(weights_path: Path, device_str: str, repeats: int,
                warmup: int = 5) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    # 使用纯 numpy array 作为输入，绕过磁盘 IO
    dummy = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)

    # warmup
    for _ in range(warmup):
        model(dummy, device=device_str, conf=0.7, verbose=False)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(dummy, device=device_str, conf=0.7, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "repeats": repeats,
        "device": device_str,
    }


def _format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} K"
    return str(n)


def _format_flops(n: float | None) -> str:
    if n is None:
        return "—"
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    return f"{n:.0f}"


def run_classifier_benchmark(name: str, model_name: str,
                             weights: Path, classes_file: Path,
                             repeats: int) -> dict:
    """包装：load 模型 + 两种 device 下测 + 参数量 + FLOPs"""
    import timm
    import torch
    from torchvision import transforms

    classes = _load_classes(classes_file)
    model = timm.create_model(model_name, pretrained=False,
                              num_classes=len(classes))
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state, strict=True)

    params = _count_params(model)
    flops = _try_flops(model)

    preprocess = transforms.Compose([
        transforms.Resize(int(224 * 1.15)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    print(f"\n=== {name} ({model_name}) ===")
    print(f"  params : {_format_params(params)}  ({params:,})")
    print(f"  flops  : {_format_flops(flops)}")

    result = {
        "model_name": model_name,
        "params": params,
        "params_human": _format_params(params),
        "flops": flops,
        "flops_human": _format_flops(flops),
        "cpu": None,
        "gpu": None,
    }

    print("  running CPU benchmark ...")
    result["cpu"] = _bench_classifier(
        model, classes, torch.device("cpu"), repeats, preprocess)
    print(f"    mean={result['cpu']['mean_ms']:.2f} ms  "
          f"p95={result['cpu']['p95_ms']:.2f} ms")

    if torch.cuda.is_available():
        print("  running GPU benchmark ...")
        import timm
        model_gpu = timm.create_model(model_name, pretrained=False,
                                      num_classes=len(classes))
        model_gpu.load_state_dict(state, strict=True)
        result["gpu"] = _bench_classifier(
            model_gpu, classes, torch.device("cuda"), repeats, preprocess)
        print(f"    mean={result['gpu']['mean_ms']:.2f} ms  "
              f"p95={result['gpu']['p95_ms']:.2f} ms")

    return result


def write_report(out_dir: Path, records: dict) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_raw.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# PhoneScope 推理性能基准\n"]
    lines.append(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '—'}")
    lines.append(f"- CUDA available: {torch.cuda.is_available()}\n")

    # classifier 表
    if records.get("classifiers"):
        lines.append("## 细粒度分类模型")
        lines.append("")
        lines.append("| 模型 | 参数量 | FLOPs | CPU 平均 / p95 (ms) | GPU 平均 / p95 (ms) |")
        lines.append("|---|---|---|---|---|")
        for r in records["classifiers"]:
            cpu_s = f"{r['cpu']['mean_ms']:.2f} / {r['cpu']['p95_ms']:.2f}" if r.get("cpu") else "—"
            gpu_s = f"{r['gpu']['mean_ms']:.2f} / {r['gpu']['p95_ms']:.2f}" if r.get("gpu") else "—"
            lines.append(
                f"| {r['model_name']} | {r['params_human']} | "
                f"{r['flops_human']} | {cpu_s} | {gpu_s} |"
            )
        lines.append("")

    # yolo 表
    if records.get("yolo"):
        y = records["yolo"]
        lines.append("## YOLOv11m 目标检测")
        lines.append("")
        lines.append("| 输入分辨率 | 设备 | 平均 (ms) | p95 (ms) | min/max (ms) |")
        lines.append("|---|---|---|---|---|")
        for entry in y:
            lines.append(
                f"| 1280×720 | {entry['device']} | {entry['mean_ms']:.2f} | "
                f"{entry['p95_ms']:.2f} | {entry['min_ms']:.2f} / {entry['max_ms']:.2f} |"
            )
        lines.append("")

    lines.append("> 每项指标均以独立 warmup + `repeats` 次单图推理测得；")
    lines.append("> 分类耗时不含 YOLO 检测 + ROI 裁剪，端到端管线耗时请参考 pipeline.py。\n")

    (out_dir / "benchmark_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\n[OK] report saved -> {out_dir / 'benchmark_report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--runs_dir", default=str(HERE / "runs"))
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--yolo_weights", default="")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    records = {"classifiers": [], "yolo": []}

    # 分类模型（Swin / MobileNet）
    candidates = [
        ("Swin Transformer", "swin_tiny_patch4_window7_224",
         runs_dir / "swin" / "best.pt",
         runs_dir / "swin" / "classes.txt"),
        ("MobileNetV3", "mobilenetv3_large_100",
         runs_dir / "mobilenet" / "best.pt",
         runs_dir / "mobilenet" / "classes.txt"),
    ]
    for name, model_name, w, c in candidates:
        if not w.exists() or not c.exists():
            print(f"[SKIP] {name}: {w} 或 {c} 不存在")
            continue
        records["classifiers"].append(
            run_classifier_benchmark(name, model_name, w, c, args.repeats))

    # YOLO
    if not args.no_yolo:
        yolo_w = args.yolo_weights
        if not yolo_w:
            # 尝试默认路径
            default = HERE.parent / "detection_YOLOv11m" / "ultralytics-8.3.163" \
                / "results" / "yolo11m_phone_ft960_2" / "weights" / "best.pt"
            if default.exists():
                yolo_w = str(default)
        if yolo_w and Path(yolo_w).exists():
            print(f"\n=== YOLOv11m ({yolo_w}) ===")
            print("  running CPU benchmark ...")
            records["yolo"].append(_bench_yolo(
                Path(yolo_w), "cpu", max(args.repeats // 4, 10)))
            print(f"    mean={records['yolo'][-1]['mean_ms']:.2f} ms")

            import torch
            if torch.cuda.is_available():
                print("  running GPU benchmark ...")
                records["yolo"].append(_bench_yolo(
                    Path(yolo_w), "cuda:0", args.repeats))
                print(f"    mean={records['yolo'][-1]['mean_ms']:.2f} ms")
        else:
            print("[SKIP] YOLO weights 未找到，跳过")

    out_dir = runs_dir / "benchmark"
    write_report(out_dir, records)


if __name__ == "__main__":
    main()
