from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, EW, LEFT, NSEW, PRIMARY, SUCCESS
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.widgets.scrolled import ScrolledText

import config as cfg
from datasets import build_datasets
from trainer import train
from utils import default_out_dir_for_backbone, get_backbone_preset, normalize_backbone_name

SURFACE = "#1f1f1f"
TEXT_BRIGHT = "#dbeafe"
TEXT_SOFT = "#cbd5e1"


class TrainPanelApp:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("PhoneScope 分类训练面板")
        self.root.geometry("1380x900")
        self.root.minsize(1260, 780)

        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.training_thread: threading.Thread | None = None

        self.backbone_var = tb.StringVar(value="mobilenetv3")
        self.data_dir_var = tb.StringVar(value=str((Path(__file__).resolve().parent / "data").resolve()))
        self.out_dir_var = tb.StringVar(value=str((Path(__file__).resolve().parent / "runs" / "mobilenetv3").resolve()))
        self.batch_size_var = tb.StringVar(value="32")
        self.epochs_var = tb.StringVar(value="30")
        self.img_size_var = tb.StringVar(value="224")
        self.lr_var = tb.StringVar(value="0.0003")
        self.weight_decay_var = tb.StringVar(value="0.05")
        self.workers_var = tb.StringVar(value="2")
        self.grad_accum_var = tb.StringVar(value="1")
        self.freeze_ratio_var = tb.StringVar(value="0.0")
        self.test_every_var = tb.StringVar(value="5")
        self.aug_level_var = tb.StringVar(value="light")
        self.read_backend_var = tb.StringVar(value="pil")
        self.pretrained_var = tb.BooleanVar(value=True)
        self.amp_var = tb.BooleanVar(value=True)
        self.cpu_var = tb.BooleanVar(value=False)

        self.status_var = tb.StringVar(value="就绪")
        self.progress_var = tb.DoubleVar(value=0.0)
        self.best_acc_var = tb.StringVar(value="—")
        self.summary_var = tb.StringVar(value="等待开始训练")
        self.out_dir_runtime_var = tb.StringVar(value="—")
        self.backbone_buttons: dict[str, tb.Button] = {}

        self._build_ui()
        self.apply_backbone_preset()
        self.root.after(120, self._drain_queue)

    def _build_ui(self):
        shell = tb.Frame(self.root, padding=18)
        shell.pack(fill=BOTH, expand=True)
        shell.columnconfigure(0, weight=7)
        shell.columnconfigure(1, weight=8)
        shell.rowconfigure(1, weight=1)

        header = tb.Frame(shell)
        header.grid(row=0, column=0, columnspan=2, sticky=EW, pady=(0, 16))
        tb.Label(header, text="PhoneScope 分类训练面板", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text="双头训练框架：品牌头 + 型号头，支持 MobileNetV3 与 Swin 主干。训练完成后自动生成曲线图、混淆矩阵、分类报告、Grad-CAM 和更多分析图。",
            font=("Segoe UI", 10),
            fg=TEXT_BRIGHT,
            bg=SURFACE,
        ).pack(anchor="w", pady=(6, 0))

        left = tb.Frame(shell)
        left.grid(row=1, column=0, sticky=NSEW, padx=(0, 12))
        left.columnconfigure(0, weight=1)

        right = tb.Frame(shell)
        right.grid(row=1, column=1, sticky=NSEW)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_form_panel(left)
        self._build_flags_panel(left)
        self._build_actions_panel(left)

        self._build_status_panel(right)
        self._build_log_panel(right)

    def _build_form_panel(self, parent):
        card = tb.Labelframe(parent, text="训练参数", padding=16)
        card.pack(fill="x", pady=(0, 14))
        tk.Label(
            card,
            text="常见超参数集中在这里，输出目录会按 backbone 自动切换。每次训练会在模型目录下自动创建新的 run_XXX 文件夹。",
            font=("Segoe UI", 9),
            fg=TEXT_SOFT,
            bg=SURFACE,
            anchor="w",
            justify=LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        for col in range(4):
            card.columnconfigure(col, weight=1, minsize=140)

        fields = [
            ("Backbone", self._backbone_selector),
            ("Batch Size", lambda p: self._entry(p, self.batch_size_var)),
            ("Epochs", lambda p: self._entry(p, self.epochs_var)),
            ("Image Size", lambda p: self._entry(p, self.img_size_var)),
            ("Learning Rate", lambda p: self._entry(p, self.lr_var)),
            ("Weight Decay", lambda p: self._entry(p, self.weight_decay_var)),
            ("Workers", lambda p: self._entry(p, self.workers_var)),
            ("Grad Accum", lambda p: self._entry(p, self.grad_accum_var)),
            ("Freeze Ratio", lambda p: self._entry(p, self.freeze_ratio_var)),
            ("Test Every", lambda p: self._entry(p, self.test_every_var)),
            ("增强等级", lambda p: self._combo(p, self.aug_level_var, ["none", "light", "medium"])),
            ("读取后端", lambda p: self._combo(p, self.read_backend_var, ["pil", "tv"])),
        ]

        for idx, (label, builder) in enumerate(fields):
            row = 1 + idx // 4
            col = idx % 4
            self._field_block(card, row, col, label, builder)

        self._field_block(card, 4, 0, "数据目录", lambda p: self._path_picker(p, self.data_dir_var, self.pick_data_dir), colspan=2)
        self._field_block(card, 4, 2, "输出目录", lambda p: self._path_picker(p, self.out_dir_var, self.pick_out_dir), colspan=2)

    def _build_flags_panel(self, parent):
        card = tb.Labelframe(parent, text="运行选项", padding=16)
        card.pack(fill="x", pady=(0, 14))
        tb.Checkbutton(card, text="使用预训练权重", variable=self.pretrained_var, bootstyle="round-toggle").grid(row=0, column=0, sticky="w", padx=(0, 14), pady=6)
        tb.Checkbutton(card, text="启用 AMP 混合精度", variable=self.amp_var, bootstyle="round-toggle").grid(row=0, column=1, sticky="w", padx=(0, 14), pady=6)
        tb.Checkbutton(card, text="强制使用 CPU", variable=self.cpu_var, bootstyle="round-toggle").grid(row=0, column=2, sticky="w", pady=6)

    def _build_actions_panel(self, parent):
        card = tb.Labelframe(parent, text="执行操作", padding=16)
        card.pack(fill="x")
        card.columnconfigure(0, weight=0)
        card.columnconfigure(1, weight=1)

        tb.Button(card, text="开始训练", bootstyle=PRIMARY, width=18, command=self.start_training).grid(row=0, column=0, sticky="w", padx=(0, 10), ipady=6)
        tb.Label(card, textvariable=self.out_dir_runtime_var, bootstyle="info", anchor="w").grid(row=0, column=1, sticky=EW)

    def _build_status_panel(self, parent):
        card = tb.Labelframe(parent, text="训练状态", padding=16)
        card.grid(row=0, column=0, sticky=EW, pady=(0, 14))
        for col in range(3):
            card.columnconfigure(col, weight=1)

        self._metric_box(card, 0, "状态", self.status_var, "primary")
        self._metric_box(card, 1, "最佳型号精度", self.best_acc_var, "success")
        self._metric_box(card, 2, "输出目录", self.out_dir_runtime_var, "info")

        tb.Progressbar(card, variable=self.progress_var, maximum=100, bootstyle=(SUCCESS, "striped")).grid(row=1, column=0, columnspan=3, sticky=EW, pady=16)
        tb.Label(card, textvariable=self.summary_var, bootstyle="light", wraplength=560, justify=LEFT).grid(row=2, column=0, columnspan=3, sticky="w")

    def _build_log_panel(self, parent):
        card = tb.Labelframe(parent, text="训练日志", padding=16)
        card.grid(row=1, column=0, sticky=NSEW)
        card.rowconfigure(0, weight=1)
        card.columnconfigure(0, weight=1)

        self.log_text = ScrolledText(
            card,
            autohide=True,
            bootstyle="dark-round",
            font=("Consolas", 10),
            height=24,
        )
        self.log_text.grid(row=0, column=0, sticky=NSEW)
        self.log_text.text.configure(state="disabled")

    def _metric_box(self, parent, column, title: str, variable: tb.StringVar, style: str):
        box = tb.Frame(parent, bootstyle=style, padding=12)
        box.grid(row=0, column=column, sticky=EW, padx=(0 if column == 0 else 8, 0))
        tb.Label(box, text=title, bootstyle=f"{style}-inverse").pack(anchor="w")
        tb.Label(box, textvariable=variable, font=("Segoe UI", 10, "bold"), bootstyle=f"{style}-inverse").pack(anchor="w", pady=(6, 0))

    def _field_block(self, parent, row, col, label, widget_builder, colspan=1):
        wrap = tb.Frame(parent)
        wrap.grid(row=row, column=col, columnspan=colspan, sticky=EW, padx=6, pady=6)
        wrap.columnconfigure(0, weight=1)
        tk.Label(
            wrap,
            text=label,
            font=("Segoe UI", 9, "bold"),
            fg=TEXT_BRIGHT,
            bg=SURFACE,
            anchor="w",
        ).pack(anchor="w", pady=(0, 6))
        widget = widget_builder(wrap)
        widget.pack(fill="x")

    def _entry(self, parent, variable):
        return tb.Entry(parent, textvariable=variable, bootstyle="light")

    def _combo(self, parent, variable, values, bind_sync=False):
        combo = tb.Combobox(parent, textvariable=variable, values=values, state="readonly", bootstyle="light")
        if bind_sync:
            combo.bind("<<ComboboxSelected>>", lambda _event: self.sync_out_dir())
        return combo

    def _backbone_selector(self, parent):
        frame = tb.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self.backbone_buttons["mobilenetv3"] = tb.Button(
            frame,
            text="MobileNetV3",
            command=lambda: self.select_backbone("mobilenetv3"),
        )
        self.backbone_buttons["mobilenetv3"].grid(row=0, column=0, sticky=EW, padx=(0, 6))
        self.backbone_buttons["swin"] = tb.Button(
            frame,
            text="Swin",
            command=lambda: self.select_backbone("swin"),
        )
        self.backbone_buttons["swin"].grid(row=0, column=1, sticky=EW, padx=(6, 0))
        self._refresh_backbone_buttons()
        return frame

    def _path_picker(self, parent, variable, command):
        frame = tb.Frame(parent)
        frame.columnconfigure(0, weight=1)
        tb.Entry(frame, textvariable=variable, bootstyle="light").grid(row=0, column=0, sticky=EW)
        tb.Button(frame, text="选择", bootstyle="info", command=command, width=8).grid(row=0, column=1, padx=(8, 0))
        return frame

    def select_backbone(self, value: str):
        self.backbone_var.set(value)
        self._refresh_backbone_buttons()
        self.apply_backbone_preset()

    def _refresh_backbone_buttons(self):
        current = self.backbone_var.get()
        for name, button in self.backbone_buttons.items():
            button.configure(
                bootstyle=SUCCESS if name == current else "outline-light"
            )

    def pick_data_dir(self):
        path = filedialog.askdirectory(title="选择训练数据目录", initialdir=self.data_dir_var.get(), parent=self.root)
        if path:
            self.data_dir_var.set(path)

    def pick_out_dir(self):
        path = filedialog.askdirectory(title="选择训练输出目录", initialdir=self.out_dir_var.get(), parent=self.root)
        if path:
            self.out_dir_var.set(path)

    def sync_out_dir(self):
        try:
            normalized = normalize_backbone_name(self.backbone_var.get())
            path = Path(__file__).resolve().parent / default_out_dir_for_backbone(normalized)
            resolved = str(path.resolve())
            self.out_dir_var.set(resolved)
            self.out_dir_runtime_var.set(resolved)
        except Exception as exc:
            self.append_log(f"[WARN] 无法同步输出目录: {exc}\n")

    def apply_backbone_preset(self):
        try:
            normalized = normalize_backbone_name(self.backbone_var.get())
            preset = get_backbone_preset(normalized)
            self.batch_size_var.set(str(preset["BATCH_SIZE"]))
            self.epochs_var.set(str(preset["EPOCHS"]))
            self.img_size_var.set(str(preset["IMG_SIZE"]))
            self.lr_var.set(str(preset["LR"]))
            self.weight_decay_var.set(str(preset["WEIGHT_DECAY"]))
            self.workers_var.set(str(getattr(cfg, "WORKERS", 2)))
            self.grad_accum_var.set(str(preset["GRAD_ACCUM"]))
            self.freeze_ratio_var.set(str(preset["FREEZE_BACKBONE_RATIO"]))
            self.test_every_var.set(str(preset["TEST_EVERY"]))
            self.aug_level_var.set(str(preset["AUG_LEVEL"]))
            self.read_backend_var.set(str(preset["READ_BACKEND"]))
            self.pretrained_var.set(bool(preset["PRETRAINED"]))
            self.amp_var.set(bool(preset["USE_AMP"]))
            self.sync_out_dir()
            self.summary_var.set(
                "已应用小样本推荐参数："
                + (" MobileNetV3 更偏稳健部署。" if "mobilenet" in normalized else " Swin 更偏精度对照。")
            )
        except Exception as exc:
            self.append_log(f"[WARN] 无法应用 backbone 预设: {exc}\n")

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            Messagebox.show_info("当前已有训练任务在运行，请等待完成。", "训练中", parent=self.root)
            return

        try:
            train_cfg = self.build_runtime_config()
        except Exception as exc:
            Messagebox.show_error(str(exc), "参数错误", parent=self.root)
            return

        self.reset_status()
        self.append_log("[INFO] 正在准备训练任务...\n")
        self.training_thread = threading.Thread(target=self.run_training, args=(train_cfg,), daemon=True)
        self.training_thread.start()

    def build_runtime_config(self):
        class RuntimeConfig:
            pass

        runtime = RuntimeConfig()
        runtime.MODEL_NAME = normalize_backbone_name(self.backbone_var.get())
        runtime.DATA_DIR = self.data_dir_var.get().strip()
        runtime.OUT_DIR = self.out_dir_var.get().strip()
        runtime.BATCH_SIZE = int(self.batch_size_var.get().strip())
        runtime.EPOCHS = int(self.epochs_var.get().strip())
        runtime.IMG_SIZE = int(self.img_size_var.get().strip())
        runtime.LR = float(self.lr_var.get().strip())
        runtime.WEIGHT_DECAY = float(self.weight_decay_var.get().strip())
        runtime.WORKERS = int(self.workers_var.get().strip())
        runtime.GRAD_ACCUM = int(self.grad_accum_var.get().strip())
        runtime.FREEZE_BACKBONE_RATIO = float(self.freeze_ratio_var.get().strip())
        runtime.TEST_EVERY = int(self.test_every_var.get().strip())
        runtime.AUG_LEVEL = self.aug_level_var.get().strip()
        runtime.READ_BACKEND = self.read_backend_var.get().strip()
        runtime.PRETRAINED = bool(self.pretrained_var.get())
        runtime.USE_AMP = bool(self.amp_var.get())
        runtime.USE_CPU = bool(self.cpu_var.get())
        runtime.PRETRAINED_SOURCE = getattr(cfg, "PRETRAINED_SOURCE", "auto")
        runtime.BRAND_LOSS_WEIGHT = getattr(cfg, "BRAND_LOSS_WEIGHT", 0.3)
        runtime.MODEL_LOSS_WEIGHT = getattr(cfg, "MODEL_LOSS_WEIGHT", 0.7)
        runtime.MAX_GRAD_NORM = getattr(cfg, "MAX_GRAD_NORM", 1.0)
        runtime.USE_TQDM = False
        runtime.PROGRESS_CALLBACK = self.progress_callback

        if not runtime.DATA_DIR:
            raise ValueError("数据目录不能为空")
        if not Path(runtime.DATA_DIR).exists():
            raise ValueError(f"数据目录不存在: {runtime.DATA_DIR}")
        if runtime.BATCH_SIZE <= 0 or runtime.EPOCHS <= 0 or runtime.IMG_SIZE <= 0:
            raise ValueError("Batch / Epochs / Image Size 必须大于 0")
        if runtime.WORKERS < 0 or runtime.GRAD_ACCUM <= 0 or runtime.TEST_EVERY <= 0:
            raise ValueError("Workers 不能小于 0，Grad Accum 与 Test Every 必须大于 0")
        if runtime.FREEZE_BACKBONE_RATIO < 0 or runtime.FREEZE_BACKBONE_RATIO > 1:
            raise ValueError("Freeze Ratio 必须在 0 到 1 之间")
        return runtime

    def run_training(self, train_cfg):
        try:
            self.log_queue.put(("log", f"[INFO] backbone={train_cfg.MODEL_NAME}\n"))
            self.log_queue.put(("log", f"[INFO] data_dir={train_cfg.DATA_DIR}\n"))
            self.log_queue.put(("log", f"[INFO] out_dir={train_cfg.OUT_DIR}\n"))
            train_ds, val_ds, test_ds = build_datasets(
                data_dir=train_cfg.DATA_DIR,
                img_size=train_cfg.IMG_SIZE,
                aug_level=train_cfg.AUG_LEVEL,
            )
            train(train_cfg, train_ds, val_ds, test_ds)
        except Exception as exc:
            self.log_queue.put(("error", f"[ERROR] {exc}\n"))
            self.log_queue.put(("status", "训练失败"))

    def progress_callback(self, payload: dict):
        self.log_queue.put(("progress", payload))

    def reset_status(self):
        self.status_var.set("训练准备中")
        self.progress_var.set(0.0)
        self.best_acc_var.set("—")
        self.summary_var.set("正在初始化...")
        self.log_text.text.configure(state="normal")
        self.log_text.text.delete("1.0", "end")
        self.log_text.text.configure(state="disabled")

    def append_log(self, text: str):
        self.log_text.text.configure(state="normal")
        self.log_text.text.insert("end", text)
        self.log_text.text.see("end")
        self.log_text.text.configure(state="disabled")

    def drain_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.append_log(str(payload))
                elif kind == "error":
                    self.append_log(str(payload))
                    self.status_var.set("训练失败")
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "progress":
                    self.handle_progress(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self.drain_queue)

    def _drain_queue(self):
        self.drain_queue()

    def handle_progress(self, payload: dict):
        stage = payload.get("stage")
        if stage == "init":
            self.status_var.set("训练初始化完成")
            self.out_dir_runtime_var.set(str(payload.get("out_dir", "—")))
            self.summary_var.set(
                f"{payload.get('model_name')} | {payload.get('device')} | "
                f"epochs={payload.get('epochs')} | batch={payload.get('batch_size')}"
            )
            self.append_log(f"[INFO] {payload.get('message')}\n")
        elif stage == "epoch_end":
            epoch = int(payload.get("epoch", 0))
            epochs = int(payload.get("epochs", 1))
            self.progress_var.set((epoch / max(epochs, 1)) * 100)
            self.status_var.set(f"训练中：Epoch {epoch}/{epochs}")
            self.best_acc_var.set(f"{payload.get('best_model_acc', 0.0):.4f}")
            self.summary_var.set(
                f"train brand/model = {payload.get('train_brand_acc', 0.0):.4f} / "
                f"{payload.get('train_model_acc', 0.0):.4f} | "
                f"val brand/model = {payload.get('val_brand_acc', 0.0):.4f} / "
                f"{payload.get('val_model_acc', 0.0):.4f}"
            )
            self.append_log(
                f"[EPOCH {epoch}] "
                f"train_brand={payload.get('train_brand_acc', 0.0):.4f} "
                f"train_model={payload.get('train_model_acc', 0.0):.4f} "
                f"val_brand={payload.get('val_brand_acc', 0.0):.4f} "
                f"val_model={payload.get('val_model_acc', 0.0):.4f}\n"
            )
        elif stage == "done":
            self.progress_var.set(100.0)
            self.status_var.set("训练完成")
            self.best_acc_var.set(f"{payload.get('best_model_acc', 0.0):.4f}")
            self.out_dir_runtime_var.set(str(payload.get("out_dir", "—")))
            self.summary_var.set("训练分析产物已生成：曲线图 / 混淆矩阵 / 报告 / Grad-CAM / 更多可视化")
            self.append_log(f"[DONE] {payload.get('message')}\n")


def main():
    root = tb.Window(themename="darkly")
    app = TrainPanelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
