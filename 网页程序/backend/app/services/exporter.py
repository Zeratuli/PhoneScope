"""PDF 识别报告导出服务（美化版）。

版面设计亮点：
- 封面页：大标题 + 副标题 + 摘要统计卡 + 生成信息分隔条；
- 页眉页脚：每页自动显示"PhoneScope · 检测报告"与"第 n / N 页"；
- 单元格统一用 Paragraph 包装，启用 CJK 换行，避免长字符串裁剪；
- Top-K 采用条形图可视化；
- 规格参数改为 2 列卡片网格；
- 统一 ACCENT 主色、柔和中性灰背景条纹，打印友好。
"""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, HRFlowable, Image as RLImage,
    KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from app.schemas.models import FusionResult, ImageResult
from app.services.phone_data import PARAM_LABELS, get_phone_identity, get_phone_info


ACCENT        = colors.HexColor("#6366f1")
ACCENT_SOFT   = colors.HexColor("#e0e7ff")
ACCENT_DARK   = colors.HexColor("#4338ca")
SUCCESS       = colors.HexColor("#10b981")
WARN          = colors.HexColor("#f59e0b")
DANGER        = colors.HexColor("#ef4444")
BG_CARD       = colors.HexColor("#f8fafc")
BG_ROW_ALT    = colors.HexColor("#f1f5f9")
BORDER_SOFT   = colors.HexColor("#e2e8f0")
TEXT_MAIN     = colors.HexColor("#0f172a")
TEXT_MUTED    = colors.HexColor("#64748b")
WHITE         = colors.white


def _register_fonts() -> str:
    """尝试注册中文字体，返回最终使用的字体名。"""
    candidates = [
        ("MSYH", "msyh.ttc"),
        ("MSYH", r"C:\Windows\Fonts\msyh.ttc"),
        ("SimSun", "simsun.ttc"),
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for name, path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return "Helvetica"


class ConfidenceBar(Flowable):
    """Top-K 条形图单元素：左侧名称，中间进度条，右侧百分比。"""

    def __init__(self, name: str, confidence: float, is_top: bool,
                 font: str, width: float = 165 * mm):
        super().__init__()
        self.name = name
        self.confidence = max(0.0, min(1.0, confidence))
        self.is_top = is_top
        self.font = font
        self.total_width = width
        self.height = 12 * mm

    def wrap(self, *_):
        return self.total_width, self.height

    def draw(self):
        c = self.canv
        label_w = 55 * mm
        pct_w = 15 * mm
        bar_x = label_w
        bar_w = self.total_width - label_w - pct_w - 2 * mm
        bar_y = 2 * mm
        bar_h = 6 * mm

        c.setFont(self.font, 9)
        c.setFillColor(TEXT_MAIN if self.is_top else TEXT_MUTED)
        c.drawString(0, self.height / 2 - 3, self.name)

        c.setFillColor(BG_ROW_ALT)
        c.roundRect(bar_x, bar_y, bar_w, bar_h, 2, stroke=0, fill=1)

        fill_w = bar_w * self.confidence
        c.setFillColor(ACCENT if self.is_top else colors.HexColor("#cbd5e1"))
        if fill_w > 0:
            c.roundRect(bar_x, bar_y, fill_w, bar_h, 2, stroke=0, fill=1)

        c.setFont(self.font, 9)
        c.setFillColor(TEXT_MAIN if self.is_top else TEXT_MUTED)
        c.drawRightString(self.total_width, self.height / 2 - 3,
                          f"{self.confidence * 100:.1f}%")


class ExportService:
    """PDF 报告导出服务。"""

    def __init__(self):
        self._font = _register_fonts()

    # ---------------- style helpers ----------------

    def _style(self, name: str, parent: str = "Normal", **kw) -> ParagraphStyle:
        base = getSampleStyleSheet()
        return ParagraphStyle(name, parent=base[parent], fontName=self._font, **kw)

    def _cell(self, text: str, size: int = 8, color=TEXT_MAIN) -> Paragraph:
        """单元格 Paragraph，启用 CJK 换行。"""
        s = ParagraphStyle(
            "cell", fontName=self._font, fontSize=size, leading=size + 3,
            wordWrap="CJK", textColor=color,
        )
        return Paragraph(str(text) if text is not None else "—", s)

    def _display_model_name(self, model_name: str | None) -> str:
        if not model_name:
            return "—"
        identity = get_phone_identity(model_name)
        if identity and identity.get("display_name"):
            return str(identity["display_name"])
        return str(model_name).replace("_", " ")

    # ---------------- page decorations ----------------

    def _on_page(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        # header
        canvas.setStrokeColor(BORDER_SOFT)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, h - 12 * mm, w - 18 * mm, h - 12 * mm)
        canvas.setFont(self._font, 8)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawString(18 * mm, h - 10 * mm, "PhoneScope · 智能手机识别系统")
        canvas.drawRightString(w - 18 * mm, h - 10 * mm, "检测报告")

        # footer: 页码
        canvas.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
        canvas.setFont(self._font, 8)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawCentredString(
            w / 2, 9 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    # ---------------- sections ----------------

    def _build_cover(self, results: list[ImageResult]) -> list:
        s_main = self._style(
            "main", "Title", fontSize=28, textColor=ACCENT,
            alignment=TA_CENTER, spaceAfter=6, leading=32,
        )
        s_sub = self._style(
            "sub", "Normal", fontSize=11, textColor=TEXT_MUTED,
            alignment=TA_CENTER, spaceAfter=16,
        )
        s_meta = self._style(
            "meta", "Normal", fontSize=9, textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        )

        total_det = sum(len(r.detections) for r in results)
        total_ms = sum(r.processing_time_ms for r in results)
        hits = sum(1 for r in results if r.detections)

        stats = [
            ["图片总数", f"{len(results)}"],
            ["检测目标", f"{total_det}"],
            ["命中率", f"{hits}/{len(results)}" if results else "—"],
            ["总耗时", f"{total_ms:.0f} ms"],
        ]

        stats_flow = [self._stats_card(label, value) for label, value in stats]
        stats_row = Table([stats_flow], colWidths=[42 * mm] * 4,
                          hAlign="CENTER")
        stats_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))

        now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        return [
            Spacer(1, 30 * mm),
            Paragraph("PhoneScope", s_main),
            Paragraph("智能手机识别检测报告", s_sub),
            HRFlowable(width=70 * mm, thickness=1.2, color=ACCENT,
                       hAlign="CENTER", spaceAfter=22),
            stats_row,
            Spacer(1, 16 * mm),
            Paragraph(f"报告生成时间：{now}", s_meta),
            Paragraph("由 PhoneScope 自动生成", s_meta),
            PageBreak(),
        ]

    def _stats_card(self, label: str, value: str) -> Table:
        t = Table(
            [[Paragraph(value, self._style(
                "sv", "Normal", fontSize=18, textColor=ACCENT,
                alignment=TA_CENTER, leading=22))],
             [Paragraph(label, self._style(
                 "sl", "Normal", fontSize=8, textColor=TEXT_MUTED,
                 alignment=TA_CENTER, leading=10))]],
            colWidths=[42 * mm], rowHeights=[14 * mm, 6 * mm],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.6, BORDER_SOFT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return t

    def _conf_badge_color(self, conf: float):
        if conf > 0.85:
            return SUCCESS
        if conf > 0.70:
            return WARN
        return DANGER

    def _build_image_section(self, idx: int, result: ImageResult) -> list:
        """每张图片一大段。使用 KeepTogether 确保不被拆分（上限 A4）。"""
        section: list = []

        # section header
        section.append(self._section_header(idx, result.filename))

        # 元数据摘要表：两行四列
        meta_tbl = Table(
            [
                [self._cell("文件名", color=TEXT_MUTED),
                 self._cell(result.filename),
                 self._cell("尺寸", color=TEXT_MUTED),
                 self._cell(f"{result.width} × {result.height}")],
                [self._cell("检测目标", color=TEXT_MUTED),
                 self._cell(f"{len(result.detections)} 个"),
                 self._cell("处理耗时", color=TEXT_MUTED),
                 self._cell(f"{result.processing_time_ms:.1f} ms")],
            ],
            colWidths=[22 * mm, 65 * mm, 22 * mm, 65 * mm],
        )
        meta_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.4, BORDER_SOFT),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER_SOFT),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        section.append(meta_tbl)
        section.append(Spacer(1, 4 * mm))

        # 标注图
        if result.annotated_image_base64:
            try:
                img_data = base64.b64decode(result.annotated_image_base64)
                max_w = 174 * mm
                aspect = result.height / result.width if result.width else 1
                img_w = max_w
                img_h = img_w * aspect
                if img_h > 110 * mm:
                    img_h = 110 * mm
                    img_w = img_h / aspect
                section.append(RLImage(io.BytesIO(img_data),
                                       width=img_w, height=img_h,
                                       hAlign="CENTER"))
                section.append(Spacer(1, 4 * mm))
            except Exception:
                pass

        if not result.detections:
            section.append(Paragraph("未检测到手机目标",
                                     self._style("none", "Normal",
                                                 fontSize=10,
                                                 textColor=TEXT_MUTED,
                                                 alignment=TA_CENTER)))
            section.append(HRFlowable(width="100%", thickness=0.4,
                                      color=BORDER_SOFT, spaceAfter=6))
            return section

        # 检测结果表
        section.append(self._subheader("检测结果"))
        det_rows = [[
            self._cell("#", color=WHITE),
            self._cell("标签", color=WHITE),
            self._cell("置信度", color=WHITE),
            self._cell("识别型号", color=WHITE),
            self._cell("型号置信度", color=WHITE),
            self._cell("边框坐标", color=WHITE),
        ]]
        for i, (det, cls) in enumerate(
                zip(result.detections, result.classifications), 1):
            bbox = (f"({det.bbox.x1:.0f}, {det.bbox.y1:.0f}) - "
                    f"({det.bbox.x2:.0f}, {det.bbox.y2:.0f})")
            det_rows.append([
                self._cell(str(i)),
                self._cell(det.label),
                self._cell(f"{det.confidence:.1%}"),
                self._cell(self._display_model_name(cls.model_name)),
                self._cell(f"{cls.confidence:.1%}"),
                self._cell(bbox),
            ])
        det_tbl = Table(
            det_rows,
            colWidths=[8 * mm, 18 * mm, 20 * mm, 40 * mm, 24 * mm, 64 * mm],
            hAlign="LEFT", repeatRows=1,
        )
        det_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("FONTNAME", (0, 0), (-1, -1), self._font),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG_ROW_ALT]),
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER_SOFT),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        section.append(det_tbl)
        section.append(Spacer(1, 5 * mm))

        # 每个目标：Top-K 条形图 + 规格卡片网格
        for i, cls in enumerate(result.classifications, 1):
            section.append(self._subheader(f"目标 #{i}  型号识别详情"))

            badge = self._conf_badge(cls.model_name, cls.confidence)
            section.append(badge)
            section.append(Spacer(1, 3 * mm))

            # Top-K 条形图
            section.append(Paragraph(
                "Top-K 分类概率",
                self._style("topk_h", "Normal", fontSize=9,
                            textColor=TEXT_MUTED,
                            spaceAfter=3)))
            for rank, item in enumerate(cls.top_k, 1):
                section.append(ConfidenceBar(
                    f"Top-{rank}  {self._display_model_name(item.name)}",
                    item.confidence, is_top=(rank == 1), font=self._font,
                ))
            section.append(Spacer(1, 4 * mm))

            # 规格参数卡片网格
            phone_info = get_phone_info(cls.model_name)
            if phone_info:
                section.extend(self._build_spec_grid(phone_info))
                section.append(Spacer(1, 3 * mm))

        section.append(HRFlowable(width="100%", thickness=0.4,
                                  color=BORDER_SOFT, spaceAfter=6))
        return section

    def _section_header(self, idx: int, filename: str) -> Table:
        """左色条 + 图片序号 + 文件名。"""
        left = Paragraph(
            f"<b>图片 #{idx}</b>",
            self._style("sh_left", "Normal", fontSize=13, textColor=ACCENT))
        right = Paragraph(
            filename,
            self._style("sh_right", "Normal", fontSize=11,
                        textColor=TEXT_MUTED))
        t = Table([[left, right]], colWidths=[28 * mm, 146 * mm])
        t.setStyle(TableStyle([
            ("LINEBEFORE", (0, 0), (0, 0), 3, ACCENT),
            ("LEFTPADDING", (0, 0), (0, 0), 8),
            ("LEFTPADDING", (1, 0), (1, 0), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    def _subheader(self, text: str) -> Paragraph:
        return Paragraph(
            text,
            self._style("sh", "Normal", fontSize=10,
                        textColor=ACCENT_DARK,
                        spaceBefore=4, spaceAfter=4))

    def _conf_badge(self, model_name: str, confidence: float) -> Table:
        """大号模型名 + 彩色置信度徽章。"""
        badge_color = self._conf_badge_color(confidence)
        name_cell = Paragraph(
            f"<b>{self._display_model_name(model_name)}</b>",
            self._style("bn", "Normal", fontSize=14, textColor=TEXT_MAIN,
                        leading=18))
        pct_cell = Paragraph(
            f"<font color='white'><b>{confidence * 100:.1f}%</b></font>",
            self._style("bp", "Normal", fontSize=12, alignment=TA_CENTER))
        t = Table([[name_cell, pct_cell]], colWidths=[135 * mm, 35 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), badge_color),
            ("BACKGROUND", (0, 0), (0, 0), BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.4, BORDER_SOFT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (0, 0), 10),
        ]))
        return t

    def _build_spec_grid(self, phone_info: dict) -> list:
        """手机规格 2 列卡片网格。长字段（屏幕/相机/电池/系统）跨两列。"""
        short_keys = ["manufacturer", "brand", "model", "released",
                      "processor", "ram", "storage", "dimensions",
                      "weight", "colors"]
        wide_keys = ["screen", "rear_camera", "front_camera", "battery", "os"]

        out: list = []
        out.append(Paragraph(
            "<b>产品规格</b>",
            self._style("sg_h", "Normal", fontSize=9, textColor=TEXT_MUTED,
                        spaceAfter=3)))

        # 两列窄卡网格
        rows = []
        buf: list = []
        for key in short_keys:
            val = phone_info.get(key)
            if not val:
                continue
            buf.append(self._spec_cell(PARAM_LABELS.get(key, key), val))
            if len(buf) == 2:
                rows.append(buf)
                buf = []
        if buf:
            buf.append("")
            rows.append(buf)
        if rows:
            grid = Table(rows, colWidths=[85 * mm, 85 * mm])
            grid.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            out.append(grid)
            out.append(Spacer(1, 2 * mm))

        # 宽卡（跨整行）
        for key in wide_keys:
            val = phone_info.get(key)
            if not val:
                continue
            out.append(self._spec_cell(PARAM_LABELS.get(key, key), val,
                                       wide=True))
            out.append(Spacer(1, 1.5 * mm))

        return out

    def _spec_cell(self, label: str, value: str, wide: bool = False) -> Table:
        w = 170 * mm if wide else 85 * mm
        label_w = 26 * mm
        t = Table(
            [[self._cell(label, size=8, color=TEXT_MUTED),
              self._cell(value, size=9, color=TEXT_MAIN)]],
            colWidths=[label_w, w - label_w],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), BG_ROW_ALT),
            ("BOX", (0, 0), (-1, -1), 0.3, BORDER_SOFT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    # ---------------- fusion specific ----------------

    def _build_fusion_cover(self, fr: FusionResult) -> list:
        s_main = self._style(
            "fmain", "Title", fontSize=26, textColor=ACCENT,
            alignment=TA_CENTER, spaceAfter=4, leading=30,
        )
        s_sub = self._style(
            "fsub", "Normal", fontSize=11, textColor=TEXT_MUTED,
            alignment=TA_CENTER, spaceAfter=14,
        )
        s_meta = self._style(
            "fmeta", "Normal", fontSize=9, textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        )
        mode_label = {"images": "多图融合", "video": "视频抽帧融合"}.get(
            fr.mode, fr.mode)

        stats = [
            ["识别类型", mode_label],
            ["总帧数", f"{fr.total_frames}"],
            ["有效帧", f"{fr.valid_frames}"],
            ["处理耗时", f"{fr.processing_time_ms:.0f} ms"],
        ]
        stats_flow = [self._stats_card(k, v) for k, v in stats]
        stats_row = Table([stats_flow], colWidths=[42 * mm] * 4,
                          hAlign="CENTER")
        stats_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))

        now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        return [
            Spacer(1, 26 * mm),
            Paragraph("PhoneScope", s_main),
            Paragraph("多证据融合识别报告", s_sub),
            HRFlowable(width=70 * mm, thickness=1.2, color=ACCENT,
                       hAlign="CENTER", spaceAfter=20),
            stats_row,
            Spacer(1, 10 * mm),
            Paragraph(f"会话 ID：<font face='{self._font}'>{fr.session_id}</font>",
                      s_meta),
            Paragraph(f"报告生成时间：{now}", s_meta),
            PageBreak(),
        ]

    def _build_fusion_conclusion(self, fr: FusionResult) -> list:
        """第二节：最终识别结论（型号 + 置信度 + Top-K + 最优裁剪图）。"""
        out: list = []
        out.append(Paragraph(
            "<b>最终识别结论</b>",
            self._style("fcl_h", "Heading2", fontSize=14, textColor=ACCENT_DARK,
                        spaceBefore=2, spaceAfter=6)))

        if not fr.final_model_name:
            out.append(Paragraph(
                "未能从上传证据中识别出手机",
                self._style("fcl_n", "Normal", fontSize=10,
                            textColor=TEXT_MUTED)))
            return out

        out.append(self._conf_badge(
            fr.final_display_name or fr.final_model_name,
            fr.final_confidence or 0.0,
        ))
        out.append(Spacer(1, 4 * mm))

        # 最优裁剪图 + Top-K 并排
        has_crop = bool(fr.best_crop_base64)
        topk_flow: list = []
        topk_flow.append(Paragraph(
            "Top-K 分类概率",
            self._style("ft_h", "Normal", fontSize=9, textColor=TEXT_MUTED,
                        spaceAfter=2)))
        if fr.final_top_k:
            for rank, item in enumerate(fr.final_top_k, 1):
                topk_flow.append(ConfidenceBar(
                    f"Top-{rank}  {self._display_model_name(item.name)}",
                    item.confidence, is_top=(rank == 1), font=self._font,
                    width=95 * mm,
                ))

        if has_crop:
            try:
                img_data = base64.b64decode(fr.best_crop_base64 or "")
                img_w = 60 * mm
                img = RLImage(io.BytesIO(img_data),
                              width=img_w, height=img_w * 4 / 3)
                row = Table([[img, topk_flow]],
                            colWidths=[img_w + 4 * mm, 105 * mm])
                row.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]))
                out.append(row)
            except Exception:
                out.extend(topk_flow)
        else:
            out.extend(topk_flow)

        out.append(Spacer(1, 4 * mm))

        # 产品规格
        if fr.final_phone_spec:
            phone_info = fr.final_phone_spec.model_dump() if hasattr(
                fr.final_phone_spec, "model_dump") else dict(fr.final_phone_spec)
            out.extend(self._build_spec_grid(phone_info))

        out.append(Spacer(1, 3 * mm))
        out.append(HRFlowable(width="100%", thickness=0.4,
                              color=BORDER_SOFT, spaceAfter=6))
        return out

    def _build_fusion_frames(self, fr: FusionResult) -> list:
        """第三节：各帧证据（缩略图 + 元数据）。"""
        out: list = []
        out.append(Paragraph(
            f"<b>证据帧明细（共 {fr.total_frames} 帧）</b>",
            self._style("ff_h", "Heading2", fontSize=14, textColor=ACCENT_DARK,
                        spaceBefore=6, spaceAfter=4)))

        # 每帧一行：缩略图 + 元数据表
        for frame in fr.frames:
            is_best = (frame.frame_index == fr.best_frame_index)
            row_cells: list = []

            # 左：缩略图
            if frame.annotated_image_base64:
                try:
                    img_data = base64.b64decode(frame.annotated_image_base64)
                    thumb = RLImage(io.BytesIO(img_data),
                                    width=58 * mm, height=58 * mm * 9 / 16)
                    row_cells.append(thumb)
                except Exception:
                    row_cells.append(self._cell("—"))
            else:
                row_cells.append(self._cell("—"))

            # 右：帧编号 + 文件名 + 质量分 + 检测数 + 有效/最优标志
            badge_bits = []
            if is_best:
                badge_bits.append(
                    f"<font color='#10b981' face='{self._font}'><b>★ 最优帧</b></font>")
            if frame.is_valid:
                badge_bits.append(
                    f"<font color='#64748b' face='{self._font}'>有效检测</font>")
            else:
                badge_bits.append(
                    f"<font color='#ef4444' face='{self._font}'>无有效检测</font>")

            meta = [
                [self._cell("帧 #", size=8, color=TEXT_MUTED),
                 self._cell(str(frame.frame_index))],
                [self._cell("文件名", size=8, color=TEXT_MUTED),
                 self._cell(frame.filename)],
                [self._cell("质量分", size=8, color=TEXT_MUTED),
                 self._cell(f"{frame.quality_score:.4f}")],
                [self._cell("检测数", size=8, color=TEXT_MUTED),
                 self._cell(str(len(frame.detections)))],
                [self._cell("状态", size=8, color=TEXT_MUTED),
                 Paragraph(" · ".join(badge_bits),
                           self._style("fs", "Normal", fontSize=8,
                                       textColor=TEXT_MAIN, leading=10))],
            ]
            meta_tbl = Table(meta, colWidths=[18 * mm, 85 * mm])
            meta_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), BG_ROW_ALT),
                ("BOX", (0, 0), (-1, -1), 0.3, BORDER_SOFT),
                ("INNERGRID", (0, 0), (-1, -1), 0.2, BORDER_SOFT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]))
            row_cells.append(meta_tbl)

            row = Table([row_cells], colWidths=[62 * mm, 108 * mm])
            row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("BACKGROUND", (0, 0), (-1, -1),
                 colors.HexColor("#f5f3ff") if is_best else WHITE),
                ("BOX", (0, 0), (-1, -1), 0.5,
                 ACCENT if is_best else BORDER_SOFT),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            out.append(KeepTogether([row, Spacer(1, 3 * mm)]))

        return out

    # ---------------- public ----------------

    def _build_doc(self, title: str) -> tuple[BaseDocTemplate, io.BytesIO]:
        buf = io.BytesIO()
        doc = BaseDocTemplate(
            buf, pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=16 * mm, bottomMargin=18 * mm,
            title=title, author="PhoneScope",
        )
        frame = Frame(doc.leftMargin, doc.bottomMargin,
                      doc.width, doc.height, id="main")
        doc.addPageTemplates([
            PageTemplate(id="page", frames=[frame], onPage=self._on_page),
        ])
        return doc, buf

    def generate_pdf(
        self, results: list[ImageResult],
        title: str = "PhoneScope 检测报告",
    ) -> bytes:
        doc, buf = self._build_doc(title)
        story: list = []
        story.extend(self._build_cover(results))
        for idx, r in enumerate(results, 1):
            story.extend(self._build_image_section(idx, r))
        doc.build(story)
        return buf.getvalue()

    def generate_pdf_fusion(
        self, result: FusionResult,
        title: str = "PhoneScope 融合识别报告",
    ) -> bytes:
        doc, buf = self._build_doc(title)
        story: list = []
        story.extend(self._build_fusion_cover(result))
        story.extend(self._build_fusion_conclusion(result))
        story.extend(self._build_fusion_frames(result))
        doc.build(story)
        return buf.getvalue()
