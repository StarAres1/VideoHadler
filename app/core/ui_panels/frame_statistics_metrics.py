from __future__ import annotations

import logging

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)

FRAME_METRICS = (
    ("RMS contrast", "rms", 1),
    ("Histogram spread", "hist_spread", 1),
    ("EME contrast", "eme", 1),
    ("BRISQUE", "brisque_rgb", -1),
    ("NIQE", "niqe_rgb", -1),
    ("PIQE", "piqe_rgb", -1),
)


def pct_gain(before: float, after: float) -> float:
    return float(((after - before) / (abs(before) + 1e-12)) * 100.0)


def extract_l_channel_norm(frame_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    return lab[..., 0].astype(np.float32) / 255.0


def rms_contrast_l(image_hw: np.ndarray) -> float:
    return float(np.sqrt(np.mean((image_hw - np.mean(image_hw)) ** 2)))


def histogram_spread_l(image_hw: np.ndarray) -> float:
    image_u8 = np.clip(image_hw * 255.0, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([image_u8], [0], None, [256], [0, 256]).ravel()
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1] + 1e-12
    q1 = np.searchsorted(cdf, 0.25)
    q3 = np.searchsorted(cdf, 0.75)
    return float((q3 - q1) / 255.0)


def eme_contrast_l(image_hw: np.ndarray, block_size: int = 8) -> float:
    image_u8 = np.clip(image_hw * 255.0, 0, 255).astype(np.uint8)
    h, w = image_u8.shape
    nbh = max(1, h // block_size)
    nbw = max(1, w // block_size)
    if nbh == 0 or nbw == 0:
        return 0.0
    total = 0.0
    count = 0
    eps = 1e-6
    for by in range(nbh):
        for bx in range(nbw):
            y0 = by * block_size
            x0 = bx * block_size
            y1 = min(h, y0 + block_size)
            x1 = min(w, x0 + block_size)
            block = image_u8[y0:y1, x0:x1]
            if block.size == 0:
                continue
            i_min = float(np.min(block))
            i_max = float(np.max(block))
            total += 20.0 * np.log10((i_max + eps) / (i_min + eps))
            count += 1
    return float(total / max(1, count))


class IqaMetricsCalculator:
    def __init__(self) -> None:
        self._metrics_cache: dict[str, object | None] | None = None

    def _get_iqa_metrics(self):
        if self._metrics_cache is not None:
            return self._metrics_cache
        self._metrics_cache = {}
        try:
            import pyiqa
        except Exception:
            return self._metrics_cache
        for key in ("brisque", "niqe", "piqe"):
            try:
                self._metrics_cache[key] = pyiqa.create_metric(key, device="cpu")
            except Exception:
                self._metrics_cache[key] = None
        return self._metrics_cache

    def compute_iqa_rgb(self, image_hwc: np.ndarray) -> dict[str, float]:
        try:
            import torch
        except Exception:
            return {}
        if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
            return {}
        image = np.clip(image_hwc.astype(np.float32), 0.0, 1.0)
        tensor_rgb = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0)
        results: dict[str, float] = {}
        metrics = self._get_iqa_metrics()
        for key, metric in metrics.items():
            if metric is None:
                results[key] = float("nan")
                continue
            try:
                with torch.no_grad():
                    results[key] = float(metric(tensor_rgb).item())
            except Exception:
                results[key] = float("nan")
        return results


def metric_bundle_l(image_l: np.ndarray) -> dict:
    return {
        "rms": rms_contrast_l(image_l),
        "hist_spread": histogram_spread_l(image_l),
        "eme": eme_contrast_l(image_l),
    }


def metric_bundle_rgb(image_rgb_hwc: np.ndarray, iqa_calc: IqaMetricsCalculator) -> dict:
    iqa = iqa_calc.compute_iqa_rgb(image_rgb_hwc)
    return {
        "brisque_rgb": float(iqa.get("brisque", float("nan"))),
        "niqe_rgb": float(iqa.get("niqe", float("nan"))),
        "piqe_rgb": float(iqa.get("piqe", float("nan"))),
    }


def _to_qpixmap(canvas_rgb: np.ndarray) -> QPixmap:
    h, w = canvas_rgb.shape[:2]
    qimg = QImage(canvas_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _draw_axes_and_labels(canvas: np.ndarray, left: int, top: int, right: int, bottom: int, y_max: float) -> None:
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    for x_val in (0, 64, 128, 192, 255):
        x = left + int((x_val / 255.0) * (right - left))
        cv2.line(canvas, (x, bottom), (x, bottom + 4), (0, 0, 0), 1)
        cv2.putText(canvas, str(x_val), (max(0, x - 12), bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - int(t * (bottom - top))
        cv2.line(canvas, (left - 4, y), (left, y), (0, 0, 0), 1)
        val = int(round(t * y_max))
        cv2.putText(canvas, str(val), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def histogram_pixmaps_l_pair(image_before_l: np.ndarray, image_after_l: np.ndarray, width: int = 460, height: int = 260, same_scale: bool = False) -> tuple[QPixmap, QPixmap]:
    hist_before = cv2.calcHist([np.clip(image_before_l * 255.0, 0, 255).astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    hist_after = cv2.calcHist([np.clip(image_after_l * 255.0, 0, 255).astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    y_max_pair_raw = float(max(np.max(hist_before), np.max(hist_after), 1.0))
    y_max_pair = y_max_pair_raw * 1.08
    left, top, right, bottom = 46, 12, width - 10, height - 42
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    def render(hist: np.ndarray) -> QPixmap:
        y_max = y_max_pair if same_scale else float(max(np.max(hist), 1.0)) * 1.08
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        _draw_axes_and_labels(canvas, left, top, right, bottom, y_max)
        for i, val in enumerate(hist):
            x = left + int((i / 255.0) * plot_w)
            y = bottom - int((float(val) / y_max) * plot_h)
            cv2.line(canvas, (x, bottom), (x, y), (0, 102, 255), 1)
        return _to_qpixmap(canvas)

    return render(hist_before), render(hist_after)


def cumulative_histogram_pixmaps_l_pair(image_before_l: np.ndarray, image_after_l: np.ndarray, width: int = 460, height: int = 260) -> tuple[QPixmap, QPixmap]:
    def cumulative(image_l: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([np.clip(image_l * 255.0, 0, 255).astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
        cdf = np.cumsum(hist)
        cdf = cdf / (cdf[-1] + 1e-12)
        return cdf

    cdf_before = cumulative(image_before_l)
    cdf_after = cumulative(image_after_l)
    left, top, right, bottom = 46, 12, width - 10, height - 42
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    def render(cdf: np.ndarray) -> QPixmap:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        _draw_axes_and_labels(canvas, left, top, right, bottom, 1.0)
        prev = None
        for i, val in enumerate(cdf):
            x = left + int((i / 255.0) * plot_w)
            y = bottom - int(float(val) * plot_h)
            if prev is not None:
                cv2.line(canvas, prev, (x, y), (0, 102, 255), 1)
            prev = (x, y)
        return _to_qpixmap(canvas)

    return render(cdf_before), render(cdf_after)


def frame_preview_pixmap(frame_rgb: np.ndarray, max_w: int = 320, max_h: int = 220) -> QPixmap:
    h, w = frame_rgb.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    qimg = QImage(small.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class FrameStatsDialog(QtWidgets.QDialog):
    def __init__(self, rows, applied_contrast_text, preview_before, preview_after, hist_before_l, hist_after_l, cum_hist_before_l, cum_hist_after_l, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Статистика по кадру")
        self.resize(1180, 900)
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(container)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        applied_box = QtWidgets.QGroupBox("Применённый метод улучшения контраста")
        applied_layout = QtWidgets.QVBoxLayout(applied_box)
        applied_label = QtWidgets.QLabel(applied_contrast_text)
        applied_label.setWordWrap(True)
        applied_layout.addWidget(applied_label)
        content_layout.addWidget(applied_box)

        previews_layout = QtWidgets.QHBoxLayout()
        for title, pix in (("Исходный кадр", preview_before), ("Улучшенный кадр", preview_after)):
            wrap = QtWidgets.QVBoxLayout()
            t = QtWidgets.QLabel(title)
            t.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            img = QtWidgets.QLabel(self)
            img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            img.setPixmap(pix)
            wrap.addWidget(t)
            wrap.addWidget(img)
            previews_layout.addLayout(wrap)
        content_layout.addLayout(previews_layout)

        table = QtWidgets.QTableWidget(len(rows), 3, self)
        self.table = table
        self.rows = rows
        table.setHorizontalHeaderLabels(["Кадр до обработки", "Кадр после обработки", "На сколько процентов изменилось значение"])
        table.setVerticalHeaderLabels([r["name"] for r in rows])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for idx, row in enumerate(rows):
            before = row["before"]
            after = row["after"]
            pct = row["pct"]
            table.setItem(idx, 0, QtWidgets.QTableWidgetItem(f"{before:.6f}" if np.isfinite(before) else "nan"))
            table.setItem(idx, 1, QtWidgets.QTableWidgetItem(f"{after:.6f}" if np.isfinite(after) else "nan"))
            table.setItem(idx, 2, QtWidgets.QTableWidgetItem(f"{pct:.2f}%" if np.isfinite(pct) else "nan"))
        table.setMinimumSize(980, 360)
        table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        content_layout.addWidget(table)

        self.checkbox_highlight = QtWidgets.QCheckBox("Отображать динамику", self)
        self.checkbox_highlight.toggled.connect(self._apply_row_coloring)
        self.checkbox_highlight.setChecked(True)
        content_layout.addWidget(self.checkbox_highlight)

        def _pair_row(title_left: str, pix_left: QPixmap, title_right: str, pix_right: QPixmap):
            row = QtWidgets.QHBoxLayout()
            for title, pix in ((title_left, pix_left), (title_right, pix_right)):
                box = QtWidgets.QVBoxLayout()
                t = QtWidgets.QLabel(title)
                t.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                img = QtWidgets.QLabel(self)
                img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                img.setPixmap(pix)
                box.addWidget(t)
                box.addWidget(img)
                row.addLayout(box)
            return row

        content_layout.addLayout(_pair_row("Гистограмма яркости L (кадр до)", hist_before_l, "Гистограмма яркости L (кадр после)", hist_after_l))
        content_layout.addLayout(_pair_row("Кумулятивная гистограмма L (кадр до)", cum_hist_before_l, "Кумулятивная гистограмма L (кадр после)", cum_hist_after_l))

    def _apply_row_coloring(self, enabled: bool):
        green = QtGui.QColor(220, 255, 220)
        red = QtGui.QColor(255, 220, 220)
        white = QtGui.QColor(255, 255, 255)
        for idx, row in enumerate(self.rows):
            direction = int(row.get("direction", 0))
            pct = float(row.get("pct", float("nan")))
            if not enabled or direction == 0 or not np.isfinite(pct):
                bg = white
            else:
                effective = pct * direction
                bg = green if effective > 0 else red if effective < 0 else white
            for col in range(self.table.columnCount()):
                item = self.table.item(idx, col)
                if item is not None:
                    item.setBackground(bg)


def show_frame_statistics_dialog(host) -> None:
    before_rgb, after_rgb, source = host._get_current_frame_pair_for_statistics()
    if before_rgb is None or after_rgb is None:
        host.statusBar().showMessage("Нет доступного кадра для анализа", 3000)
        return
    host._pause_preview_for_stats(source)
    before_l = extract_l_channel_norm(before_rgb)
    after_l = extract_l_channel_norm(after_rgb)
    before_rgb_norm = np.clip(before_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    after_rgb_norm = np.clip(after_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    before_metrics = metric_bundle_l(before_l)
    after_metrics = metric_bundle_l(after_l)
    before_metrics.update(metric_bundle_rgb(before_rgb_norm, host._iqa_calc))
    after_metrics.update(metric_bundle_rgb(after_rgb_norm, host._iqa_calc))
    rows = []
    for title, key, direction in FRAME_METRICS:
        v_before = float(before_metrics[key])
        v_after = float(after_metrics[key])
        rows.append({"name": title, "before": v_before, "after": v_after, "pct": pct_gain(v_before, v_after), "direction": direction})

    hist_before, hist_after = histogram_pixmaps_l_pair(before_l, after_l, same_scale=False)
    cum_hist_before, cum_hist_after = cumulative_histogram_pixmaps_l_pair(before_l, after_l)
    preview_before = frame_preview_pixmap(before_rgb)
    preview_after = frame_preview_pixmap(after_rgb)
    applied_contrast_text = host._build_applied_contrast_text(source)
    dlg = FrameStatsDialog(
        rows=rows,
        applied_contrast_text=applied_contrast_text,
        preview_before=preview_before,
        preview_after=preview_after,
        hist_before_l=hist_before,
        hist_after_l=hist_after,
        cum_hist_before_l=cum_hist_before,
        cum_hist_after_l=cum_hist_after,
        parent=host,
    )
    dlg.finished.connect(lambda _: host._resume_preview_after_stats())
    host._frame_stats_dialog = dlg
    dlg.show()
    logger.info("Оператор: открыта статистика по текущему кадру (%s)", source)
