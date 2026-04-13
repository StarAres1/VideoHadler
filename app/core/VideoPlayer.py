import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
from PyQt6.QtCore import QTimer, Qt, QObject, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel

from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.FrameProcessor import FrameProcessor


class VideoPlayer(QObject):
    position_changed = pyqtSignal(int)
    time_changed = pyqtSignal(str)
    playback_state_changed = pyqtSignal(bool)
    file_opened = pyqtSignal(str)
    show_fps = pyqtSignal(float)
    frame_ready = pyqtSignal(object, bool)

    def __init__(self, label_record_format: QLabel):
        super().__init__()
        self.label_record_format = label_record_format
        self.cap = None
        self.timer = None
        self.current_frame = None
        self.raw_frame = None
        self.current_file = ""
        self.duration_ms = 0.0
        self._is_playing = False

        self.width = 0
        self.height = 0
        self.prop = 1.0

        self.method_for_contrast = ContrastImprovement.NotImprove
        self.method_for_noise = NoiseReduction.NotReduction
        self.processor = FrameProcessor()
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = 1
        self.roi_height = 1
        self.show_roi_content = False

        self._fps_frame_acc = 0
        self._fps_window_start = time.perf_counter()

        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._render_future = None
        self.frame_ready.connect(self._set_rendered_frame)

    @pyqtSlot(str)
    def play(self, file_path: str):
        self.stop()
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print(f"Не удалось открыть видео: {file_path}")
            return

        self.current_file = file_path
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration_ms = (frame_count / fps) * 1000.0 if fps > 0 and frame_count > 0 else 0.0

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.prop = self.width / self.height
            self.roi_x = 0
            self.roi_y = 0
            self.roi_width = self.width
            self.roi_height = self.height
        self.show_roi_content = False

        interval = max(1, int(round(1000.0 / fps))) if fps > 0 else 33
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(interval)
        self._is_playing = True
        self.playback_state_changed.emit(True)
        self._fps_window_start = time.perf_counter()
        self._fps_frame_acc = 0

        self.file_opened.emit(file_path)

    def _snapshot_params(self):
        return {
            "width": self.width,
            "height": self.height,
            "roi_x": self.roi_x,
            "roi_y": self.roi_y,
            "roi_w": self.roi_width,
            "roi_h": self.roi_height,
            "noise": self.method_for_noise,
            "contrast": self.method_for_contrast,
            "show_roi_content": self.show_roi_content,
        }

    def _process_frame(self, frame_bgr, p):
        if frame_bgr is None:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        width = max(1, int(p["width"]))
        height = max(1, int(p["height"]))
        roi_x = max(0, min(int(p["roi_x"]), max(0, width - 1)))
        roi_y = max(0, min(int(p["roi_y"]), max(0, height - 1)))
        roi_w = max(1, min(int(p["roi_w"]), max(1, width - roi_x)))
        roi_h = max(1, min(int(p["roi_h"]), max(1, height - roi_y)))
        if p.get("show_roi_content"):
            roi_frame = frame_rgb[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            if roi_frame.size == 0:
                return None
            rw, rh = roi_frame.shape[1], roi_frame.shape[0]
            return self.processor.process(
                roi_frame,
                rw,
                rh,
                0,
                0,
                rw,
                rh,
                p["contrast"],
                p["noise"],
            )
        return self.processor.process(
            frame_rgb,
            width,
            height,
            0,
            0,
            width,
            height,
            p["contrast"],
            p["noise"],
        )

    @pyqtSlot(object, bool)
    def _set_rendered_frame(self, frame_rgb, count_for_playback_fps):
        if frame_rgb is None:
            return
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.label_record_format.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label_record_format.setPixmap(scaled)
        self.current_frame = frame_rgb.copy()
        if count_for_playback_fps:
            self._fps_frame_acc += 1
            now = time.perf_counter()
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self.show_fps.emit(self._fps_frame_acc / elapsed)
                self._fps_window_start = now
                self._fps_frame_acc = 0

    def _render_current_frame_async(self, count_for_playback_fps: bool = False):
        if self.raw_frame is None:
            return
        if self._render_future and not self._render_future.done():
            return
        frame = self.raw_frame.copy()
        params = self._snapshot_params()

        def _task():
            return self._process_frame(frame, params)

        self._render_future = self._render_executor.submit(_task)

        def _done(fut):
            try:
                out = fut.result()
            except Exception:
                out = None
            self.frame_ready.emit(out, count_for_playback_fps)

        self._render_future.add_done_callback(_done)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.pause()
            return
        self.raw_frame = frame.copy()
        self._render_current_frame_async(count_for_playback_fps=True)
        self._emit_progress()

    def _emit_progress(self):
        if not self.cap:
            return
        current_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        percent = int((current_ms / self.duration_ms) * 100) if self.duration_ms > 0 else 0
        percent = max(0, min(100, percent))
        self.position_changed.emit(percent)
        self.time_changed.emit(self._format_time(current_ms))

    def _format_time(self, ms: float) -> str:
        total_seconds = int(ms // 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def is_loaded(self) -> bool:
        return self.cap is not None

    def is_playing(self) -> bool:
        return self._is_playing

    def toggle_play_pause(self):
        if not self.cap:
            return
        if self._is_playing:
            self.pause()
        else:
            self.resume()

    def pause(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
        self._is_playing = False
        self.playback_state_changed.emit(False)

    def resume(self):
        if self.timer and not self.timer.isActive():
            self.timer.start()
        self._is_playing = True
        self._fps_frame_acc = 0
        self._fps_window_start = time.perf_counter()
        self.playback_state_changed.emit(True)

    def seek_seconds(self, seconds: int):
        if not self.cap:
            return
        current_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        target_ms = max(0.0, current_ms + seconds * 1000.0)
        if self.duration_ms > 0:
            target_ms = min(target_ms, self.duration_ms)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ret, frame = self.cap.read()
        if ret:
            self.raw_frame = frame.copy()
            self._render_current_frame_async()
        self._emit_progress()

    def set_position_percent(self, percent: int):
        if not self.cap:
            return
        percent = max(0, min(100, percent))
        target_ms = self.duration_ms * (percent / 100.0)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ret, frame = self.cap.read()
        if ret:
            self.raw_frame = frame.copy()
            self._render_current_frame_async()
        self._emit_progress()

    def refresh_current_frame(self):
        self._render_current_frame_async()

    def set_method_for_contrast(self, method): self.method_for_contrast = method; self.refresh_current_frame()
    def set_method_for_noise(self, method): self.method_for_noise = method; self.refresh_current_frame()
    def set_clipLimit_CLAHE(self, value): self.processor.config.clip_limit = float(value); self.refresh_current_frame()
    def set_titleGridSize_CLAHE(self, value): self.processor.config.tile_grid_size = int(value); self.refresh_current_frame()
    def set_alpha_adjust(self, value): self.processor.config.alpha = float(value); self.refresh_current_frame()
    def set_beta_adjust(self, value): self.processor.config.beta = int(value); self.refresh_current_frame()
    def set_he_color(self, value): self.processor.config.he_color = bool(value); self.refresh_current_frame()
    def set_gamma_value(self, value): self.processor.config.gamma = float(value); self.refresh_current_frame()
    def set_sigmoid_cutoff(self, value): self.processor.config.sigmoid_cutoff = float(value); self.refresh_current_frame()
    def set_sigmoid_gain(self, value): self.processor.config.sigmoid_gain = float(value); self.refresh_current_frame()
    def set_auto_gamma_target_brightness(self, value): self.processor.config.auto_gamma_target_brightness = int(value); self.refresh_current_frame()
    def set_auto_gamma_color(self, value): self.processor.config.auto_gamma_color = bool(value); self.refresh_current_frame()
    def set_nn_skip_frames(self, value): self.processor.config.nn_skip_frames = int(value); self.refresh_current_frame()
    def set_median_ksize(self, value): self.processor.config.median_ksize = int(value); self.refresh_current_frame()
    def set_fast_gaussian_ksize(self, value): self.processor.config.fast_gaussian_ksize = int(value); self.refresh_current_frame()
    def set_fast_gaussian_sigma(self, value): self.processor.config.fast_gaussian_sigma = float(value); self.refresh_current_frame()
    def set_monochrome(self, value): self.processor.config.monochrome = bool(value); self.refresh_current_frame()
    def set_roi_x(self, value): self.roi_x = int(value); self.refresh_current_frame()
    def set_roi_y(self, value): self.roi_y = int(value); self.refresh_current_frame()
    def set_roi_width(self, value): self.roi_width = int(value); self.refresh_current_frame()
    def set_roi_height(self, value): self.roi_height = int(value); self.refresh_current_frame()

    def save_screenshot(self, output_dir: str = "screenshots") -> str | None:
        if self.current_frame is None:
            return None
        os.makedirs(output_dir, exist_ok=True)
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(output_dir, filename)
        ok = cv2.imwrite(filepath, self.current_frame)
        return filepath if ok else None

    def stop(self):
        if self.timer:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.raw_frame = None
        self.duration_ms = 0.0
        self.current_file = ""
        self._is_playing = False
        self._fps_frame_acc = 0
        self._fps_window_start = time.perf_counter()
        self.playback_state_changed.emit(False)
        self.position_changed.emit(0)
        self.time_changed.emit("0:00:00")
        self.show_roi_content = False
        if self.label_record_format:
            self.label_record_format.clear()
