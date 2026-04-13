from dataclasses import dataclass

import cv2
import numpy as np

from app.core.ContrastImprover import ContrastImprover
from app.core.Enums import ContrastImprovement, NoiseReduction
from app.neural_network.NNContrastSelector import NN_SELECTOR
from app.core.QualityImprover import QualityImprover


@dataclass
class ProcessingConfig:
    clip_limit: float = 2.0
    tile_grid_size: int = 4
    alpha: float = 1.0
    beta: int = 0
    gamma: float = 1.5
    sigmoid_cutoff: float = 0.5
    sigmoid_gain: float = 12.0
    auto_gamma_target_brightness: int = 128
    auto_gamma_color: bool = True
    he_color: bool = True
    median_ksize: int = 3
    fast_gaussian_ksize: int = 3
    fast_gaussian_sigma: float = 1.0
    nn_skip_frames: int = 0
    monochrome: bool = False


class FrameProcessor:
    def __init__(self, config: ProcessingConfig | None = None):
        self.config = config or ProcessingConfig()
        self.nn_selector = NN_SELECTOR
        self._nn_skip_counter = 0
        self._nn_last_label = ""

    def process(
        self,
        frame_rgb: np.ndarray,
        width: int,
        height: int,
        roi_x: int,
        roi_y: int,
        roi_w: int,
        roi_h: int,
        contrast_method: ContrastImprovement,
        noise_method: NoiseReduction,
    ) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        roi_x = max(0, min(int(roi_x), max(0, width - 1)))
        roi_y = max(0, min(int(roi_y), max(0, height - 1)))
        roi_w = max(1, min(int(roi_w), max(1, width - roi_x)))
        roi_h = max(1, min(int(roi_h), max(1, height - roi_y)))

        roi_frame = frame_rgb[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        if roi_frame.size != 0:
            frame_rgb = cv2.resize(roi_frame, (width, height), interpolation=cv2.INTER_LINEAR)

        if noise_method == NoiseReduction.MedianBlur:
            frame_rgb = QualityImprover.medianBlur(frame_rgb, int(self.config.median_ksize))
        elif noise_method == NoiseReduction.FastGaussian:
            frame_rgb = QualityImprover.fast_gaussian(
                frame_rgb,
                int(self.config.fast_gaussian_ksize),
                float(self.config.fast_gaussian_sigma),
            )

        if contrast_method == ContrastImprovement.CLAHE:
            frame_rgb = ContrastImprover.CLAHE(
                frame_rgb,
                clipLimit=float(self.config.clip_limit),
                titleGridSizeX=int(self.config.tile_grid_size),
                titleGridSizeY=int(self.config.tile_grid_size),
            )
        elif contrast_method == ContrastImprovement.adjust_contrast:
            frame_rgb = ContrastImprover.adjust_contrast(
                frame_rgb, alpha=float(self.config.alpha), beta=int(self.config.beta)
            )
        elif contrast_method == ContrastImprovement.HE:
            frame_rgb = ContrastImprover.HE(frame_rgb)
        elif contrast_method == ContrastImprovement.gamma:
            frame_rgb = ContrastImprover.gamma_correction(frame_rgb, gamma=float(self.config.gamma))
        elif contrast_method == ContrastImprovement.autoGamma:
            frame_rgb = ContrastImprover.auto_gamma(
                frame_rgb,
                target_brightness=int(self.config.auto_gamma_target_brightness),
            )
        elif contrast_method == ContrastImprovement.sigmoid:
            frame_rgb = ContrastImprover.sigmoid_correction(
                frame_rgb,
                cutoff=float(self.config.sigmoid_cutoff),
                gain=float(self.config.sigmoid_gain),
            )
        elif contrast_method == ContrastImprovement.nn:
            if self._nn_skip_counter <= 0 or not self._nn_last_label:
                predicted = self.nn_selector.predict_label(frame_rgb)
                if predicted:
                    self._nn_last_label = predicted
                self._nn_skip_counter = max(0, int(self.config.nn_skip_frames))
            else:
                self._nn_skip_counter -= 1
            if self._nn_last_label:
                frame_rgb = self.nn_selector.apply_label(frame_rgb, self._nn_last_label)

        if self.config.monochrome:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return frame_rgb
