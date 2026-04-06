import os
import re
from typing import Callable
import threading

import cv2
import numpy as np

from app.core.ContrastImprover import ContrastImprover


class NNContrastSelector:
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.class_mapping = {}
        self.last_error = ""
        self._is_loading = False
        self._lock = threading.Lock()

    def _build_model(self, num_classes, torch, nn, models):
        model = models.resnet18(weights=None)
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _parse_mapping_file(self, mapping_path: str):
        mapping = {}
        if not os.path.exists(mapping_path):
            return mapping
        pattern = re.compile(r"^\s*(\d+)\s+\d+\s+([A-Za-z0-9_+.\-]+)\s*$")
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    idx = int(m.group(1))
                    label = m.group(2).strip()
                    mapping[idx] = label
        return mapping

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_loading(self) -> bool:
        return self._is_loading

    def ensure_loaded_with_progress(self, progress_cb: Callable[[int, str], None] | None = None):
        with self._lock:
            if self.model is not None:
                if progress_cb:
                    progress_cb(100, "Модель уже загружена")
                return True
            if self._is_loading:
                return False
            self._is_loading = True
        try:
            if progress_cb:
                progress_cb(5, "Импорт библиотек нейросети...")
            import torch
            import torch.nn as nn
            from torchvision import transforms, models
            from PIL import Image
        except Exception as e:
            self.last_error = f"Не удалось импортировать зависимости нейросети: {e}"
            self._is_loading = False
            return False

        if progress_cb:
            progress_cb(20, "Подготовка файлов модели...")
        model_path = os.path.join("app", "neural_network", "change_cool", "saved_models", "best_model.pth")
        mapping_path = os.path.join("app", "neural_network", "change_cool", "logs", "class_mapping.txt")
        if not os.path.exists(model_path):
            self.last_error = f"Файл модели не найден: {model_path}"
            self._is_loading = False
            return False

        self.class_mapping = self._parse_mapping_file(mapping_path)
        num_classes = max(self.class_mapping.keys(), default=5) + 1

        try:
            if progress_cb:
                progress_cb(35, "Чтение весов из файла...")
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            if progress_cb:
                progress_cb(55, "Построение архитектуры модели...")
            self.model = self._build_model(num_classes, torch, nn, models)
            if progress_cb:
                progress_cb(75, "Загрузка весов в модель...")
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)

            if progress_cb:
                progress_cb(90, "Подготовка преобразований...")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            self._pil_image_class = Image
            if progress_cb:
                progress_cb(100, "Модель готова")
            self._is_loading = False
            return True
        except Exception as e:
            self.last_error = f"Ошибка загрузки модели: {e}"
            self.model = None
            self._is_loading = False
            return False

    def predict_label(self, frame_rgb: np.ndarray) -> str | None:
        if not self.is_loaded():
            return None
        try:
            import torch
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            pil = self._pil_image_class.fromarray(gray)
            tensor = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                idx = int(torch.argmax(logits, dim=1).item())
            return self.class_mapping.get(idx, "")
        except Exception as e:
            self.last_error = f"Ошибка инференса: {e}"
            return None

    def apply_label(self, frame_rgb: np.ndarray, label: str) -> np.ndarray:
        if not label:
            return frame_rgb
        if label.startswith("CLAHE_"):
            # CLAHE_3.0_4_4
            parts = label.split("_")
            if len(parts) == 4:
                return ContrastImprover.CLAHE(
                    frame_rgb,
                    clipLimit=float(parts[1]),
                    titleGridSizeX=int(parts[2]),
                    titleGridSizeY=int(parts[3]),
                )
        if label.startswith("adjust_contrast_"):
            # adjust_contrast_2.5_10
            parts = label.split("_")
            if len(parts) >= 4:
                return ContrastImprover.adjust_contrast(frame_rgb, alpha=float(parts[2]), beta=float(parts[3]))
        if label.startswith("gamma_"):
            # gamma_1.9
            parts = label.split("_")
            if len(parts) == 2:
                return ContrastImprover.gamma_correction(frame_rgb, gamma=float(parts[1]))
        if label.startswith("sigmoid+HE_"):
            # sigmoid+HE_0.3_12
            parts = label.split("_")
            if len(parts) == 3:
                frame = ContrastImprover.HE(frame_rgb)
                return ContrastImprover.sigmoid_correction(frame, cutoff=float(parts[1]), gain=float(parts[2]))
        if label.startswith("sigmoid_"):
            # sigmoid_0.3_12
            parts = label.split("_")
            if len(parts) == 3:
                return ContrastImprover.sigmoid_correction(frame_rgb, cutoff=float(parts[1]), gain=float(parts[2]))
        return frame_rgb


NN_SELECTOR = NNContrastSelector()
