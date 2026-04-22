import logging
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _ResnetBlock:
    def __init__(self, nn, dim: int):
        self.module = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
        )


class _ResnetGenerator:
    def __init__(self, torch, nn, in_nc: int = 3, out_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [_ResnetBlock(nn, ngf * mult).module]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


class EnlightenGanEnhancer:
    def __init__(self):
        self._model = None
        self._torch = None
        self._device = None
        self._lock = threading.Lock()
        self._is_loading = False
        self.last_error = ""

    def _weights_path(self) -> Path:
        return Path(__file__).resolve().parent / "weights" / "enlightengan_generator.pth"

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_loading(self) -> bool:
        return self._is_loading

    def _select_device(self, torch):
        try:
            import torch_directml

            return torch_directml.device(), "DirectML"
        except Exception:
            if torch.cuda.is_available():
                return torch.device("cuda"), "CUDA"
            return torch.device("cpu"), "CPU"

    def ensure_loaded_with_progress(self, progress_cb=None) -> bool:
        with self._lock:
            if self._model is not None:
                if progress_cb:
                    progress_cb(100, "Модель EnlightenGAN уже загружена")
                return True
            if self._is_loading:
                return False
            self._is_loading = True
        try:
            if progress_cb:
                progress_cb(5, "Импорт библиотек EnlightenGAN...")
            import torch
            import torch.nn as nn
        except Exception as exc:
            self.last_error = f"EnlightenGAN: не удалось импортировать torch: {exc}"
            logger.error(self.last_error)
            self._is_loading = False
            return False

        model_path = self._weights_path()
        if not model_path.is_file():
            self.last_error = f"EnlightenGAN: файл весов не найден: {model_path}"
            logger.error(self.last_error)
            self._is_loading = False
            return False
        try:
            if progress_cb:
                progress_cb(20, "Построение архитектуры EnlightenGAN...")
            model = _ResnetGenerator(torch, nn, in_nc=3, out_nc=3, ngf=64, n_blocks=9).model
            if progress_cb:
                progress_cb(45, "Чтение весов из файла...")
            state = torch.load(str(model_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            if progress_cb:
                progress_cb(70, "Загрузка весов в модель...")
            model.load_state_dict(state, strict=False)
            model.eval()
            self._device, device_name = self._select_device(torch)
            model = model.to(self._device)
            self._torch = torch
            self._model = model
            self._is_loading = False
            if progress_cb:
                progress_cb(90, f"Устройство инференса: {device_name}")
                progress_cb(100, "EnlightenGAN готова к работе")
            logger.info("EnlightenGAN: модель загружена (%s)", device_name)
            return True
        except Exception as exc:
            self.last_error = f"EnlightenGAN: ошибка загрузки весов: {exc}"
            logger.exception(self.last_error)
            self._model = None
            self._is_loading = False
            return False

    def enhance(self, frame_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if not self.ensure_loaded_with_progress():
            return frame_rgb
        torch = self._torch
        if torch is None or self._model is None:
            return frame_rgb
        try:
            img = frame_rgb.astype(np.float32) / 255.0
            inp = (img * 2.0) - 1.0
            tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(self._device)
            with torch.no_grad():
                out = self._model(tensor)
            out_np = out.squeeze(0).detach().cpu().numpy()
            out_np = np.transpose(out_np, (1, 2, 0))
            out_np = ((out_np + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            alpha = float(np.clip(strength, 0.0, 1.0))
            if alpha < 1.0:
                out_np = cv2.addWeighted(frame_rgb, 1.0 - alpha, out_np, alpha, 0.0)
            return out_np
        except Exception as exc:
            logger.exception("EnlightenGAN: ошибка инференса: %s", exc)
            return frame_rgb


ENLIGHTENGAN_ENHANCER = EnlightenGanEnhancer()
