"""Unit tests: NNContrastSelector parsing/loading/predict/apply_label."""

import sys
from types import SimpleNamespace

import numpy as np

from app.neural_network.NNContrastSelector import NNContrastSelector


class TestParseMappingFile:
    def test_positive_parses_valid_lines(self, tmp_path):
        p = tmp_path / "map.txt"
        p.write_text("0 0 CLAHE_2_4_4\n  3   9  gamma_1.0  \n", encoding="utf-8")
        sel = NNContrastSelector()
        m = sel._parse_mapping_file(str(p))
        assert m[0] == "CLAHE_2_4_4"
        assert m[3] == "gamma_1.0"

    def test_negative_missing_file(self, tmp_path):
        sel = NNContrastSelector()
        assert sel._parse_mapping_file(str(tmp_path / "nope.txt")) == {}

    def test_negative_malformed_lines_ignored(self, tmp_path):
        p = tmp_path / "map.txt"
        p.write_text("not a valid recording_separator\n0 0 ok_label\n", encoding="utf-8")
        sel = NNContrastSelector()
        m = sel._parse_mapping_file(str(p))
        assert m == {0: "ok_label"}


class TestPredictLabel:
    def test_negative_not_loaded_returns_none(self):
        sel = NNContrastSelector()
        sel.model = None
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        assert sel.predict_label(frame) is None

    def test_positive_predict_label_from_mock_torch(self):
        class DummyTorch:
            class no_grad:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            @staticmethod
            def argmax(_logits, dim=1):
                return SimpleNamespace(item=lambda: 2)

        class DummyTensor:
            def unsqueeze(self, _):
                return self

            def to(self, _):
                return self

        class DummyTransform:
            def __call__(self, _pil):
                return DummyTensor()

        class DummyImage:
            @staticmethod
            def fromarray(arr):
                return arr

        sel = NNContrastSelector()
        sel.model = lambda _tensor: "logits"
        sel.device = "cpu"
        sel.transform = DummyTransform()
        sel.class_mapping = {2: "gamma_1.2"}
        sel._pil_image_class = DummyImage

        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = DummyTorch
        try:
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            assert sel.predict_label(frame) == "gamma_1.2"
        finally:
            if old_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = old_torch

    def test_negative_predict_label_exception_updates_last_error(self):
        sel = NNContrastSelector()
        sel.model = object()
        sel.device = "cpu"
        sel.transform = None
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        assert sel.predict_label(frame) is None
        assert "Ошибка:" in sel.last_error


class TestLoadingState:
    def test_select_inference_device_fallback_cpu(self):
        sel = NNContrastSelector()
        import torch

        old_dml = sys.modules.get("torch_directml")
        if "torch_directml" in sys.modules:
            del sys.modules["torch_directml"]
        try:
            device, name = sel._select_inference_device(torch)
            assert str(device) == "cpu"
            assert name == "CPU"
        finally:
            if old_dml is not None:
                sys.modules["torch_directml"] = old_dml

    def test_ensure_loaded_returns_true_when_already_loaded(self):
        sel = NNContrastSelector()
        sel.model = object()
        calls = []
        assert sel.ensure_loaded_with_progress(lambda p, m: calls.append((p, m))) is True
        assert calls[-1][0] == 100

    def test_ensure_loaded_rejects_when_loading_in_progress(self):
        sel = NNContrastSelector()
        sel._is_loading = True
        assert sel.ensure_loaded_with_progress() is False

    def test_ensure_loaded_sets_error_if_model_file_missing(self):
        sel = NNContrastSelector()
        ok = sel.ensure_loaded_with_progress()
        assert ok is False
        assert "Файл модели не найден" in sel.last_error


class TestApplyLabel:
    def test_empty_label_returns_frame(self):
        sel = NNContrastSelector()
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        out = sel.apply_label(frame, "")
        assert out is frame

    def test_clahe_label_positive(self):
        sel = NNContrastSelector()
        frame = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "CLAHE_2.0_4_4")
        assert out.shape == frame.shape

    def test_gamma_label_positive(self):
        sel = NNContrastSelector()
        frame = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "gamma_1.5")
        assert out.shape == frame.shape

    def test_he_label_positive(self):
        sel = NNContrastSelector()
        frame = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "HE_")
        assert out.shape == frame.shape

    def test_unknown_label_returns_original(self):
        sel = NNContrastSelector()
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "totally_unknown_format")
        assert np.array_equal(out, frame)

    def test_adjust_contrast_label_positive(self):
        sel = NNContrastSelector()
        frame = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "adjust_contrast_1.7_10")
        assert out.shape == frame.shape

    def test_sigmoid_he_label_positive(self):
        sel = NNContrastSelector()
        frame = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "sigmoid+HE_0.5_8")
        assert out.shape == frame.shape

    def test_sigmoid_label_positive(self):
        sel = NNContrastSelector()
        frame = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        out = sel.apply_label(frame, "sigmoid_0.3_10")
        assert out.shape == frame.shape
