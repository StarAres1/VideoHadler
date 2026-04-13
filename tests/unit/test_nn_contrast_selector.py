"""Unit tests: NNContrastSelector parsing and apply_label (no full model load)."""

import numpy as np
import pytest

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
