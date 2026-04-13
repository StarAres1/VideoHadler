"""Unit tests: main.load_stylesheet."""

import pytest

import main


class TestLoadStylesheet:
    def test_negative_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.qss"
        assert main.load_stylesheet(str(path)) == ""

    def test_positive_reads_utf8(self, tmp_path):
        p = tmp_path / "s.qss"
        p.write_text("QWidget { color: red; }\n", encoding="utf-8")
        content = main.load_stylesheet(str(p))
        assert "QWidget" in content
        assert "red" in content
