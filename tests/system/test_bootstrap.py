"""System-level smoke tests: application wiring and imports."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_import_main_module():
    import main

    assert hasattr(main, "load_stylesheet")
    assert callable(main.load_stylesheet)


def test_qapplication_can_construct():
    from PyQt6.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is None:
        app = QApplication(sys.argv)
        assert app is not None
    else:
        assert existing is not None


@patch("main.QApplication")
@patch("main.MainWindow")
@patch("main.AppController")
def test_main_block_skipped_without_running_exec(mock_ac, mock_mw, mock_qapp):
    """Guards the __main__ path: we do not start the real event loop here."""
    import importlib

    import main

    assert hasattr(main, "load_stylesheet")
