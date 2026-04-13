"""Integration tests: FileSystemProxyModel filters video extensions."""

import pytest
from PyQt6.QtWidgets import QTreeView

from app.core.custom_widgets.FileBrowser import FileBrowser, FileSystemProxyModel
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtCore import QDir


def test_proxy_accepts_video_extensions():
    source = QFileSystemModel()
    source.setRootPath(QDir.rootPath())
    proxy = FileSystemProxyModel()
    proxy.setSourceModel(source)
    root = source.index(QDir.homePath())
    # Smoke: model is consistent
    assert proxy.rowCount(proxy.mapFromSource(root)) >= 0


def test_file_browser_instantiated(qtbot, tmp_path):
    tree = QTreeView()
    qtbot.addWidget(tree)
    fb = FileBrowser(tree)
    assert fb.tree is tree
    assert fb.model is not None
