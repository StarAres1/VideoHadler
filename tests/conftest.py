"""Shared fixtures for QtTrial tests."""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture
def sample_rgb_frame():
    """Small RGB image (H, W, 3) uint8."""
    import numpy as np

    return np.zeros((32, 48, 3), dtype=np.uint8)


@pytest.fixture
def video_file_mp4(tmp_path):
    """Minimal valid MP4 via OpenCV for VideoPlayer integration tests."""
    import cv2
    import numpy as np

    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = h = 64
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for _ in range(5):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 80, 120)
        out.write(frame)
    out.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("OpenCV VideoWriter did not produce a file on this system")
    return str(path)


def pytest_configure(config):
    # Headless / CI-friendly Qt when supported
    if os.environ.get("QT_QPA_PLATFORM") is None and sys.platform != "win32":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
