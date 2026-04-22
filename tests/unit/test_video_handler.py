"""Unit tests: VideoHandler signal API (no camera loop)."""

from unittest.mock import MagicMock

import numpy as np

from app.core.VideoHandler import VideoHandler


def test_video_handler_has_expected_signals():
    cam = MagicMock()
    vh = VideoHandler(cam)
    assert hasattr(vh, "paint_frame")
    assert hasattr(vh, "open_file")
    assert hasattr(vh, "record_frame")
    assert hasattr(vh, "close_file")
    assert hasattr(vh, "show_fps")


def test_run_emits_close_file_in_finally_when_record_was_opened():
    cam = MagicMock()
    cam.name = "Cam"
    cam.flag_capture = True
    cam.flag_record = True
    cam.width = 4
    cam.height = 4
    cam.fps = 30.0
    cam.record_format = "avi"
    cam.channel = 3
    cam.show_roi_content = False
    cam.method_for_contrast = MagicMock()
    cam.method_for_noise = MagicMock()
    cam.roi_x = 0
    cam.roi_y = 0
    cam.roi_width = 4
    cam.roi_height = 4
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    cam.cap.read.return_value = (False, frame_bgr)
    cam.disconnect = MagicMock()
    vh = VideoHandler(cam)

    closed = []
    vh.close_file.connect(lambda: closed.append(True))

    def stop_capture(_seconds):
        cam.flag_capture = False

    import app.core.VideoHandler as vh_mod
    orig_sleep = vh_mod.time.sleep
    vh_mod.time.sleep = stop_capture
    try:
        vh.run()
    finally:
        vh_mod.time.sleep = orig_sleep

    assert closed
    cam.disconnect.assert_called_once()


def test_run_emits_record_frame_when_recording_enabled():
    cam = MagicMock()
    cam.name = "Cam"
    cam.flag_capture = True
    cam.flag_record = True
    cam.width = 4
    cam.height = 4
    cam.fps = 25.0
    cam.record_format = "avi"
    cam.channel = 3
    cam.show_roi_content = False
    cam.method_for_contrast = MagicMock()
    cam.method_for_noise = MagicMock()
    cam.roi_x = 0
    cam.roi_y = 0
    cam.roi_width = 4
    cam.roi_height = 4
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    cam.cap.read.return_value = (True, frame_bgr)
    cam.disconnect = MagicMock()
    vh = VideoHandler(cam)
    vh.current_time = 0.0
    vh.current_frame = 0.0

    def process_once(frame, *_args, **_kwargs):
        cam.flag_capture = False
        return frame

    vh.processor.process = process_once
    recorded = []
    vh.record_frame.connect(lambda _img: recorded.append(True))

    vh.run()

    assert recorded
    cam.disconnect.assert_called_once()
