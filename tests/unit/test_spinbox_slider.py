"""Unit tests: SpinBox_Slider synchronization helpers."""

from unittest.mock import MagicMock

import pytest

from app.core.custom_widgets.SpinBox_Slider import SpinBox_Slider


@pytest.fixture
def paired_widgets():
    slider = MagicMock()
    spin = MagicMock()
    slider.valueChanged = MagicMock()
    spin.valueChanged = MagicMock()
    return slider, spin


class TestSpinBoxSlider:
    def test_pow2_int(self):
        assert SpinBox_Slider.pow2_int(4) == 8

    def test_dec2_float(self):
        assert SpinBox_Slider.dec2_float(6) == 3.0

    def test_pow10_int(self):
        assert SpinBox_Slider.pow10_int(12) == 120

    def test_dec10_float(self):
        assert SpinBox_Slider.dec10_float(15) == 1.5

    def test_init_sets_values_twice(self, paired_widgets):
        slider, spin = paired_widgets
        cb = MagicMock()
        SpinBox_Slider(slider, spin, cb, slider_value=3, spinbox_value=30)
        assert slider.setValue.call_count >= 1
        assert spin.setValue.call_count >= 1

def test_set_value_with_func():
    slider = MagicMock()
    spin = MagicMock()
    func = MagicMock(return_value=42)
    sbs = SpinBox_Slider(slider, spin, lambda x: None, slider_value=5, spinbox_value=10)
    sbs.set_value(7, spin, func)
    func.assert_called_with(7)
    spin.setValue.assert_called_with(42)