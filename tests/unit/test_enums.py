"""Unit tests: ContrastImprovement and NoiseReduction enums."""

import pytest

from app.core.Enums import ContrastImprovement, NoiseReduction


class TestContrastImprovement:
    def test_members_exist_and_unique(self):
        names = [m.name for m in ContrastImprovement]
        assert len(names) == len(set(names))

    def test_values_are_integers(self):
        for m in ContrastImprovement:
            assert isinstance(m.value, int)

    def test_lookup_by_name_positive(self):
        assert ContrastImprovement["CLAHE"] == ContrastImprovement.CLAHE

    def test_lookup_by_name_negative(self):
        with pytest.raises(KeyError):
            _ = ContrastImprovement["NonExistent"]


class TestNoiseReduction:
    def test_expected_members(self):
        assert NoiseReduction.NotReduction.value == 0
        assert NoiseReduction.MedianBlur.value == 3
        assert NoiseReduction.FastGaussian.value == 5

    def test_lookup_negative(self):
        with pytest.raises(KeyError):
            _ = NoiseReduction["Unknown"]
