"""
Tests of the signal cleaning helpers on small synthetic series.
"""

import numpy as np
import pytest

from ddtw_ml.signal.cleaning import PreprocessingUtils


def test_interpolation_fills_the_gaps_and_both_ends():
    s = np.arange(30, dtype=float)
    s[[0, 1, 10, 11, 29]] = np.nan
    out = PreprocessingUtils.interpolate_data(s, method="linear")
    np.testing.assert_allclose(out[2:29], np.arange(2, 29))
    assert out[0] == out[1] == 2 and out[29] == 28


def test_interpolation_raises_when_too_few_samples_are_valid():
    s = np.full(100, np.nan)
    s[:19] = 1.0
    with pytest.raises(ValueError, match="19 valid samples of 100"):
        PreprocessingUtils.interpolate_data(s, method="linear")


def test_iqr_outlier_removal_checks_every_sample():
    s = np.array([1000.0, 10, 11, 12, 10, 11, 12, 10, 11, 1000, np.nan])
    out = PreprocessingUtils.iqr_outlier_removal(s, threshold=1.5)
    assert np.isnan(out[[0, 9, 10]]).all()
    np.testing.assert_array_equal(out[1:9], s[1:9])
    assert s[0] == 1000  # the input is not changed
