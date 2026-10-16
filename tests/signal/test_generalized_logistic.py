"""
Tests of the generalized logistic normalisation on synthetic data.
"""

import numpy as np
import pytest

from ddtw_ml.signal.generalized_logistic import GeneralizedLogistic


@pytest.mark.parametrize("seed", range(3))
def test_initial_curve_passes_through_one_tenth_at_the_minimum_and_half_at_the_median(seed):
    data = np.random.default_rng(seed).normal(100, 10, 500)
    gl = GeneralizedLogistic(data)
    Q0, B0, M0, nu0 = gl.initialize_parameters()
    x_min, x_med, _ = gl.get_statistics()
    assert B0 > 0
    assert gl.generalized_logistic(x_med, Q0, B0, M0, nu0) == pytest.approx(0.5)
    assert gl.generalized_logistic(x_min, Q0, B0, M0, nu0) == pytest.approx(0.1)


def test_normalised_values_follow_the_ecdf():
    data = np.random.default_rng(0).normal(100, 10, 500)
    values = GeneralizedLogistic(data)()
    assert np.all((values > 0) & (values < 1))
    assert np.abs(np.sort(values) - np.arange(1, 501) / 500).max() < 0.05
