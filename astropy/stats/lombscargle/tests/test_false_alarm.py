import numpy as np

from ....tests.helper import pytest

from ..false_alarm import false_alarm_probability, significance_level, METHODS
from .. import LombScargle


METHOD_KWDS = dict(bootstrap={'n_bootstraps': 20, 'random_seed': 42})
NORMALIZATIONS = ['standard', 'model', 'log', 'psd']


@pytest.fixture
def data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 5 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)

    return t, y, dy


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_false_alarm_smoketest(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    fap = false_alarm_probability(Z, fmax, t, y, dy,
                                  normalization=normalization,
                                  method=method,
                                  method_kwds=METHOD_KWDS.get(method, None))
    assert len(fap) == len(Z)
    if method != 'davies':
        assert np.all(fap <= 1)
        assert np.all(fap[:-1] >= fap[1:])  # monotonically decreasing


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_significance_smoketest(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data
    fmax = 5

    # sig can't go above 0.9 because bootstrap limited to 10 samples
    sig = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    Z = significance_level(sig, fmax, t, y, dy,
                           normalization=normalization,
                           method=method,
                           method_kwds=METHOD_KWDS.get(method, None))
    assert len(Z) == len(sig)
    assert np.all(Z > 0)
    assert np.all(Z[:-1] <= Z[1:])  # monotonically increasing


# TODO: test inverted() round-trip on a couple functions

# TODO: test that significance & fap are consistent with each other
