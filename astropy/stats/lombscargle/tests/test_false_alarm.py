import numpy as np
from numpy.testing import assert_allclose

from ....tests.helper import pytest

from .. import false_alarm
from ..false_alarm import false_alarm_probability, significance_level, METHODS, inverted
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


# @pytest.mark.parametrize('method', set(METHODS) - {'bootstrap'})
# @pytest.mark.parametrize('normalization', NORMALIZATIONS)
# @pytest.mark.parametrize('fmax', [5])
# def test_inverted_function(method, normalization, fmax, data):
#     kwds = METHOD_KWDS.get(method, None)
#
#     # shuffle the data to find useful range of Z
#     t, y, dy = data
#     rng = np.random.RandomState(0)
#     i = np.arange(len(t))
#     rng.shuffle(i)
#     y, dy = y[i], dy[i]
#     freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
#                                                   maximum_frequency=fmax)
#     Z = np.linspace(power.min(), power.max(), 30)
#
#     args = (fmax, t, y, dy)
#     kwargs = dict(normalization=normalization,
#                   method=method,
#                   method_kwds=METHOD_KWDS.get(method, None))
#     fap = false_alarm_probability(Z, *args, **kwargs)
#     Z_out = inverted(false_alarm_probability)(fap, *args, **kwargs)
#     assert_allclose(Z, Z_out, rtol=1E-8, atol=1E-8)


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

    # sig can't go above 0.95 because bootstrap limited to 20 samples
    sig = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    Z = significance_level(sig, fmax, t, y, dy,
                           normalization=normalization,
                           method=method,
                           method_kwds=METHOD_KWDS.get(method, None))
    assert len(Z) == len(sig)
    assert np.all(Z > 0)
    assert np.all(Z[:-1] <= Z[1:])  # monotonically increasing


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_log_fap_single(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_single(Z, len(t), normalization)
    logFAP = false_alarm.log_FAP_single(Z, len(t), normalization)

    assert_allclose(FAP, np.exp(logFAP))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_log_P_single(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    P = false_alarm.P_single(Z, len(t), normalization)
    logP = false_alarm.log_P_single(Z, len(t), normalization)

    assert_allclose(P, np.exp(logP))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_log_tau_davies(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    tau = false_alarm.tau_davies(Z, fmax, t, y, dy, normalization)
    log_tau = false_alarm.log_tau_davies(Z, fmax, t, y, dy, normalization)

    assert_allclose(tau, np.exp(log_tau))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_log_FAP_simple(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_simple(Z, fmax, t, y, dy, normalization)
    logFAP = false_alarm.log_FAP_simple(Z, fmax, t, y, dy, normalization)

    assert_allclose(FAP, np.exp(logFAP), atol=1E-14)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_log_FAP_davies(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_davies(Z, fmax, t, y, dy, normalization)
    logFAP = false_alarm.log_FAP_davies(Z, fmax, t, y, dy, normalization)

    assert_allclose(FAP, np.exp(logFAP), atol=1E-14)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
@pytest.mark.parametrize('fmax', [5])
def test_log_FAP_baluev(method, normalization, fmax, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_baluev(Z, fmax, t, y, dy, normalization)
    logFAP = false_alarm.log_FAP_baluev(Z, fmax, t, y, dy, normalization)

    assert_allclose(FAP, np.exp(logFAP), atol=1E-14)


# @pytest.mark.parametrize('method', METHODS)
# @pytest.mark.parametrize('normalization', NORMALIZATIONS)
# @pytest.mark.parametrize('fmax', [5])
# def test_false_alarm_roundtrip(method, normalization, fmax, data):
#     # sig can't go above 0.95 because bootstrap limited to 20 samples
#     t, y, dy = data
#     sig = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
#     Z = significance_level(sig, fmax, t, y, dy,
#                            normalization=normalization,
#                            method=method,
#                            method_kwds=METHOD_KWDS.get(method, None))
#     fap = false_alarm_probability(Z, fmax, t, y, dy,
#                                   normalization=normalization,
#                                   method=method,
#                                   method_kwds=METHOD_KWDS.get(method, None))
#     assert_allclose(1 - sig, fap)
