import numpy as np
from numpy.testing import assert_allclose

from ....tests.helper import pytest

from .. import false_alarm
from ..false_alarm import false_alarm_probability, METHODS
from ..utils import convert_normalization, compute_chi2_ref
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


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_log_fap_single(normalization, data):
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_single(Z, len(t), normalization)
    logFAP = false_alarm.log_FAP_single(Z, len(t), normalization)

    assert_allclose(FAP, np.exp(logFAP))


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_log_P_single(normalization, data):
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)

    P = false_alarm.P_single(Z, len(t), normalization)
    logP = false_alarm.log_P_single(Z, len(t), normalization)

    assert_allclose(P, np.exp(logP))


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_log_tau_davies(normalization, data):
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)

    tau = false_alarm.tau_davies(Z, fmax, t, y, dy, normalization)
    log_tau = false_alarm.log_tau_davies(Z, fmax, t, y, dy, normalization)

    assert_allclose(tau, np.exp(log_tau))


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_log_FAP_simple(normalization, data):
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_simple(Z, fmax, t, y, dy, normalization)
    logFAP = false_alarm.log_FAP_simple(Z, fmax, t, y, dy, normalization)

    assert_allclose(FAP, np.exp(logFAP), atol=1E-14)


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_log_FAP_davies(normalization, data):
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_davies(Z, fmax, t, y, dy, normalization)
    logFAP = false_alarm.log_FAP_davies(Z, fmax, t, y, dy, normalization)

    assert_allclose(FAP, np.exp(logFAP), atol=1E-14)


@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_log_FAP_baluev(normalization, data):
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)

    FAP = false_alarm.FAP_baluev(Z, fmax, t, y, dy, normalization)
    logFAP = false_alarm.log_FAP_baluev(Z, fmax, t, y, dy, normalization)

    assert_allclose(FAP, np.exp(logFAP), atol=1E-14)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('normalization', NORMALIZATIONS)
def test_false_alarm_smoketest(method, normalization, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
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
def test_false_alarm_equivalence(method, normalization, data):
    kwds = METHOD_KWDS.get(method, None)
    t, y, dy = data
    fmax = 5

    freq, power = LombScargle(t, y, dy).autopower(normalization=normalization,
                                                  maximum_frequency=fmax)
    Z = np.linspace(power.min(), power.max(), 30)
    fap = false_alarm_probability(Z, fmax, t, y, dy,
                                  normalization=normalization,
                                  method=method,
                                  method_kwds=METHOD_KWDS.get(method, None))

    # Compute the equivalent Z values in the standard normalization
    # and check that the FAP is consistent
    Z_std = convert_normalization(Z, len(t),
                                  from_normalization=normalization,
                                  to_normalization='standard',
                                  chi2_ref=compute_chi2_ref(y, dy))
    fap_std = false_alarm_probability(Z_std, fmax, t, y, dy,
                                      normalization='standard',
                                      method=method,
                                      method_kwds=METHOD_KWDS.get(method, None))

    assert_allclose(fap, fap_std, rtol=0.1)
