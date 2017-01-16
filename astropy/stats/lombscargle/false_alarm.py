"""
False Alarm Probabilities for Lomb-Scargle Periodograms
=======================================================

This module implements several approaches to computing the false alarm
probabilities for Lomb-Scargle periodograms.
The three methods include:

- 'simple': estimates based on assumption of independent frequencies. This is
  fast but tends to under-estimate the false alarm probability.
- 'baluev': estimates based on extreme value statistics, following
  Baluev 2008. Results are analytic, and provide an upper-limit to the false
  alarm probability, but one which loses accuracy in the highly-aliased limit.
- 'bootstrap': estimates based on non-parametric bootstrap. This can be very
  accurate for enough resamplings, but can be quite slow because it involves
  computing a large number of periodograms.
"""
import numpy as np
from scipy.special import gammaln

from .core import LombScargle


def _weighted_sum(val, dy):
    return (val / dy ** 2).sum()


def _weighted_mean(val, dy):
    return _weighted_sum(val, dy) / _weighted_sum(np.ones_like(val), dy)


def _weighted_var(val, dy):
    return _weighted_mean(val ** 2, dy) - _weighted_mean(val, dy) ** 2


def _gamma(N):
    # Note: this is closely approximated by (1 - 1 / N) for large N
    return np.sqrt(2 / N) * np.exp(gammaln(N / 2) - gammaln((N - 1) / 2))


def FAP_single(Z, N, normalization='standard', dH=1, dK=3):
    """
    False Alarm Probability for a single observation

    These are adapted from table 1 of Baluev 2008
    """
    NH = N - dH  # DOF for null hypothesis
    NK = N - dK  # DOF for periodic hypothesis
    if normalization == 'psd':
        # 'psd' normalization is same as Baluev's z
        return np.exp(-Z)
    elif normalization == 'standard':
        # 'standard' normalization is Z = 2/NH * z_1
        return (1 - Z) ** (NK / 2)
    elif normalization == 'model':
        # 'model' normalization is Z = 2/NK * z_2
        return (1 + Z) ** -(NK / 2)
    elif normalization == 'log':
        # 'log' normalization is Z = 2/NK * z_3
        return np.exp(-0.5 * NK * Z)
    else:
        raise NotImplementedError("normalization={0}".format(normalization))


def tau_davies(Z, fmax, t, y, dy, normalization='standard', dH=1, dK=3):
    """tau factor for estimating Davies bound (Baluev 2008, Table 1)"""
    N = len(t)
    NH = N - dH  # DOF for null hypothesis
    NK = N - dK  # DOF for periodic hypothesis
    Dt = _weighted_var(t, dy)
    Teff = np.sqrt(4 * np.pi * Dt)
    W = fmax * Teff
    if normalization == 'psd':
        # 'psd' normalization is same as Baluev's z
        return W * np.exp(-Z) * np.sqrt(Z)
    elif normalization == 'standard':
        # 'standard' normalization is Z = 2/NH * z_1
        return (_gamma(NH) * W * (1 - Z) ** (0.5 * (NK - 1))
                * np.sqrt(0.5 * NH * Z))
    elif normalization == 'model':
        # 'model' normalization is Z = 2/NK * z_2
        return (_gamma(NK) * W * (1 + Z) ** (-0.5 * NK)
                * np.sqrt(0.5 * NK * Z))
    elif normalization == 'log':
        # 'log' normalization is Z = 2/NK * z_3
        return (_gamma(NK) * W * np.exp(-0.5 * Z * (NK - 0.5))
                * np.sqrt(NK * np.sinh(0.5 * Z)))
    else:
        raise NotImplementedError("normalization={0}".format(normalization))


def FAP_simple(Z, fmax, t, y, dy, normalization='standard'):
    """False Alarm Probability based on estimated number of indep frequencies"""
    N = len(t)
    T = max(t) - min(t)
    N_eff = fmax * T
    return 1 - (1 - FAP_single(Z, N, normalization=normalization)) ** N_eff


def FAP_davies(Z, fmax, t, y, dy, normalization='standard'):
    """Davies bound (Eqn 5 of Baluev 2008)"""
    N = len(t)
    FAP_s = FAP_single(Z, N, normalization=normalization)
    tau = tau_davies(Z, fmax, t, y, dy, normalization=normalization)
    return FAP_s + tau


def FAP_baluev(Z, fmax, t, y, dy, normalization='standard'):
    """Alias-free approximation to FAP (Eqn 6 of Baluev 2008)"""
    N = len(t)
    P_s = 1 - FAP_single(Z, N, normalization=normalization)
    tau = tau_davies(Z, fmax, t, y, dy, normalization=normalization)
    return 1 - P_s * np.exp(-tau)


def FAP_bootstrap(Z, fmax, t, y, dy, normalization='standard',
                  n_bootstraps=1000, random_seed=None):
    rng = np.random.RandomState(random_seed)

    def bootstrapped_power():
        resample = rng.randint(0, len(y), len(y))  # sample with replacement
        ls_boot = LombScargle(t, y[resample], dy[resample])
        freq, power = ls_boot.autopower(normalization=normalization,
                                        maximum_frequency=fmax)
        return power.max()

    pmax = np.array([bootstrapped_power() for i in range(n_bootstraps)])
    pmax.sort()
    return 1 - np.searchsorted(pmax, Z) / len(pmax)


METHODS = {'simple': FAP_simple,
           'davies': FAP_davies,
           'baluev': FAP_baluev,
           'bootstrap': FAP_bootstrap}


def false_alarm_probability(Z, fmax, t, y, dy, normalization,
                            method='baluev',
                            method_kwds=None):
    """Approximate the False Alarm Probability

    Parameters
    ----------
    TODO
    """
    if method not in METHODS:
        raise ValueError("Unrecognized method: {0}".format(method))
    method = METHODS[method]
    method_kwds = method_kwds or {}

    return method(Z, fmax, t, y, dy, normalization, **method_kwds)


def significance_level(sig, fmax, t, y, dy, normalization,
                       method='baluev',
                       method_kwds=None):
    pass
