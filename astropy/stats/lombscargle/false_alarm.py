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
from scipy import interpolate, optimize

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


def _log_gamma(N):
    return 0.5 * np.log(2 / N) + gammaln(N / 2) - gammaln((N - 1) / 2)


def vectorize_first_argument(func):
    def vectorized_func(x, *args, **kwargs):
        x = np.asarray(x)
        y = np.array([func(xi, *args, **kwargs) for xi in x.flat])
        return y.reshape(x.shape)
    return vectorized_func


def inverted(func):
    """
    Numerically invert a monotonic function with domain (0, inf)

    Parameters
    ----------
    func : function
        The function to be inverted. Assumed to be monotonic with positive domain.

    Returns
    -------
    invfunc : function
        The numerical inverse of func.
    """
    @vectorize_first_argument
    def invf(y, *args, **kwargs):
        # solve for logx to avoid domain errors
        minfunc = lambda logx: (func(np.exp(logx), *args, **kwargs) - y) ** 2
        result = optimize.minimize_scalar(minfunc)
        if not result.success:
            raise ValueError("could not invert function '{0.__name__}' "
                             "at y={1}".format(func, y))
        return np.exp(result.x)
    return invf


def FAP_single(Z, N, normalization='standard', dH=1, dK=3):
    """
    Cumulative probability for a single frequency
    These are adapted from table 1 of Baluev 2008
    """
    NH = N - dH  # DOF for null hypothesis
    NK = N - dK  # DOF for periodic hypothesis
    if normalization == 'psd':
        # 'psd' normalization is same as Baluev's z
        return np.exp(-Z)
    elif normalization == 'standard':
        # 'standard' normalization is Z = 2/NH * z_1
        return (1 - Z) ** (0.5 * NK)
    elif normalization == 'model':
        # 'model' normalization is Z = 2/NK * z_2
        return (1 + Z) ** -(0.5 * NK)
    elif normalization == 'log':
        # 'log' normalization is Z = 2/NK * z_3
        return np.exp(-0.5 * NK * Z)
    else:
        raise NotImplementedError("normalization={0}".format(normalization))


def log_FAP_single(Z, N, normalization='standard', dH=1, dK=3):
    """
    log of cumulative probability for a single frequency
    These are adapted from table 1 of Baluev 2008
    """
    NH = N - dH  # DOF for null hypothesis
    NK = N - dK  # DOF for periodic hypothesis
    if normalization == 'psd':
        # 'psd' normalization is same as Baluev's z
        return -Z
    elif normalization == 'standard':
        # 'standard' normalization is Z = 2/NH * z_1
        return 0.5 * NK * np.log(1 - Z)
    elif normalization == 'model':
        # 'model' normalization is Z = 2/NK * z_2
        return -0.5 * NK * np.log(1 + Z)
    elif normalization == 'log':
        # 'log' normalization is Z = 2/NK * z_3
        return -0.5 * NK * Z
    else:
        raise NotImplementedError("normalization={0}".format(normalization))


def P_single(Z, N, normalization='standard', dH=1, dK=3):
    return 1 - FAP_single(Z, N, normalization=normalization, dH=dH, dK=dK)

def log_P_single(Z, N, normalization='standard', dH=1, dK=3):
    return np.log1p(-FAP_single(Z, N, normalization=normalization, dH=dH, dK=dK))


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


def log_tau_davies(Z, fmax, t, y, dy, normalization='standard', dH=1, dK=3):
    """tau factor for estimating Davies bound (Baluev 2008, Table 1)"""
    N = len(t)
    NH = N - dH  # DOF for null hypothesis
    NK = N - dK  # DOF for periodic hypothesis
    Dt = _weighted_var(t, dy)
    Teff = np.sqrt(4 * np.pi * Dt)
    W = fmax * Teff
    if normalization == 'psd':
        # 'psd' normalization is same as Baluev's z
        return np.log(W) - Z + 0.5 * np.log(Z)
    elif normalization == 'standard':
        # 'standard' normalization is Z = 2/NH * z_1
        return (_log_gamma(NH) + np.log(W) + 0.5 * (NK - 1) * np.log(1 - Z)
                + 0.5 * np.log(0.5 * NH * Z))
    elif normalization == 'model':
        # 'model' normalization is Z = 2/NK * z_2
        return (_log_gamma(NK) + np.log(W) - 0.5 * NK * np.log(1 + Z)
                + 0.5 * np.log(0.5 * NK * Z))
    elif normalization == 'log':
        # 'log' normalization is Z = 2/NK * z_3
        return (_log_gamma(NK) + np.log(W) - 0.5 * Z * (NK - 0.5)
                + 0.5 * np.log(NK * np.sinh(0.5 * Z)))
    else:
        raise NotImplementedError("normalization={0}".format(normalization))


def FAP_simple(Z, fmax, t, y, dy, normalization='standard'):
    """False Alarm Probability based on estimated number of indep frequencies"""
    N = len(t)
    T = max(t) - min(t)
    N_eff = fmax * T
    P_s = P_single(Z, N, normalization=normalization)
    return 1 - P_s ** N_eff


def log_FAP_simple(Z, fmax, t, y, dy, normalization='standard'):
    """log FAP based on estimated number of indep frequencies"""
    N = len(t)
    T = max(t) - min(t)
    N_eff = fmax * T
    log_P_s = log_P_single(Z, N, normalization=normalization)
    return np.log1p(-np.exp(N_eff * log_P_s))


def FAP_davies(Z, fmax, t, y, dy, normalization='standard'):
    """Davies bound (Eqn 5 of Baluev 2008)"""
    N = len(t)
    FAP_s = FAP_single(Z, N, normalization=normalization)
    tau = tau_davies(Z, fmax, t, y, dy, normalization=normalization)
    return FAP_s + tau


def log_FAP_davies(Z, fmax, t, y, dy, normalization='standard'):
    """Davies bound (Eqn 5 of Baluev 2008)"""
    N = len(t)
    log_FAP_s = log_FAP_single(Z, N, normalization=normalization)
    log_tau = log_tau_davies(Z, fmax, t, y, dy, normalization=normalization)
    return np.logaddexp(log_FAP_s, log_tau)


def FAP_baluev(Z, fmax, t, y, dy, normalization='standard'):
    """Alias-free approximation to FAP (Eqn 6 of Baluev 2008)"""
    P_s = P_single(Z, len(t), normalization=normalization)
    tau = tau_davies(Z, fmax, t, y, dy, normalization=normalization)
    return 1 - P_s * np.exp(-tau)


def log_FAP_baluev(Z, fmax, t, y, dy, normalization='standard'):
    """Alias-free approximation to FAP (Eqn 6 of Baluev 2008)"""
    log_P_s = log_P_single(Z, len(t), normalization=normalization)
    tau = tau_davies(Z, fmax, t, y, dy, normalization=normalization)
    return np.log1p(-np.exp(log_P_s - tau))


def _bootstrap(fmax, t, y, dy, normalization='standard',
               n_bootstraps=1000, random_seed=None):
    rng = np.random.RandomState(random_seed)

    def bootstrapped_power():
        resample = rng.randint(0, len(y), len(y))  # sample with replacement
        ls_boot = LombScargle(t, y[resample], dy[resample])
        freq, power = ls_boot.autopower(normalization=normalization,
                                        maximum_frequency=fmax)
        return power.max()

    pmax = np.array([bootstrapped_power() for i in range(n_bootstraps)])
    return pmax


def FAP_bootstrap(Z, fmax, t, y, dy, normalization='standard',
                  n_bootstraps=1000, random_seed=None):
    pmax = _bootstrap(fmax, t, y, dy, normalization=normalization,
                      n_bootstraps=n_bootstraps, random_seed=random_seed)
    pmax.sort()
    return 1 - np.searchsorted(pmax, Z) / len(pmax)


def significance_bootstrap(significance, fmax, t, y, dy,
                           normalization='standard',
                           n_bootstraps=1000, random_seed=None):
    pmax = _bootstrap(fmax, t, y, dy, normalization=normalization,
                      n_bootstraps=n_bootstraps, random_seed=random_seed)
    pmax.sort()
    sig = np.linspace(0, 1, n_bootstraps, endpoint=False)
    return interpolate.interp1d(sig, pmax)(significance)


METHODS = {'simple': FAP_simple,
           'davies': FAP_davies,
           'baluev': FAP_baluev,
           'bootstrap': FAP_bootstrap}


def false_alarm_probability(Z, fmax, t, y, dy, normalization,
                            method='baluev', method_kwds=None):
    """Approximate the False Alarm Probability

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    if method not in METHODS:
        raise ValueError("Unrecognized method: {0}".format(method))
    method = METHODS[method]
    method_kwds = method_kwds or {}

    return method(Z, fmax, t, y, dy, normalization, **method_kwds)


def significance_level(significance, fmax, t, y, dy, normalization,
                       method='baluev', method_kwds=None):
    """Peak significance level

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    significance = np.asarray(significance)
    method_kwds = method_kwds or {}
    if np.any((significance <= 0) | (significance > 1)):
        raise ValueError("significance is out of range (0, 1]")
    if method == 'bootstrap':
        return significance_bootstrap(significance, fmax, t, y, dy,
                                      normalization, **method_kwds)
    else:
        return inverted(false_alarm_probability)(1 - significance, fmax, t, y, dy,
                                                 normalization=normalization,
                                                 method=method,
                                                 method_kwds=method_kwds)
