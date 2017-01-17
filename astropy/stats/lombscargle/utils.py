import numpy as np

def compute_chi2_ref(y, dy):
    """Compute the reference chi-square for a particular dataset.

    Note: this is only valid for center_data=True or fit_mean=True.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    y, dy = np.broadcast_arrays(y, dy)
    w = dy ** -2.0
    yw = (y - np.dot(w, y) / w.sum()) / dy
    return np.dot(yw, yw) / w.mean()


def convert_normalization(Z, N, from_normalization, to_normalization,
                          chi2_ref=None, dH=1, dK=3):
    """Convert power from one normalization to another.

    This currently only works for standard & floating-mean models.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    NK = N - dK
    NH = N - dH
    valid_norms = ['standard', 'psd', 'model', 'log']

    if from_normalization not in valid_norms:
        raise ValueError("{0} is not a valid normalization"
                         "".format(from_normalization))
    if to_normalization not in valid_norms:
        raise ValueError("{0} is not a valid normalization"
                         "".format(to_normalization))

    if from_normalization == to_normalization:
        return Z

    from_to = (from_normalization, to_normalization)

    if "psd" in from_to and chi2_ref is None:
        raise ValueError("must supply reference chi^2 when converting "
                         "to or from psd normalization")

    if from_to == ('log', 'standard'):
        return 1 - np.exp(-Z)
    elif from_to == ('standard', 'log'):
        return -np.log(1 - Z)
    elif from_to == ('log', 'model'):
        return np.exp(Z) - 1
    elif from_to == ('model', 'log'):
        return np.log(Z + 1)
    elif from_to == ('model', 'standard'):
        return Z / (1 + Z)
    elif from_to == ('standard', 'model'):
        return Z / (1 - Z)
    elif from_normalization == "psd":
        return convert_normalization(2 / chi2_ref * Z, N,
                                     from_normalization='standard',
                                     to_normalization=to_normalization,
                                     dK=dK, dH=dH)
    elif to_normalization == "psd":
        Z_standard = convert_normalization(Z, N,
                                           from_normalization=from_normalization,
                                           to_normalization='standard', dK=dK, dH=dH)
        return 0.5 * chi2_ref * Z_standard
    else:
        raise NotImplementedError("conversion from '{0}' to '{1}'"
                                  "".format(from_normalization,
                                            to_normalization))
