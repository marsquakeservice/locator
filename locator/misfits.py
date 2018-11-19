import numpy as np


def calc_p(dep, dis, sigma, tt, tt_meas, weights):
    misfit = ((tt - tt_meas) / sigma) ** 2
    nmodel, ndepth, ndist, nphase = tt.shape

    p = 1. / np.sqrt((2 * np.pi) ** nphase * np.prod(sigma**2)) \
        * np.exp(-0.5 * np.sum(misfit, axis=3))

    # Normalize probability for different step sizes in depth
    # and distance
    # deldis = calc_del(dis)
    # deldep = calc_del(dep)
    # p *= deldis.reshape((1, 1, ndist))
    # p *= deldep.reshape((1, ndepth, 1))

    # Multiply probability with a priori weights of models
    p *= weights.reshape((nmodel, 1, 1))
    return p


def calc_del(dis):
    """
    Calculate vector of depth/distance step length
    :param dis: array of values
    :return: deldis: array of differences in dis, has same shape
    """
    ndist = len(dis)
    deldis = np.zeros((ndist))
    deldis[0] = (dis[1] - dis[0]) * 0.5
    deldis[1:-1] = (dis[2:] - dis[0:-2]) * 0.5
    deldis[-1] = (dis[-1] - dis[-2]) * 0.5
    return deldis