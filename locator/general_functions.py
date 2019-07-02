import numpy as np

from scipy.interpolate import interp1d


def _get_median(x, p_x):
    ipl = interp1d(x, p_x, kind='cubic')
    x_ipl = np.arange(x[0], x[-1], 0.1)
    p_ipl = ipl(x_ipl)
    p_ipl /= np.sum(p_ipl)
    p_cum = np.cumsum(p_ipl)
    return x_ipl[np.argmin(abs(p_cum - 0.5))]


def _calc_marginals(dep, dis, p):
    """
    Calculate depth and distance marginals from full P-matrix taking the
    irregular support points for depth and distance into account
    :param dep: 1D array with depth support points
    :param dis: 1D array with distance support points
    :param p: 3D array with probabilities: (nmodels, ndepths, ndistances)
    :return:
    """
    deldis = calc_del(dis).reshape((1, 1, len(dis)))
    deldep = calc_del(dep).reshape((1, len(dep), 1))
    p_dist = np.sum(p * deldep, axis=(0, 1))
    p_depth = np.sum(p * deldis, axis=(0, 2))
    p_depdis = np.sum(p * deldis * deldep, axis=0)

    depth_mean = _get_median(dep, p_depth)
    dist_mean = _get_median(dis, p_dist)
    return depth_mean, dist_mean, p_depdis, p_depth, p_dist


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