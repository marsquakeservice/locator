import numpy as np


def calc_p(dep, dis, sigma, tt, tt_meas):
    misfit = ((tt - tt_meas) / sigma) ** 2
    nmodel, ndepth, ndist, nphase = tt.shape
    deldis = calc_del(dis)
    deldep = calc_del(dep)
    # deldep = np.zeros((1, ndepth, 1))
    # deldep[0, 0, 0] = (dep[1] - dep[0]) * 0.5
    # deldep[0, 1:-1, 0] = (dep[2:] - dep[0:-2]) * 0.5
    # deldep[0, -1, 0] = (dep[-1] - dep[-2]) * 0.5
    p = 1. / np.sqrt((2 * np.pi) ** nphase * np.prod(sigma)) \
        * np.exp(-0.5 * np.sum(misfit, axis=3))
    p *= deldis
    return p


def calc_del(dis):
    ndist = len(dis)
    deldis = np.zeros((1, 1, ndist))
    deldis[0, 0, 0] = (dis[1] - dis[0]) * 0.5
    deldis[0, 0, 1:-1] = (dis[2:] - dis[0:-2]) * 0.5
    deldis[0, 0, -1] = (dis[-1] - dis[-2]) * 0.5
    return deldis