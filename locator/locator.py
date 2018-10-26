#!/usr/bin/env python
"""

"""
__author__ = "Simon Stähler"
__license__ = "none"

import numpy as np
from h5py import File
import glob
import matplotlib.pyplot as plt
from obspy import UTCDateTime

def load_tt(files, phase_list):

    # Get dimension of TT variable (i.e. number of depths and distances)
    with File(files[0]) as f:
        tts = f['/body_waves/times']
        ndepth, ndist, nphase = tts.shape
        phase_names = f['/body_waves/phase_names'].value.tolist()
        depths = f['/body_waves/depths'].value
        distances = f['/body_waves/distances'].value

    tt = np.zeros((len(files), ndepth, ndist, len(phase_list)), dtype='float32')

    for ifile, file in enumerate(files):
        with File(file) as f:
            for iphase, phase in enumerate(phase_list):
                idx = phase_names.index(phase.encode())
                tt[ifile, :, :, iphase] = f['/body_waves/times'][:, :, idx]
    tt_P = np.zeros((len(files), ndepth, ndist, 1), dtype='float32')
    tt_P[:, :, :, 0] = tt[:, :, :, 0]
    tt -= tt_P
    return tt, depths, distances


if __name__ == '__main__':
    tt, dep, dis = load_tt(files=glob.glob('tt/mantlecrust_00???.h5'),
                           phase_list=['P', 'pP', 'S'])

    tt_meas = np.asarray([float(UTCDateTime('2019-01-03T15:09:54.9')),
                          #float(UTCDateTime('2019-01-03T15:11:47.5')),
                          float(UTCDateTime('2019-01-03T15:10:07.8')),
                          float(UTCDateTime('2019-01-03T15:18:34.6'))])

    tt_meas -= tt_meas[0]
    tt_meas[tt_meas==-1] = 1e5
    modelling_error = 2
    sigma = np.asarray([1, 2, 5])
    sigma += modelling_error
    misfit = ((tt - tt_meas) / sigma)**2
    nphase = len(tt_meas)
    nmodel = tt.shape[0]
    ndepth = len(dep)
    ndist = len(dis)
    p = 1./np.sqrt((2*np.pi)**nphase*np.prod(sigma)) * np.prod(np.exp(-misfit), axis=3)

    depthdist = np.sum(p, axis=(0)) / nmodel
    plt.contourf(dis, dep, depthdist)
    plt.colorbar()
    plt.xlabel('distance')
    plt.ylabel('depth')
    plt.tight_layout()
    plt.savefig('depth_distance.png')
    plt.show()

    plt.plot(np.sum(p, axis=(1,2)) / ndepth / ndist, '.')
    plt.xlabel('Model index')
    plt.ylabel('likelihood')
    plt.tight_layout()
    plt.savefig('model_likelihood.png')
    plt.show()
