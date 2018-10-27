import numpy as np
from h5py import File


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
            _read_body_waves(f, ifile, phase_list, phase_names, tt)
    try:
        idx_ref = phase_list.index('P')
    except ValueError:
        idx_ref = phase_list.index('P1')

    tt_P = np.zeros((len(files), ndepth, ndist, 1), dtype='float32')
    tt_P[:, :, :, 0] = tt[:, :, :, idx_ref]
    tt -= tt_P
    return tt, depths, distances, tt_P


def _read_body_waves(f, ifile, phase_list, phase_names, tt):
    for iphase, phase in enumerate(phase_list):
        idx = phase_names.index(phase.encode())
        tt[ifile, :, :, iphase] = f['/body_waves/times'][:, :, idx]

