import numpy as np
from h5py import File
from scipy.interpolate import interp2d

_type = dict(R1 = 'rayleigh',
             G1 = 'love')


def load_tt(files, phase_list, freqs, backazimuth):

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
            _read_surface_waves(f, ifile=ifile, phase_list=phase_list,
                                freqs=freqs, distances=distances, tt=tt)
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
        # Is it a body wave?
        if phase.encode() in phase_names:
            idx = phase_names.index(phase.encode())
            tt[ifile, :, :, iphase] = f['/body_waves/times'][:, :, idx]


def _read_surface_waves(f, ifile, phase_list, freqs, distances, tt):
    baz = 180
    periods = f['/surface_waves/periods'].value
    dist_model = f['/surface_waves/distances']
    dist_pad = np.zeros(len(dist_model) + 1)
    dist_pad[1:] = dist_model
    baz_model = f['/surface_waves/backazimuths']

    _tmp =  f['/surface_waves/period_00/tt_love']
    tt_pad = np.zeros((_tmp.shape[0] + 1, _tmp.shape[1]))

    for iphase, phase in enumerate(phase_list):
        if phase in ['R1', 'G1']:
            idx_freq = np.argmin(abs(1./freqs[iphase] - periods))
            tt_pad[1:, :] = f['/surface_waves/period_%02d/tt_%s' %
                              (idx_freq, _type[phase])]
            ipl = interp2d(x=baz_model, y=dist_pad, z=tt_pad, kind='cubic')
            tt[ifile, :, :, iphase] = ipl(baz,
                                          distances).T

