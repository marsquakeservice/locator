import numpy as np
from h5py import File
from obspy import UTCDateTime
from scipy.interpolate import interp2d
from yaml import load

_type = dict(R1 = 'rayleigh',
             G1 = 'love')

def read_model_list(fnam_models, fnam_weights):
    """
    Read model list and weights from files
    :param fnam_models: Path to model file, format: lines of "model name, path"
    :param fnam_weights: Path to weight file, format: lines of "model name, weight"
    :return: Numpy arrays of file names and weights
    """
    fnams = np.loadtxt(fnam_models, dtype=str, usecols=(1))
    weights = np.loadtxt(fnam_weights, dtype=float, usecols=(1))
    return fnams, weights


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
                                freqs=freqs, distances=distances, tt=tt,
                                backazimuth=backazimuth)
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


def _read_surface_waves(f, ifile, phase_list, freqs, distances, tt, backazimuth):
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
            tt[ifile, :, :, iphase] = ipl(backazimuth,
                                          distances).T


def read_input(filename):
    with open(filename, 'r') as f:
        input_yml = load(f)
        phase_list, tt_meas, sigma, freqs, tt_ref = \
            serialize_phases(input_yml['phases'])
        try:
            backazimuth = input_yml['backazimuth']['value']
        except KeyError:
            backazimuth = 0.0

        model_name = input_yml['velocity_model']
        sigma_model = input_yml['velocity_model_uncertainty']


    return model_name, phase_list, tt_meas, sigma, freqs, backazimuth, tt_ref, sigma_model


def serialize_phases(phases):

    phase_list = []
    tt_meas = np.zeros(len(phases))
    sigma = np.zeros_like(tt_meas)
    freqs = np.zeros_like(tt_meas)
    iref = 0
    for iphase, phase in enumerate(phases):
        phase_list.append(phase['code'])
        if phase['code'] in ['P', 'P1', 'PKP']:
            iref = iphase

        tt_meas[iphase] = float(UTCDateTime(phase['datetime']))
        sigma[iphase] = (phase['uncertainty_upper'] + phase['uncertainty_lower']) * 0.5
        try:
            freqs[iphase] = phase['frequency']
        except:
            freqs[iphase] = 0

    tt_ref = tt_meas[iref]
    tt_meas -= tt_ref

    return phase_list, tt_meas, sigma, freqs, tt_ref