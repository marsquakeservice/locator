# -*- coding: utf-8 -*-
import numpy as np
from h5py import File
from obspy import UTCDateTime
from scipy.interpolate import interp2d
from yaml import load
from os.path import join as pjoin

_type = dict(R1 = 'rayleigh',
             G1 = 'love')

def read_model_list(fnam_models, fnam_weights, weight_lim=1e-5):
    """
    Read model list and weights from files
    :param fnam_models: Path to model file, format: lines of "model name, path"
    :param fnam_weights: Path to weight file, format: lines of "model name, weight"
    :return: Numpy arrays of file names and weights
    """
    modelnames = np.loadtxt(fnam_models, dtype=str, usecols=[0])
    fnams = np.loadtxt(fnam_models, dtype=str, usecols=[1])
    modelnames_weights = np.loadtxt(fnam_weights, dtype=str, usecols=[0])
    weights = np.loadtxt(fnam_weights, dtype=float, usecols=[1])
    if not (modelnames == modelnames_weights).all():
        raise RuntimeError('Model names have to be identical in model and weight file')
    return fnams[weights>weight_lim], weights[weights>weight_lim], modelnames, weights


def load_slowness(files, tt_path, phase_list):

    # Get dimension of TT variable (i.e. number of depths and distances)
    with File(pjoin(tt_path, 'tt', files[0])) as f:
        tts = f['/body_waves/times']
        ndepth, ndist, nphase = tts.shape
        phase_names = f['/body_waves/phase_names'][()].tolist()
        depths = f['/body_waves/depths'][()]
        distances = f['/body_waves/distances'][()]

    slowness = np.zeros((len(files), ndepth, ndist,
                         len(phase_list)), dtype='float32')

    for ifile, file in enumerate(files):
        with File(pjoin(tt_path, 'tt', file)) as f:
            for iphase, phase in enumerate(phase_list):
                # Is it a body wave?
                if phase.encode() in phase_names:
                    idx = phase_names.index(phase.encode())
                    slowness[ifile, :, :, iphase] = \
                        f['/body_waves/slowness'][:, :, idx]

    # -1 is the value for no arrival at this distance/model/depth combo
    slowness[slowness < 1e-9] = -1

    return slowness


def load_tt(files, tt_path, phase_list, freqs, backazimuth, idx_ref):

    # Get dimension of TT variable (i.e. number of depths and distances)
    with File(pjoin(tt_path, 'tt', files[0]), mode='r') as f:
        tts = f['/body_waves/times']
        ndepth, ndist, nphase = tts.shape
        phase_names = f['/body_waves/phase_names'][()].tolist()
        depths = f['/body_waves/depths'][()]
        distances = f['/body_waves/distances'][()]

    tt = np.zeros((len(files), ndepth, ndist, len(phase_list)), dtype='float32')

    for ifile, file in enumerate(files):
        with File(pjoin(tt_path, 'tt', file), mode='r') as f:
            _read_body_waves(f, ifile, phase_list, phase_names, tt)
            if 'R1' in phase_list or 'G1' in phase_list:
                _read_surface_waves(f, ifile=ifile, phase_list=phase_list,
                                    freqs=freqs, distances=distances, tt=tt,
                                    backazimuth=backazimuth)

    tt_P = np.zeros((len(files), ndepth, ndist, 1), dtype='float32')
    tt_P[:, :, :, 0] = tt[:, :, :, idx_ref]

    # -1 is the value for no arrival at this distance/model/depth combo
    bool = tt == -1
    bool_P = tt_P == -1
    tt[bool] = 1e6
    tt_P[bool_P] = 1e6

    tt -= tt_P
    tt[bool] = 1e6
    return tt, depths, distances, tt_P


def _read_body_waves(f, ifile, phase_list, phase_names, tt):
    for iphase, phase in enumerate(phase_list):
        # Is it a body wave?
        if phase.encode() in phase_names:
            idx = phase_names.index(phase.encode())
            tt[ifile, :, :, iphase] = f['/body_waves/times'][:, :, idx]


def _read_surface_waves(f, ifile, phase_list, freqs, distances, tt, backazimuth):
    periods = f['/surface_waves/periods'][()]
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
        phase_list, tt_meas, sigma, freqs, idx_ref, tt_ref = \
            serialize_phases(input_yml['phases'])
        try:
            backazimuth = input_yml['backazimuth']['value']
        except KeyError:
            backazimuth = 0.0

        model_name = input_yml['velocity_model']
        sigma_model = input_yml['velocity_model_uncertainty']

    input = {'model_name': model_name,
             'phase_list': phase_list,
             'tt_meas': tt_meas,
             'sigma': sigma,
             'freqs': freqs,
             'backazimuth': backazimuth,
             'idx_ref': idx_ref,
             'tt_ref': tt_ref,
             'sigma_model': sigma_model}

    return input


def serialize_phases(phases):

    phase_list = []
    tt_meas = np.zeros(len(phases))
    sigma = np.zeros_like(tt_meas)
    freqs = np.zeros_like(tt_meas)
    iref = -1
    for iphase, phase in enumerate(phases):
        phase_list.append(phase['code'])

        tt_meas[iphase] = float(UTCDateTime(phase['datetime']))
        sigma[iphase] = (phase['uncertainty_upper'] + phase['uncertainty_lower']) * 0.5
        if phase['code'] in ['R1', 'G1']:
            try:
                freqs[iphase] = phase['frequency']
            except KeyError:
                raise ValueError('Surface wave phase %s has no frequency' % phase['code'])
    iref = np.argmin(tt_meas)

    tt_ref = tt_meas[iref]
    tt_meas -= tt_ref

    return phase_list, tt_meas, sigma, freqs, iref, tt_ref


def read_h5_locator_output(fnam):
    """
    Read model specific part of locator output file
    :param fnam: file name of HDF5 file
    :return: p: probability matrix
             dis: distance support points
             dep: depth support points
             weights: a priori weights
    """
    with File(fnam, mode='r') as f:
        p = np.asarray(f['p'][()])
        dis = np.asarray(f['distances'][()])
        dep = np.asarray(f['depths'][()])
        weights = np.asarray(f['weights'][()])
        model_names = f['model_names']
        modelset_name = '%s' % f['modelset_name'][()]

        # f.create_dataset('modelset_name', data=modelset_name)
        # f.create_dataset('phase_list', data=[n.encode("utf-8", "ignore")
        #                                      for n in phase_list])
        # f.create_dataset('model_names', data=[n.encode("utf-8", "ignore")
        #                                       for n in model_names])
        # f.create_dataset('weights', data=weights)
        # f.create_dataset('sigma', data=sigma)
        # f.create_dataset('tt_meas', data=tt_meas)
        # f.create_dataset('freqs', data=freqs)
        # f.create_dataset('t_ref', data=t_ref)
        # f.create_dataset('backazimuth', data=baz)
        # f.create_dataset('origin_time', data=float(origin_time))
    return p, dep, dis, weights, modelset_name, model_names
