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
from yaml import load
import argparse

def define_arguments():
    helptext = 'Locate event based on travel times on single station'
    parser = argparse.ArgumentParser(description=helptext)

    helptext = "Input YAML file"
    parser.add_argument('input_file', help=helptext)

    helptext = "Output YAML file"
    parser.add_argument('output_file', help=helptext)

    helptext = "Path to model files"
    parser.add_argument('model_path', help=helptext)

    return parser.parse_args()


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

    idx_P = phase_list.index('P')
    tt_P = np.zeros((len(files), ndepth, ndist, 1), dtype='float32')
    tt_P[:, :, :, 0] = tt[:, :, :, idx_P]
    tt -= tt_P
    return tt, depths, distances, tt_P


def plot(p, dis, dep):
    nmodel, ndepth, ndist = p.shape
    depthdist = np.sum(p, axis=(0)) / nmodel
    plt.contourf(dis, dep, depthdist)
    plt.colorbar()
    plt.xlabel('distance')
    plt.ylabel('depth')
    plt.tight_layout()
    plt.savefig('depth_distance.png')
    plt.show()
    plt.plot(np.sum(p, axis=(1, 2)) / ndepth / ndist, '.')
    plt.xlabel('Model index')
    plt.ylabel('likelihood')
    plt.tight_layout()
    plt.savefig('model_likelihood.png')
    plt.show()


def open_yaml(filename):
    with open(filename, 'r') as f:
        input = load(f)
    return input['phases']


def serialize_phases(phases):
    global iref
    phase_list = []
    tt_meas = np.zeros(len(phases))
    sigma = np.zeros_like(tt_meas)

    for iphase, phase in enumerate(phases):
        phase_list.append(phase['code'])
        if phase['code'] in ['P', 'P1', 'PKP']:
            iref = iphase

        tt_meas[iphase] = float(UTCDateTime(phase['datetime']))
        sigma[iphase] = phase['uncertainty_upper'] + phase['uncertainty_lower']

    tt_ref = tt_meas[iref]
    tt_meas -= tt_ref

    return phase_list, tt_meas, sigma, tt_ref


def write_result(file_out, p, dep, dis, tt_P):
    p_dist = np.sum(p, axis=(0, 1))
    p_depth = np.sum(p, axis=(0, 2))

    pdf_depth_sum = []
    for idepth, d in enumerate(dep):
        pdf_depth_sum.append([d, p_depth[idepth]])
    depth_sum = np.sum(dep * p_depth) / np.sum(p_depth)

    pdf_dist_sum = []
    for idist, d in enumerate(dis):
        pdf_dist_sum.append([d, p_dist[idist]])
    dist_sum = np.sum(dis * p_dist) / np.sum(p_dist)

    origin_pdf, origin_times = np.histogram(a=-tt_P.flatten(),
                                            weights=p.flatten(),
                                            bins=np.linspace(-600, 0, 100),
                                            density=True)
    pdf_origin_sum = []
    time_bin_mid = (origin_times[0:-1] + origin_times[1:]) / 2.
    for itime, time in enumerate(time_bin_mid):
        pdf_origin_sum.append([time,
                               origin_pdf[itime]])
    origin_time_sum = UTCDateTime(tt_ref) + np.sum(origin_pdf * time_bin_mid)

    with open(file_out, 'w') as f:
        write_prob(f, pdf_depth_sum=pdf_depth_sum,
                   pdf_dist_sum=pdf_dist_sum,
                   pdf_otime_sum=pdf_origin_sum)
        write_single(f, depth_sum=depth_sum, dist_sum=dist_sum,
                     origin_time_sum=origin_time_sum)
        # §f.write('pdf_depth_sum:\n\n')
        # §f.write('  probabilities: ')
        # §print(pdf_depth_sum, file=f)


def write_prob(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: \n' % key)
        f.write('  probabilities: ')
        print(value, file=f)
        f.write('\n\n')


def write_single(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: %f\n\n' % (key, value))


if __name__ == '__main__':
    args = define_arguments()

    input_file = args.input_file # './data/locator_input.yaml'
    output_file = args.output_file # './data/locator_output.yaml'
    model_path = args.model_path # '../tt/mantlecrust_00???.h5'

    phase_list, tt_meas, sigma, tt_ref = serialize_phases(open_yaml(input_file))

    files = glob.glob(model_path)
    files.sort()
    tt, dep, dis, tt_P = load_tt(files=files,
                                 phase_list=phase_list)
    tt[tt == -1] = 1e5
    tt_P[tt_P == -1] = 1e5

    modelling_error = 2
    misfit = ((tt - tt_meas) / sigma)**2
    nphase = len(tt_meas)
    nmodel = tt.shape[0]
    ndepth = len(dep)
    ndist = len(dis)
    p = 1./np.sqrt((2*np.pi)**nphase*np.prod(sigma)) * np.prod(np.exp(-misfit), axis=3)

    write_result(file_out=output_file, p=p, dep=dep, dis=dis, tt_P=tt_P)
    plot(p, dep=dep, dis=dis)
