#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from locator.graphics import plot,plot_phases
from locator.output import write_result
from locator.read_models import load_tt, read_model_list

__author__ = "Simon Stähler"
__license__ = "none"

import numpy as np
import glob
from obspy import UTCDateTime
from yaml import load
from argparse import ArgumentParser

def define_arguments():
    helptext = 'Locate event based on travel times on single station'
    parser = ArgumentParser(description=helptext)

    helptext = "Input YAML file"
    parser.add_argument('input_file', help=helptext)

    helptext = "Output YAML file"
    parser.add_argument('output_file', help=helptext)

    helptext = "Path to model list file"
    parser.add_argument('model_path', help=helptext)

    helptext = "Path to model weight file"
    parser.add_argument('weight_path', help=helptext)

    helptext = "Create plots"
    parser.add_argument('--plot', help=helptext,
                        default=False, action='store_true')

    return parser.parse_args()


def serialize_phases(filename):
    with open(filename, 'r') as f:
        input = load(f)
    phases = input['phases']

    try:
        backazimuth = input['backazimuth']['value']
    except KeyError:
        backazimuth = 0.0

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

    return phase_list, tt_meas, sigma, freqs, backazimuth, tt_ref


if __name__ == '__main__':
    args = define_arguments()

    input_file = args.input_file
    output_file = args.output_file
    # model_path = args.model_path
    # files = glob.glob(model_path)
    # files.sort()
    files, weights = read_model_list(args.model_path, args.weight_path)

    phase_list, tt_meas, sigma, freqs, backazimuth, t_ref = serialize_phases(input_file)

    tt, dep, dis, tt_P = load_tt(files=files,
                                 phase_list=phase_list,
                                 freqs=freqs,
                                 backazimuth=backazimuth)
    tt[tt == -1] = 1e5
    tt_P[tt_P == -1] = 1e5

    modelling_error = 2
    misfit = ((tt - tt_meas) / sigma)**2

    nmodel, ndepth, ndist, nphase = tt.shape

    deldis = np.zeros((1, 1, ndist))
    deldis[0, 0, 0] = (dis[1] - dis[0]) * 0.5
    deldis[0, 0, 1:-1] = (dis[2:] - dis[0:-2]) * 0.5
    deldis[0, 0, -1] = (dis[-1] - dis[-2]) * 0.5

    deldep = np.zeros((1, ndepth, 1))
    deldep[0, 0, 0] = (dep[1] - dep[0]) * 0.5
    deldep[0, 1:-1, 0] = (dep[2:] - dep[0:-2]) * 0.5
    deldep[0, -1, 0] = (dep[-1] - dep[-2]) * 0.5

    p = 1. / np.sqrt((2*np.pi)**nphase * np.prod(sigma)) \
        * np.exp(-0.5 * np.sum(misfit, axis=3))

    p *= deldis
    #p *= deldep
    if args.plot:
        plot(p, dep=dep, dis=dis)
        plot_phases(tt, p, phase_list, tt_meas, sigma)

    write_result(file_out=output_file,
                 p=p, dep=dep, dis=dis,
                 phase_list=phase_list,
                 tt_meas=tt_meas,
                 tt_P=tt_P, t_ref=t_ref)



