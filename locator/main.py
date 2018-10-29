#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from graphics import plot,plot_phases
from output import write_result
from read_models import load_tt

__author__ = "Simon Stähler"
__license__ = "none"

import numpy as np
import glob
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


def serialize_phases(filename):
    with open(filename, 'r') as f:
        input = load(f)
    phases = input['phases']

    backazimuth = 180 # input['backazimuth']['value']

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
        sigma[iphase] = phase['uncertainty_upper'] + phase['uncertainty_lower']
        try:
            freqs[iphase] = phase['frequency']
        except:
            freqs[iphase] = 0

    tt_ref = tt_meas[iref]
    tt_meas -= tt_ref

    return phase_list, tt_meas, sigma, freqs, backazimuth, tt_ref


if __name__ == '__main__':
    args = define_arguments()

    input_file = args.input_file # './data/locator_input.yaml'
    output_file = args.output_file # './data/locator_output.yaml'
    model_path = args.model_path # '../tt/mantlecrust_00???.h5'

    phase_list, tt_meas, sigma, freqs, backazimuth, t_ref = serialize_phases(input_file)

    files = glob.glob(model_path)
    files.sort()
    tt, dep, dis, tt_P = load_tt(files=files,
                                 phase_list=phase_list,
                                 freqs=freqs,
                                 backazimuth=backazimuth)
    tt[tt == -1] = 1e5
    tt_P[tt_P == -1] = 1e5

    modelling_error = 2
    misfit = ((tt - tt_meas) / sigma)**2
    nphase = len(tt_meas)
    nmodel = tt.shape[0]
    ndepth = len(dep)
    ndist = len(dis)
    p = 1. / np.sqrt((2*np.pi)**nphase * np.prod(sigma)) \
        * np.exp(-0.5 * np.sum(misfit, axis=3))

    write_result(file_out=output_file,
                 p=p, dep=dep, dis=dis,
                 tt_P=tt_P, t_ref=t_ref)
    plot(p, dep=dep, dis=dis)

    plot_phases(tt, p, phase_list, tt_meas, sigma)
