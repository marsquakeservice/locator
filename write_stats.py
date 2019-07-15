#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Simon Stähler"

import numpy as np
from locator.input import load_tt, read_model_list, load_slowness
from h5py import File
from argparse import ArgumentParser
from os.path import join as pjoin
from os import environ as env



def define_arguments():
    helptext = 'Predict phase arrivals based on locator solution'
    parser = ArgumentParser(description=helptext)

    helptext = "Output file of locator"
    parser.add_argument('locator_output', help=helptext)

    return parser.parse_args()



def load_H5(fnam):
    with File(fnam, 'r') as f:
        H5 = {
            'model_name': f['modelset_name'].value,
            'p': f['p'].value,
            'depths': f['depths'].value,
            'distances': f['distances'].value,
            'phase_list': f['phase_list'].value,
            't_ref': f['t_ref'].value,
            'baz': f['backazimuth'].value,
            'tt_meas': f['tt_meas'].value,
            'freqs': f['freqs'].value,
            'periods': 1./f['freqs'].value}
    return H5


def write_variable(var, p, fnam, p_thresh = 1e-5,
                   var_thresh=[1e-3, 1e4]):
    p_thresh /= p.max(axis=None)
    var_out = var[p>p_thresh].flatten()
    p_out = p[p>p_thresh].flatten()

    p_out = p_out[var_out<var_thresh[1]].flatten()
    var_out = var_out[var_out<var_thresh[1]].flatten()
    p_out = p_out[var_out>var_thresh[0]].flatten()
    var_out = var_out[var_out>var_thresh[0]].flatten()
    np.savetxt(fname=fnam, X=np.asarray([p_out, var_out]).T)


def main(args):
    fnam_locatoroutput = args.locator_output
    H5 = load_H5(fnam_locatoroutput)
    phase_list = ['P', 'S']
    tt_path = pjoin(env['SINGLESTATION'],
                    'data', 'bodywave',
                    H5['model_name'])
    model_path = pjoin(tt_path,
                       '%s.models' % H5['model_name'])
    weight_path = pjoin(tt_path,
                        '%s.weights' % H5['model_name'])
    files, weights, model_names, all_weights  = \
        read_model_list(model_path, weight_path)
    slowness = load_slowness(files=files,
                             tt_path=tt_path,
                             phase_list=phase_list)


    tt = load_tt(files=files,
                 tt_path=tt_path,
                 phase_list=phase_list,
                 freqs=H5['freqs'],
                 backazimuth=H5['baz'],
                 idx_ref=0)[0]

    write_variable(var = tt[:,:,:,1] - tt[:,:,:,0],
                   p=H5['p'], fnam='tS_tP_diff.txt')
    for i in range(0, slowness.shape[3]):
        write_variable(var = np.deg2rad(slowness[:,:,:,i]),
                       p=H5['p'],
                       fnam='slowness_%s.txt' % phase_list[i])


if __name__ == '__main__':
    args = define_arguments()
    main(args)
