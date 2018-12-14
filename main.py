#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from locator.graphics import plot_phases, plot, plot_models
from locator.misfits import calc_p
from locator.output import write_result
from locator.input import load_tt, read_model_list, read_input

__author__ = "Simon Stähler"
__license__ = "none"

from argparse import ArgumentParser
import os
import numpy as np


def define_arguments():
    helptext = 'Locate event based on travel times on single station'
    parser = ArgumentParser(description=helptext)

    helptext = "Input YAML file"
    parser.add_argument('input_file', help=helptext)

    helptext = "Output YAML file"
    parser.add_argument('output_file', help=helptext)

    helptext = "Path to model list file"
    parser.add_argument('--model_path', help=helptext,
                        default=None)

    helptext = "Path to model weight file"
    parser.add_argument('--model_weights', help=helptext,
                        default=None)

    helptext = "Create plots"
    parser.add_argument('--plot', help=helptext,
                        default=False, action='store_true')

    helptext = "A priori 1/e depth"
    parser.add_argument('-d', '--max_depth', help=helptext,
                        type=float, default=100.)

    helptext = "Use distance prior information based on area"
    parser.add_argument('--dist_prior', help=helptext,
                        default=False, action='store_true')

    return parser.parse_args()


def main(input_file, output_file, model_path, weight_path, plot_output, max_depth, use_distance_prior):
    # model_name, phase_list, tt_meas, sigma_pick, freqs, backazimuth, t_ref, sigma_model = read_input(input_file)
    input = read_input(input_file)

    tt_path = os.path.join(os.environ['SINGLESTATION'], 
                           'data', 'bodywave',
                           input['model_name'])
    if not model_path:
        model_path=os.path.join(tt_path,
                                '%s.models' % input['model_name'])
    if not weight_path:
        weight_path=os.path.join(tt_path,
                                 '%s.weights' % input['model_name'])
    files, weights, models, prior_weights = read_model_list(model_path, weight_path)

    tt, dep, dis, tt_P = load_tt(files=files,
                                 tt_path=tt_path,
                                 phase_list=input['phase_list'],
                                 freqs=input['freqs'],
                                 backazimuth=input['backazimuth'],
                                 idx_ref=input['idx_ref'])

    # Total sigma is sigma of modelled travel time plus picking uncertainty
    sigma = input['sigma_model'] + input['sigma']

    # depth prior
    depth_prior = np.exp(-(dep/max_depth)**2)

    # distance prior
    if use_distance_prior:
        distance_prior = np.sin(np.deg2rad(dis))
    else:
        distance_prior = None

    # Calculate probability
    p = calc_p(dep, dis, sigma, tt, input['tt_meas'], weights,
               depth_prior=depth_prior, distance_prior=distance_prior)

    if np.max(p, axis=None) < 1E-30:
        raise ValueError('Travel times are incompatible \n' +
                         '  highest p: %8.2e\n' % np.max(p, axis=None) +
                         '  threshold: %8.2e\n ' % 1e-30)


    if plot_output:
        plot(p, dep=dep, dis=dis, depth_prior=depth_prior,
             distance_prior=distance_prior)
        plot_phases(tt, p, input['phase_list'],
                    input['freqs'], input['tt_meas'],
                    input['sigma'])
        plot_models(p, files, tt_path)
    write_result(file_out=output_file,
                 modelset_name=input['model_name'],
                 p=p, dep=dep, dis=dis,
                 phase_list=input['phase_list'],
                 freqs=input['freqs'],
                 tt_meas=input['tt_meas'],
                 baz=input['backazimuth'],
                 tt_P=tt_P, t_ref=input['tt_ref'],
                 weights=weights,
                 model_names=models)


if __name__ == '__main__':
    args = define_arguments()
    main(input_file=args.input_file,
         output_file=args.output_file,
         model_path=args.model_path,
         weight_path=args.weight_path,
         plot_output=args.plot,
         max_depth=args.max_depth,
         use_distance_prior=args.dist_prior)
