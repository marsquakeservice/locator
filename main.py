#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from locator.graphics import plot_phases, plot
from locator.misfits import calc_p
from locator.output import write_result
from locator.input import load_tt, read_model_list, read_input

__author__ = "Simon Stähler"
__license__ = "none"

from argparse import ArgumentParser
import os


def define_arguments():
    helptext = 'Locate event based on travel times on single station'
    parser = ArgumentParser(description=helptext)

    helptext = "Input YAML file"
    parser.add_argument('input_file', help=helptext)

    helptext = "Output YAML file"
    parser.add_argument('output_file', help=helptext)

    # helptext = "Path to model list file"
    # parser.add_argument('model_path', help=helptext)

    # helptext = "Path to model weight file"
    # parser.add_argument('weight_path', help=helptext)

    helptext = "Create plots"
    parser.add_argument('--plot', help=helptext,
                        default=False, action='store_true')

    return parser.parse_args()


def main(input_file, output_file, plot_output=False):
    model_name, phase_list, tt_meas, sigma_pick, freqs, backazimuth, t_ref, sigma_model = read_input(input_file)

    ssl_path = os.environ['SINGLESTATION']
    model_path=os.path.join(ssl_path, model_name, '%s.models' % model_name)
    weight_path=os.path.join(ssl_path, model_name, '%s.weights' % model_name)
    files, weights = read_model_list(model_path, weight_path)

    tt, dep, dis, tt_P = load_tt(files=files,
                                 phase_list=phase_list,
                                 freqs=freqs,
                                 backazimuth=backazimuth)

    # Total sigma is sigma of modelled travel time plus picking uncertainty
    sigma = sigma_model + sigma_pick

    # Calculate probability
    p = calc_p(dep, dis, sigma, tt, tt_meas, weights)

    if plot_output:
        plot(p, dep=dep, dis=dis)
        plot_phases(tt, p, phase_list, tt_meas, sigma)
    write_result(file_out=output_file,
                 p=p, dep=dep, dis=dis,
                 phase_list=phase_list,
                 tt_meas=tt_meas,
                 tt_P=tt_P, t_ref=t_ref)


if __name__ == '__main__':
    args = define_arguments()
    main(input_file=args.input_file,
         output_file=args.output_file,
         plot_output=args.plot)




