#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read results of previous locator run and create model output:
  - AxiSEM files for each model
  - Table with prior and posterior weights
"""
__author__ = "Simon Staehler"

import numpy as np
import os
from argparse import ArgumentParser

from locator.general_functions import calc_marginal_models
from locator.input import read_h5_locator_output, read_model_list
from locator.output import write_models_to_disk, write_weight_file
from locator.graphics import plot_model_density

def define_arguments():
    helptext = 'Create model output files based on previous locator run'
    parser = ArgumentParser(description=helptext)

    helptext = "Locator HDF5 output file to read"
    parser.add_argument('input_file', help=helptext)

    helptext = "Output directory name"
    parser.add_argument('output_path', help=helptext)

    helptext = "Path to model list file"
    parser.add_argument('--model_path', help=helptext,
                        default=None)

    return parser.parse_args()


def main(input_file, output_path, model_path):

    p, dep, dis, weights, modelset, model_names = read_h5_locator_output(
        input_file)

    if model_path is None:
        model_path = os.path.join(os.environ['SINGLESTATION'],
                                  'data', 'bodywave', modelset)

    fnam_models=os.path.join(model_path, '%s.models' % modelset)
    fnam_weights=os.path.join(model_path, '%s.weights' % modelset)
    filenames, weights_all, model_names, weights = \
        read_model_list(fnam_models=fnam_models,
                        fnam_weights=fnam_weights)

    p_model = calc_marginal_models(dep=dep, dis=dis, p=p)
    p_model /= np.sum(p_model)


    vp_all, vs_all = write_models_to_disk(p_model=p_model,
                         files=filenames, tt_path=model_path,
                         weights=weights,
                         model_names=model_names,
                         model_out_path=output_path)

    plot_model_density(p_model=p_model,
                       prior=weights[weights>1e-5],
                       vp_all=vp_all,
                       vs_all=vs_all)

if __name__ == '__main__':
    args = define_arguments()
    main(input_file=args.input_file,
         output_path=args.output_path,
         model_path=args.model_path)

