#!/usr/bin/env python
"""
Compare weights returned by multiple runs
"""
__author__ = "Simon Stähler"

import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from argparse import ArgumentParser

def define_arguments():
    helptext = 'Compare the model weighting returned by two or more runs of the locator'
    parser = ArgumentParser(description=helptext)

    helptext = "Output files of the locator"
    parser.add_argument('input_file', nargs='+', help=helptext)

    return parser.parse_args()

if __name__ == '__main__':
    args = define_arguments()
    p_models = []
    for fnam in args.input_file:
        with File(fnam, 'r') as f:
            p_model = np.sum(f['p'], axis=(1, 2))
            p_models.append(p_model)

    plt.scatter(x=p_models[0], y=p_models[1])
    plt.show()


