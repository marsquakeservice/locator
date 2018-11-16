#!/usr/bin/env python
# -*- coding: utf-8 -*-

from h5py import File
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    helptext = 'Compare H5 output files to given numerical precision'
    parser = ArgumentParser(helptext)

    helptext = "HDF5 file 1"
    parser.add_argument('input_1', help=helptext)
    helptext = "HDF5 file 2"
    parser.add_argument('input_2', help=helptext)
    helptext = "desired relative precision"
    parser.add_argument('-p', '--precision', type=float, default=1e-4)

    args = parser.parse_args()

    f1 = File(args.input_1, 'r')
    f2 = File(args.input_2, 'r')

    np.testing.assert_allclose(actual=f1['p'], desired=f2['p'], rtol=args.precision)

    print('all fine')
