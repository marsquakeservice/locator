#!/usr/bin/env python
"""

"""
__author__ = "Simon Stähler"

import numpy as np
from os.path import join as pjoin
import glob
from yaml import load
import matplotlib.pyplot as plt

files = glob.glob('event*')
files.sort()

for file in files:
    depth_true = float(file[16:19])
    dist_true = float(file[25:32])
    print(file[25:32])
    print(dist_true)

    with open(pjoin(file, 'locator_output.yml')) as f:
        data_yaml = load(f)
        pdf = np.asarray(data_yaml['pdf_dist_sum']['probabilities'])
        plt.plot(pdf[:,0], pdf[:,1] / max(pdf[:,1]))
        plt.axvline(dist_true)
        plt.show()