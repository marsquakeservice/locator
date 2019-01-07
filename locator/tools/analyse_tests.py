#!/usr/bin/env python
"""

"""
__author__ = "Simon Stähler"

import numpy as np
from os.path import join as pjoin
import glob
from yaml import load
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

files = glob.glob('tests/event*')
files.sort()
fig, ax = plt.subplots(1,2)
for ifile, file in enumerate(files):
    depth_true = float(file[22:25])
    dist_true = float(file[31:38])
    with open(pjoin(file, 'locator_output.yml')) as f:
        data_yaml = load(f)
        pdf = np.asarray(data_yaml['pdf_dist_sum']['probabilities'])

        pdf[:,1] /= max(pdf[:,1])
        cdf = np.cumsum(pdf[:, 1])
        ax[1].scatter(x=pdf[:,0] - dist_true,
                      y=dist_true + np.zeros(pdf.shape[0]),
                      c=pdf[:,1] / max(pdf[:,1]), vmin=0, vmax=1)

        distribution = interp1d(cdf,
                                pdf[:,0] - dist_true,
                                bounds_error=False,
                                fill_value='extrapolate')
        ax[0].plot((distribution(0.1), distribution(0.9)),
                   (dist_true, dist_true), 'k', lw=2.5)
        ax[0].plot((distribution(0.4), distribution(0.6)),
                 (dist_true, dist_true), 'r', lw=2.0)
        ax[0].plot((-9,0,9), (180,0,180), 'b--')
plt.show()