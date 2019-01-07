#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Simon Stähler"

import numpy as np
from obspy import UTCDateTime
from obspy.taup import TauPyModel
import argparse
from os.path import join as pjoin
from os import makedirs
from tqdm import tqdm

def define_arguments():
    parser = argparse.ArgumentParser(
        description='Create test file with surface and body wave traveltimes')
    parser.add_argument('ievent',
                        help='event number in surface wave traveltime file')
    parser.add_argument('depth', type=float,
                       help='event depth in kilometer')
    return parser.parse_args()


def create_input(phases, baz, outdir):
    makedirs(outdir, exist_ok=True)
    with open(pjoin(outdir, 'locator_input.yml'), 'w') as f:
        f.write('velocity_model:             MQS_Ops\n')
        f.write('velocity_model_uncertainty: 1.5\n')
        f.write('backazimuth:\n')
        f.write('    value: %5.1f\n' % baz)
        f.write('phases:\n\n')
        for phase in phases:
            f.write(' -\n')
            f.write('    code: %s\n' % phase['code'])
            f.write('    datetime: %s\n' % phase['datetime'])
            f.write('    uncertainty_lower: %4.1f\n' % (phase['sigma'] / 2))
            f.write('    uncertainty_upper: %4.1f\n' % (phase['sigma'] / 2))
            if phase['code'] in ['R1', 'G1']:
                f.write('    frequency: %s\n' % phase['frequency'])


def create_event(ievent, depth, outdir='.'):
    model = TauPyModel('./locator/data/tests/MSS_ORT/MQSORT_TAY.npz')
    origin_time = UTCDateTime('2019-01-01T00:00:00')
    phases = []
    for period in [14, 20., 40., 60., 113.]:
        surface_wave_times = np.loadtxt('./locator/data/tests/MSS_ORT/ttr_%03d.txt' % period)
        time = surface_wave_times[ievent]
        phase = {'code': 'R1',
                 'datetime': '%s' % (origin_time + time[2]),
                 'frequency': 1. / period,
                 'sigma': 80.
                 }
        phases.append(phase)
    baz = time[1]
    dist = time[0]
    for phase_name in ['P', 'S', 'PP', 'pP']:
        arr = model.get_travel_times(source_depth_in_km=depth,
                                     distance_in_degree=dist,
                                     phase_list=[phase_name])
        if len(arr) > 0:
            phase = {'code': phase_name,
                     'datetime': '%s' % (origin_time + arr[0].time),
                     'sigma': 5.
                     }
            phases.append(phase)
    create_input(phases, baz, outdir)
    return dist


if __name__ == '__main__':
    args = define_arguments()
    if type(args.ievent) == int:
        create_event(args.ievent, args.depth)
    elif args.ievent == 'all':
        for i in tqdm(range(0, 200)):
            depth = np.random.rand((1))[0] * 50
            dist = create_event(i, depth)
            create_event(i, depth,
                         'tests/event_%03d_depth_%03d_dist_%5.1f' %
                         (i, depth, dist))
