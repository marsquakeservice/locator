from __future__ import print_function
import numpy as np
from obspy import UTCDateTime

def write_result(file_out, p, dep, dis, tt_P, t_ref):
    p_dist = np.sum(p, axis=(0, 1))
    p_depth = np.sum(p, axis=(0, 2))

    pdf_depth_sum = []
    for idepth, d in enumerate(dep):
        pdf_depth_sum.append([d, p_depth[idepth]])
    depth_sum = np.sum(dep * p_depth) / np.sum(p_depth)

    pdf_dist_sum = []
    for idist, d in enumerate(dis):
        pdf_dist_sum.append([d, p_dist[idist]])
    dist_sum = np.sum(dis * p_dist) / np.sum(p_dist)

    origin_pdf, origin_times = np.histogram(a=-tt_P.flatten(),
                                            weights=p.flatten(),
                                            bins=np.linspace(-1200, 0, 120),
                                            density=True)
    pdf_origin_sum = []
    time_bin_mid = (origin_times[0:-1] + origin_times[1:]) / 2.
    for itime, time in enumerate(time_bin_mid):
        pdf_origin_sum.append([time,
                               origin_pdf[itime]])
    origin_time_sum = UTCDateTime(t_ref) + np.sum(origin_pdf * time_bin_mid)

    with open(file_out, 'w') as f:
        _write_prob(f, pdf_depth_sum=pdf_depth_sum,
                   pdf_dist_sum=pdf_dist_sum,
                   pdf_otime_sum=pdf_origin_sum)
        _write_single(f, depth_sum=depth_sum, dist_sum=dist_sum)
        f.write('%s: %s\n \n' % ('origin_time_sum', origin_time_sum))

    _write_model(p, origin_time_sum)


def _write_prob(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: \n' % key)
        f.write('  probabilities: ')
        print(value, file=f)
        f.write('\n\n')


def _write_single(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: %f\n\n' % (key, value))


def _write_model(p, origin_time_sum):
    """

    :type origin_time_sum: obspy.UTCDateTime
    """
    fnam = 'model_misfits_%s.txt' % \
        (origin_time_sum.strftime(format='%y-%m-%dT%H%M'))
    p_model = np.sum(p, axis=(1, 2))
    with open(fnam, 'w') as f:
        for imodel, model in enumerate(p_model):
            f.write('%5d, %8.3e\n' % (imodel, model))
