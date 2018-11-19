from __future__ import print_function
import numpy as np
from h5py import File

def calc_origin_time(p, t_ref, tt_P):
    # Calculate origin time PDF using weighted histogram over P travel
    # times. This works because code uses P-arrival as time 0 elsewhere
    origin_pdf, origin_times = np.histogram(a=-tt_P.flatten(),
                                            weights=p.flatten(),
                                            bins=np.linspace(-1200, 0, 120),
                                            density=True)
    time_bin_mid = (origin_times[0:-1] + origin_times[1:]) / 2.
    origin_time_sum = UTCDateTime(np.sum(origin_pdf * (time_bin_mid))
                                  / np.sum(origin_pdf) + t_ref)
    return origin_pdf, origin_time_sum, time_bin_mid


def write_result(file_out, p, dep, dis, phase_list, tt_meas, tt_P, t_ref):
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

    origin_pdf, origin_time_sum, time_bin_mid = calc_origin_time(p, t_ref, tt_P)
    pdf_origin_sum = []
    for itime, time in enumerate(time_bin_mid):
        pdf_origin_sum.append([t_ref + time,
                               origin_pdf[itime]])

    # Write serialized YAML output for the GUI.
    # TODO: Might be something wrong with the origin times
    with open(file_out, 'w') as f:
        _write_prob(f, pdf_depth_sum=pdf_depth_sum,
                   pdf_dist_sum=pdf_dist_sum)
        _write_prob_time(f, pdf_otime_sum=pdf_origin_sum)
        _write_single(f, depth_sum=depth_sum, dist_sum=dist_sum)
        f.write('%s: %s\n \n' % ('origin_time_sum', origin_time_sum))

    _write_model_misfits(p, origin_time_sum)

    _write_h5_output(p, depths=dep, distances=dis,
                     phase_list=phase_list,
                     tt_meas=tt_meas,
                     t_ref=t_ref,
                     origin_time=origin_time_sum)


def _write_prob(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: \n' % key)
        f.write('  probabilities: ')
        print(value, file=f)
        f.write('\n\n')


def _write_prob_time(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: \n' % str(key))
        f.write('  probabilities: ')
        print(value, file=f)
        f.write('\n\n')


def _write_single(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: %f\n\n' % (key, value))


def _write_model_misfits(p, origin_time_sum):
    """
    Write probability for each model
    :type origin_time_sum: obspy.UTCDateTime
    """
    fnam = 'model_misfits_%s.txt' % \
        (origin_time_sum.strftime(format='%y-%m-%dT%H%M'))
    p_model = np.sum(p, axis=(1, 2))
    with open(fnam, 'w') as f:
        for imodel, model in enumerate(p_model):
            f.write('%5d, %8.3e\n' % (imodel, model))

def _write_h5_output(p, depths, distances, phase_list, tt_meas, t_ref, origin_time):
    fnam = 'locator_output_%s.h5' % \
           (origin_time.strftime(format='%y-%m-%dT%H%M'))

    with File(fnam, 'w') as f:
        f.create_dataset('p', data=p)
        f.create_dataset('depths', data=depths)
        f.create_dataset('distances', data=distances)
        f.create_dataset('tt_meas', data=tt_meas)
        f.create_dataset('t_ref', data=t_ref)
        f.create_dataset('phase_list', data=[n.encode("utf-8", "ignore") for n in phase_list])
