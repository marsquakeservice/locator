# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from h5py import File
from obspy import UTCDateTime
from platform import uname
import sys

# let YAML output fit into a IEEE 754 single format float
YAML_OUTPUT_SMALLEST_FLOAT_ABOVE_ZERO = 1.0e-37


def calc_origin_time(p, t_ref, tt_P):
    # Calculate origin time PDF using weighted histogram over P travel
    # times. This works because code uses P-arrival as time 0 elsewhere
    bol = p > 1e-3 * p.max(axis=None)
    origin_min = int(min(-tt_P[bol])) - 10
    origin_max = int(max(-tt_P[bol])) + 10
    origin_pdf, origin_times = np.histogram(a=-tt_P.flatten(),
                                            weights=p.flatten(),
                                            bins=np.arange(origin_min, origin_max, 2),
                                            density=True)
    time_bin_mid = (origin_times[0:-1] + origin_times[1:]) / 2.
    origin_time_sum = UTCDateTime(np.sum(origin_pdf * (time_bin_mid))
                                  / np.sum(origin_pdf) + t_ref)
    return origin_pdf, origin_time_sum, time_bin_mid


def write_result(file_out, model_output, modelset_name,
                 p, dep, dis, phase_list, tt_meas, freqs,
                 tt_P, t_ref, baz,
                 weights, model_names,
                 p_threshold=1e-3):
    p_dist = np.sum(p, axis=(0, 1))
    p_depth = np.sum(p, axis=(0, 2))
    p_depdis = np.sum(p, axis=0)

    pdf_depth_sum = _listify(dep, p_depth)
    depth_sum = np.sum(dep * p_depth) / np.sum(p_depth)

    pdf_dist_sum = _listify(dis, p_dist)
    dist_sum = np.sum(dis * p_dist) / np.sum(p_dist)

    origin_pdf, origin_time_sum, time_bin_mid = calc_origin_time(p, t_ref, tt_P)
    pdf_origin_sum = _listify(time_bin_mid, origin_pdf)

    # Create depth-distance score list
    p_depdis /= p_depdis.max(axis=None)
    p_depdis = p_depdis.flatten()
    p_depdis[p_depdis < YAML_OUTPUT_SMALLEST_FLOAT_ABOVE_ZERO] = 0.0
    bol = p_depdis > p_threshold
    depdep, disdis = np.meshgrid(dep, dis)
    ddscore = np.zeros((3, sum(bol)))
    ddscore[0, :] = depdep.flatten()[bol]
    ddscore[1, :] = disdis.flatten()[bol]
    ddscore[2, :] = p_depdis[bol]
    ddscore = ddscore.T.tolist()


    # Get unamy stuff TODO: Python 2 version
    if sys.version_info.major == 3:
        uname_dict = uname()._asdict()
        uname_fmt = "\"uname: ('{system}', '{node}', '{release}', '{version}'" + \
            ", '{machine}', '{processor}')\""
        uname_string = uname_fmt.format(**uname_dict)
    else:
        uname_string = '\"%s\"' % sys.version.format('%s')

    # Write serialized YAML output for the GUI.
    # TODO: Might be something wrong with the origin times
    with open(file_out, 'w') as f:
        _write_prob(f, pdf_depth_sum=pdf_depth_sum,
                   pdf_dist_sum=pdf_dist_sum)
        _write_prob_time(f, pdf_otime_sum=pdf_origin_sum)
        _write_ddscore(f, dep=dep.tolist(), dis=dis.tolist(),
                       ddscore=ddscore)

        _write_single(f, depth_sum=depth_sum, distance_sum=dist_sum,
                      depth_phase_count=int('pP' in phase_list),
                      origin_time_sum=origin_time_sum,
                      system_configuration=uname_string)

    if model_output:
        _write_model_misfits(p, origin_time_sum)
        _write_weight_file(p, model_names=model_names,
                           prior_weights=weights,
                           origin_time_sum=origin_time_sum)

    _write_h5_output(p, modelset_name=modelset_name,
                     depths=dep, distances=dis,
                     phase_list=phase_list,
                     tt_meas=tt_meas,
                     freqs=freqs,
                     t_ref=t_ref,
                     baz=baz,
                     origin_time=origin_time_sum,
                     weights=weights,
                     model_names=model_names)


def _listify(val, p_val):
    """
    Create list of list of 2 numpy arrays of same length.
    This list of list can be printed nicely
    :param val: np.ndarray
    :param p_val: np.ndarray
    :return: list
    """

    p_val[p_val < YAML_OUTPUT_SMALLEST_FLOAT_ABOVE_ZERO] = 0.0
    pdf_sum = np.zeros((len(val), 2))
    pdf_sum[:, 0] = val
    pdf_sum[:, 1] = p_val
    pdf_sum = pdf_sum.tolist()
    return pdf_sum


def _write_prob(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: \n' % key)
        f.write('  probabilities: ')
        print(value, file=f)
        f.write('\n\n')


def _write_ddscore(f, dep, dis, ddscore):
    f.write('%s: \n' % 'depth_distance_score')
    f.write('  depths: ')
    print(dep, file=f)
    f.write('  distances: ')
    print(dis, file=f)
    f.write('  scores: ')
    print(ddscore, file=f)
    f.write('\n\n')


def _write_prob_time(f, **kwargs):
    for key, value in kwargs.items():
        f.write('%s: \n' % str(key))
        f.write('  probabilities: ')
        print(value, file=f)
        f.write('\n\n')


def _write_single(f, **kwargs):
    for key, value in kwargs.items():
        if type(value)==int:
            f.write('%s: %d\n\n' % (key, value))
        elif type(value) == float:
            f.write('%s: %9.4f\n\n' % (key, value))
        else:
            f.write('%s: %s\n\n' % (key, value))


def _write_weight_file(p, model_names, prior_weights, origin_time_sum, weight_lim=1e-5):
    fnam = 'model_weights_%s.txt' % \
           (origin_time_sum.strftime(format='%y-%m-%dT%H%M'))
    p_model = np.sum(p, axis=(1, 2))
    model_weights = p_model / np.max(p_model)
    with open(fnam, 'w') as fid:
        imodel = 0
        for model_name, prior_weight in zip(model_names, prior_weights):
            if prior_weight > weight_lim:
                weight = model_weights[imodel] * prior_weight
                fid.write('%s %5.2f\n' % (model_name, weight))
                imodel += 1
            else:
                fid.write('%s %5.2f\n' % (model_name, prior_weight))


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
            # if model > 0.5:
            #     np.savetxt('model_%03d.txt' % imodel,
            #                np.asarray([f['mantle/radius'], f['mantle/vp'],
            #                            f['mantle/vs'], f['mantle/rho']]).T,
            #                fmt = '%9.2e, %9.2e, %9.2e, %9.2e', header = 'vp, vs, rho')


def _write_h5_output(p, modelset_name, depths, distances,
                     phase_list, tt_meas, t_ref, baz,
                     freqs, origin_time, model_names, weights):
    fnam = 'locator_output_%s.h5' % \
           (origin_time.strftime(format='%y-%m-%dT%H%M'))

    # Calculate model misfits
    models_p = np.sum(p, axis=(1, 2))
    model_p_all = np.zeros(len(model_names))
    weight_bol = weights>1e-3
    model_p_all[weight_bol] = models_p


    with File(fnam, 'w') as f:
        f.create_dataset('p', data=p)
        f.create_dataset('modelset_name', data=modelset_name)
        f.create_dataset('depths', data=depths)
        f.create_dataset('distances', data=distances)
        f.create_dataset('phase_list', data=[n.encode("utf-8", "ignore") for n in phase_list])
        f.create_dataset('model_names', data=[n.encode("utf-8", "ignore") for n in model_names])
        f.create_dataset('weights', data=weights)
        f.create_dataset('tt_meas', data=tt_meas)
        f.create_dataset('freqs', data=freqs)
        f.create_dataset('t_ref', data=t_ref)
        f.create_dataset('backazimuth', data=baz)
        f.create_dataset('origin_time', data=float(origin_time))
