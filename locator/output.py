# -*- coding: utf-8 -*-
from __future__ import print_function

from os.path import join as pjoin, split as psplit
from os import makedirs as mkdir
import os

import numpy as np
from h5py import File
from obspy import UTCDateTime
from platform import uname
import sys

# let YAML output fit into a IEEE 754 single format float
from locator.general_functions import calc_marginals_depdis, \
    calc_marginal_models

YAML_OUTPUT_SMALLEST_FLOAT_ABOVE_ZERO = 1.0e-37


def calc_origin_time(p, t_ref, tt_P):
    # Calculate origin time PDF using weighted histogram over P travel
    # times. This works because code uses P-arrival as time 0 elsewhere
    bol = p > 1e-2 * p.max(axis=None)
    origin_min = np.max((-1500, int(min(-tt_P[bol])) - 10))
    origin_max = int(max(-tt_P[bol])) + 10
    origin_pdf, origin_times = np.histogram(a=-tt_P.flatten(),
                                            weights=p.flatten(),
                                            bins=np.arange(origin_min, origin_max, 2),
                                            density=True)
    time_bin_mid = (origin_times[0:-1] + origin_times[1:]) / 2.
    origin_time_mean = np.sum(origin_pdf * (time_bin_mid)) / np.sum(origin_pdf) 
    time_bin_mid -= origin_time_mean
    origin_time_sum = UTCDateTime(origin_time_mean + t_ref)
    return origin_pdf, origin_time_sum, time_bin_mid


def write_result(file_out, model_output, modelset_name,
                 p, dep, dis, phase_list, tt_meas, sigma, freqs,
                 tt_P, t_ref, baz,
                 weights, model_names,
                 p_threshold=1e-2):

    depth_mean, dist_mean, p_depdis, p_depth, p_dist = \
        calc_marginals_depdis(dep, dis, p)

    origin_pdf, origin_time_sum, time_bin_mid = calc_origin_time(p, t_ref, tt_P)

    # Listify things for output to the YAML files
    pdf_depth_sum = _listify(dep, p_depth)
    pdf_dist_sum = _listify(dis, p_dist)
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
    with open(file_out, 'w') as f:
        _write_prob(f, pdf_depth_sum=pdf_depth_sum,
                   pdf_dist_sum=pdf_dist_sum)
        _write_prob_time(f, pdf_otime_sum=pdf_origin_sum)
        _write_ddscore(f, dep=dep.tolist(), dis=dis.tolist(),
                       ddscore=ddscore)

        _write_single(f, depth_sum=depth_mean, distance_sum=dist_mean,
                      depth_phase_count=int('pP' in phase_list) +
                                        int('sP' in phase_list) +
                                        int('pS' in phase_list) +
                                        int('sS' in phase_list),
                      origin_time_sum=origin_time_sum,
                      system_configuration=uname_string)

    if model_output:
        p_model = calc_marginal_models(dep, dis, p)
        p_model /= p_model.sum()
        fnam = 'model_misfits_%s.txt' % \
               (origin_time_sum.strftime(format='%y-%m-%dT%H%M'))

    _write_h5_output(p,
                     modelset_name=modelset_name,
                     depths=dep, distances=dis,
                     phase_list=phase_list,
                     tt_meas=tt_meas,
                     sigma=sigma,
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


def write_weight_file(p_model, model_names, prior_weights, fnam_out,
                      weight_lim=1e-5):
    model_weights = p_model / np.max(p_model)
    with open(fnam_out, 'w') as fid:
        imodel = 0
        for model_name, prior_weight in zip(model_names, prior_weights):
            if prior_weight > weight_lim:
                weight = model_weights[imodel] * prior_weight
                fid.write('%s %5.2f\n' % (model_name, weight))
                imodel += 1
            else:
                fid.write('%s %5.2f\n' % (model_name, prior_weight))


def write_model_misfits(p_model, model_names, fnam_out,
                        prior_weights):
    """
    Write probability for each model
    """
    with open(fnam_out, 'w') as f:
        f.write('model ID,  model name,    prior,  posterior\n')
        for imodel, (prior, post, name) in enumerate(
                zip(prior_weights, p_model, model_names)):
            f.write('%8d, %10s, %8.6f, %10.8f\n' %
                    (imodel, name, prior, post))


def _write_h5_output(p, modelset_name, depths, distances,
                     phase_list, tt_meas, sigma, t_ref, baz,
                     freqs, origin_time, model_names, weights):
    fnam = 'locator_output_%s.h5' % \
           (origin_time.strftime(format='%y-%m-%dT%H%M'))

    # Calculate model misfits
    models_p = calc_marginal_models(depths, distances, p)
    model_p_all = np.zeros(len(model_names))
    weight_bol = weights>1e-3
    model_p_all[weight_bol] = models_p


    with File(fnam, 'w') as f:
        f.create_dataset('p', data=p)
        f.create_dataset('modelset_name', data=modelset_name)
        f.create_dataset('depths', data=depths)
        f.create_dataset('distances', data=distances)
        f.create_dataset('phase_list', data=[n.encode("utf-8", "ignore")
                                             for n in phase_list])
        f.create_dataset('model_names', data=[n.encode("utf-8", "ignore")
                                              for n in model_names])
        f.create_dataset('weights', data=weights)
        f.create_dataset('sigma', data=sigma)
        f.create_dataset('tt_meas', data=tt_meas)
        f.create_dataset('freqs', data=freqs)
        f.create_dataset('t_ref', data=t_ref)
        f.create_dataset('backazimuth', data=baz)
        f.create_dataset('origin_time', data=float(origin_time))


def _write_axisem_file(h5_file, fnam_out):
    line_fmt = '      %8.0f %8.2f %8.2f %8.2f %9.1f %9.1f %8.2f %8.2f 1.0\n'
    with open(fnam_out, 'w') as f:
        name = psplit(('%s' % h5_file['model_name'].value))[-1]
        f.write('NAME            %s\n' % (name))
        f.write('ANELASTIC     true\n')
        f.write('ANISOTROPIC  false\n')
        f.write('UNITS            m\n')
        f.write('COLUMNS radius      rho      vpv      vsv       qmu       '
                'qka      vph      vsh eta\n')
        for ilayer in range(0, len(h5_file['mantle/vp'].value)):
            f.write(line_fmt % (
                h5_file['mantle/radius'][ilayer],
                h5_file['mantle/rho'][ilayer],
                h5_file['mantle/vp'][ilayer],
                h5_file['mantle/vs'][ilayer],
                h5_file['mantle/qmu'][ilayer],
                h5_file['mantle/qka'][ilayer],
                h5_file['mantle/vp'][ilayer],
                h5_file['mantle/vs'][ilayer]) )


def write_models_to_disk(p_model, files, model_names, tt_path,
                         weights, model_out_path='./models_location'):
    depths_target = np.arange(0.0, 200.0, 5.0)
    vp_sums = np.zeros_like(depths_target)
    vp_sums2 = np.zeros_like(depths_target)
    vs_sums = np.zeros_like(depths_target)
    vs_sums2 = np.zeros_like(depths_target)
    vp_all = np.zeros((p_model.shape[0], depths_target.shape[0]))
    vs_all = np.zeros((p_model.shape[0], depths_target.shape[0]))
    nmodel = 0

    if not os.path.exists(model_out_path):
        mkdir(model_out_path)

    weight_bol = weights>1e-3
    for imodel, (fnam, model_p, model_name) in \
            enumerate(zip(files, p_model, model_names[weight_bol])):

        with File(pjoin(tt_path, 'tt', fnam)) as f:
            fnam_out = pjoin(model_out_path, model_name)
            _write_axisem_file(h5_file=f, fnam_out=fnam_out)

            nmodel += 1
            radius = np.asarray(f['mantle/radius'])
            depths = (max(radius) - radius) * 1e-3

            vp_ipl = np.interp(xp=depths[::-1],
                               fp=f['mantle/vp'].value[::-1],
                               x=depths_target)
            vs_ipl = np.interp(xp=depths[::-1],
                               fp=f['mantle/vs'].value[::-1],
                               x=depths_target)
            vp_sums += vp_ipl * model_p
            vp_sums2 += vp_ipl**2 * model_p
            vs_sums += vs_ipl * model_p
            vs_sums2 += vs_ipl**2 * model_p

            vp_all[imodel, :] = vp_ipl
            vs_all[imodel, :] = vs_ipl

    fnam = pjoin(model_out_path, 'model_mean_sigma.txt')
    with open(fnam, 'w') as f:
        vp_mean = vp_sums
        vs_mean = vs_sums
        vp_sigma = np.sqrt(vp_sums2 - vp_mean**2)
        vs_sigma = np.sqrt(vs_sums2 - vs_mean**2)
        f.write('%6s, %8s, %8s, %8s, %8s\n' %
                ('depth', 'vp_mean', 'vp_sig', 'vs_mean', 'vs_sig'))
        for idepth in range(0, len(depths_target)):
            f.write('%6.1f, %8.2f, %8.2f, %8.2f, %8.2f\n' %
                    (depths_target[idepth],
                     vp_mean[idepth], vp_sigma[idepth],
                     vs_mean[idepth], vs_sigma[idepth]))

    fnam_pp_out = pjoin(model_out_path, 'model_prior_post.txt')
    write_model_misfits(p_model, fnam_out=fnam_pp_out,
                        model_names=model_names[weight_bol],
                        prior_weights=weights[weight_bol] / weights.sum())

    fnam_weight_out = pjoin(model_out_path, 'model_weights_new.txt')

    write_weight_file(p_model, model_names=model_names,
                      fnam_out=fnam_weight_out,
                      prior_weights=weights)

    return vp_all, vs_all
