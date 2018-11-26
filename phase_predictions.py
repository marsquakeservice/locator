# -*- coding: utf-8 -*-


# TODO: hardcoded to MSS event, need to include argument
#       parsing for the waveform directory etc
import numpy as np
import matplotlib.pyplot as plt
from locator.input import load_tt, read_model_list
from h5py import File
from obspy import read, UTCDateTime, Stream, read_inventory
from argparse import ArgumentParser
from os.path import join as pjoin
from os import environ as env

t_post = 10800
t_pre = 50
fmin = 1./150.

def define_arguments():
    helptext = 'Predict phase arrivals based on locator solution'
    parser = ArgumentParser(description=helptext)

    helptext = "Output file of locator"
    parser.add_argument('locator_output', help=helptext)

    return parser.parse_args()



def load_H5(fnam):
    with File(fnam, 'r') as f:
        model_name = f['model_name'].value
        p = f['p'].value
        depths = f['depths'].value
        distances = f['distances'].value
        phase_list = f['phase_list'].value
        t_ref = f['t_ref'].value
        baz = f['backazimuth'].value
    return p, model_name, depths, distances, phase_list, t_ref, baz


def plot_cwf(tr, ax, t_ref=0, fmin=1./50, fmax=1./2):
    from obspy.signal.tf_misfit import cwt
    npts = tr.stats.npts
    dt = tr.stats.delta
    t = np.linspace(0, dt * npts, npts)

    scalogram = abs(cwt(tr.data, dt, w0=6,
                        fmin=fmin, fmax=fmax, nf=100))

    x, y = np.meshgrid(t + t_ref,
                       1./np.logspace(np.log10(fmin),
                                      np.log10(fmax),
                                      scalogram.shape[0]))

    m = ax.pcolormesh(x, y, np.log10((scalogram)) * 10.,
                      cmap='plasma',
                      vmin=-100, vmax=-82)


def main(args):
    phase_list_prediction = ['P', 'PP', 'PPP', 'PcP', 'PKKP',
                             'S', 'SS', 'SSS', 'ScS', 'SKKS'
                             ]
    args = define_arguments()
    fnam_locatoroutput = args.locator_output
    p, model_name, depths, distances, phase_list, t_ref, baz = load_H5(fnam_locatoroutput)
    t0 = UTCDateTime(t_ref)
    stat_net, stat_station = env['STATION'].split('.')
    waveform_dir = pjoin(env['WAVEFORM_DIR'],
                         'waveform',
                         str(t0.year),
                         stat_net, stat_station)
    tt_path = pjoin(env['SINGLESTATION'],
                    'data', 'bodywave',
                    model_name)
    model_path = pjoin(tt_path,
                       '%s.models' % model_name)
    weight_path = pjoin(tt_path,
                        '%s.weights' % model_name)
    files, weights = read_model_list(model_path, weight_path)

    # Load body waves
    tt = load_tt(files=files, tt_path=tt_path, phase_list=phase_list_prediction,
                 freqs=np.ones(len(phase_list_prediction)),
                 backazimuth=baz)[0]

    nfreq = 21
    p0 = 5
    freqs = [1.]
    phase_list = ['P']
    for i in range(nfreq):
        freqs.append(1./p0 / 2.**(i/4.))
        phase_list.append('R1')

    tt_r = load_tt(files=files, tt_path=tt_path, phase_list=phase_list,
                   freqs=freqs,
                   backazimuth=baz)[0]
    phase_list = ['P']
    for i in range(nfreq):
        phase_list.append('G1')
    tt_g = load_tt(files=files, tt_path=tt_path, phase_list=phase_list,
                   freqs=freqs,
                   backazimuth=baz)[0]
    st = read_waveform(waveform_dir, t0, stat=stat_station, net=stat_net, baz=baz)


    fig, ax = plt.subplots(nrows=4, ncols=1,
                           figsize=(10, 10), sharex='col')
    for iphase, phase in enumerate(phase_list_prediction):
        y, x = np.histogram(tt[:, :, :, iphase].flatten(), weights=p.flatten(),
                            bins=np.arange(-t_pre, t_post, 2),
                            density=False)
        ax[3].plot(x[1:], y / np.max(np.sqrt(y)), label=phase)
    ax[3].set_ylim(0, 1)
    ax[3].legend(ncol=2)
    for i in range(0, 3):
        plot_cwf(st[i], ax[i], t_ref=-t_pre,
                 fmax=1, fmin=fmin)
        ax[i].grid(axis='x')
        ax[i].set_ylabel('period / seconds')


    tt_g_res = tt_g[:,:,:,1:].reshape((-1, nfreq))
    p_flat = p.reshape(tt_g_res.shape[0])
    p_flat /= p_flat.max()
    bol = p_flat > 0.01
    ax[2].plot(tt_g_res[bol, :].T - t_pre, 1./np.array(freqs[1:]),
               zorder=100, color='k', alpha=0.1)
    # for icomb in bol: # range(0, tt_r_res.shape[0]):
    #     ax[0].plot(tt_r_res[icomb, :], 1./np.array(freqs[1:]), zorder=100)
    ax[3].set_xlim(-t_pre, t_post)
    ax[3].set_xlabel('time after P / seconds')
    for a in ax[0:3]:
        a.set_ylim(0, 1./fmin)
    plt.tight_layout()
    fig.savefig('phase_prediction_spec_long.png', dpi=200)
    ax[3].set_xlim(-t_pre, 1000)
    fig.savefig('phase_prediction_spec.png', dpi=200)


    st.integrate()
    st.filter('highpass', freq=fmin, zerophase=True)
    for i in range(0, 3):
        ax[i].clear()
        ax[i].plot(st[i].times() - t_pre, st[i].data)
        ax[i].grid(axis='x')
    ax[3].set_xlim(-t_pre, t_post)
    fig.savefig('phase_prediction_seis_long.png', dpi=200)
    ax[3].set_xlim(-t_pre, 1000)
    for a in ax[0:3]:
        a.set_ylim(-1e-8, 1e-8)
    fig.savefig('phase_prediction_seis.png', dpi=200)
    plt.close()


def read_waveform(waveform_dir, t_ref, stat, net, baz,
                  channels=['MHU', 'MHV', 'MHW'], location='02'):
    st = Stream()
    inv = read_inventory('7J.SYNT3.seed')
    t_end = t_ref + t_post
    t_start = t_ref - t_pre
    for channel in channels:
        fnam = (pjoin(waveform_dir,
                      channel+'.D',
                      '%s.%s.%s.%s.D.%04d.%03d' %
                      (net, stat, location, channel, t_ref.year, t_ref.julday)))
        st += read(fnam, starttime=t_start-3600, endtime=t_end+3600)
        fnam = (pjoin(waveform_dir,
                      channel+'.D',
                      '%s.%s.%s.%s.D.%04d.%03d' %
                      (net, stat, location, channel, t_ref.year, t_ref.julday+1)))
        st += read(fnam, starttime=t_start-3600, endtime=t_end+3600)
    st.merge()
    st.remove_response(inv, pre_filt=(fmin*0.8, fmin, 1./1.5, 1./2))
    st.differentiate()
    #st.filter('lowpass', freq=1. / 2., zerophase=True)
    #st.filter('highpass', freq=fmin, zerophase=True)
    st.trim(starttime=t_start, endtime=t_end)
    #st_ZNE = st.rotate(method='->ZNE', inventory=inv)
    st_ZNE = st._rotate_specific_channels_to_zne(network=net, station=stat,
                                                 location='02',
                                                 channels=['MHU', 'MHV', 'MHW'],
                                                 inventory=inv)
    st_ZRT = st_ZNE.rotate(method='NE->RT', back_azimuth=baz)
    return st_ZRT


if __name__ == '__main__':
    args = define_arguments()
    main(args)

