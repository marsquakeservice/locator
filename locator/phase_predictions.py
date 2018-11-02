# -*- coding: utf-8 -*-


# TODO: hardcoded to MSS event, need to include argument
#       parsing for the waveform directory etc
import numpy as np
import matplotlib.pyplot as plt
from read_models import load_tt
import glob
from h5py import File
from obspy import read, UTCDateTime


model_path = '../tt/*.h5'
model_list = glob.glob(model_path)
waveform_path = '/opt/InSight_MSSROT/mss_event.mseed'

t_post = 3600
t_pre = 50

def load_H5(fnam):
    with File(fnam, 'r') as f:
        p = f['p'].value
        depths = f['depths'].value
        distances = f['distances'].value
        phase_list = f['phase_list'].value
        t_ref = f['t_ref'].value
    return p, depths, distances, phase_list, t_ref


def plot_cwf(tr, ax, t_ref=0, fmin=1./30, fmax=1./2):
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


phase_list_prediction = ['P', 'PP', 'PPP', 'PcP', 'PKKP',
                         'S', 'SS', 'SSS', 'ScS', 'SKKS'
                         ]
fnam_modelmisfit = 'model_misfits_19-01-03T1500.txt'
fnam_locatoroutput = 'locator_output_19-01-03T1500.h5'

tt = load_tt(model_list, phase_list=phase_list_prediction,
             freqs=np.ones(len(phase_list_prediction)),
             backazimuth=0.0)[0]


p, depths, distances, phase_list, t_ref = load_H5(fnam_locatoroutput)

data = np.loadtxt(fnam_modelmisfit, delimiter=',')

fig, ax = plt.subplots(nrows=4, ncols=1,
                       figsize=(10, 10), sharex='col')

for iphase, phase in enumerate(phase_list_prediction):
    y, x = np.histogram(tt[:,:,:,iphase].flatten(), weights=p.flatten(),
                        bins=np.arange(-t_pre, t_post, 2),
                        density=False)
    ax[3].plot(x[1:], y / np.max(np.sqrt(y)), label=phase)
ax[3].set_ylim(0, 1)

ax[3].legend(ncol=2)
st = read(waveform_path)
st.differentiate()
st.filter('lowpass', freq=1./2., zerophase=True)
st.filter('highpass', freq=1./30., zerophase=True)
st.trim(starttime=UTCDateTime(t_ref - t_pre),
        endtime=UTCDateTime(t_ref + t_post))

for i in range(0, 3):
    # ax[i].plot(st[i].times() + t_ref - t_pre, st[i].data)
    plot_cwf(st[i], ax[i], t_ref=-t_pre, #t_ref - t_pre,
             fmax=1)
    ax[i].grid(axis='x')
    ax[i].set_ylabel('period / seconds')
ax[3].set_xlim(-t_pre, t_post)
ax[3].set_xlabel('time after P / seconds')
plt.tight_layout()

fig.savefig('phase_prediction_spec_long.png', dpi=200)
ax[3].set_xlim(-t_pre, 1000)
fig.savefig('phase_prediction_spec.png', dpi=200)

st.integrate()
st.filter('highpass', freq=1./30., zerophase=True)
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
plt.show()
plt.close()

