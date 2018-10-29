import numpy as np
from matplotlib import pyplot as plt


def plot(p, dis, dep):
    nmodel, ndepth, ndist = p.shape

    # Depth-distance matrix
    depthdist = np.sum(p, axis=(0)) / nmodel
    plt.contourf(dis, dep, depthdist)
    plt.colorbar()
    plt.xlabel('distance')
    plt.ylabel('depth')
    plt.tight_layout()
    plt.savefig('depth_distance.png')
    plt.close('all')

    # Model likelihood plot
    plt.plot(np.sum(p, axis=(1, 2)) / ndepth / ndist, '.')
    plt.xlabel('Model index')
    plt.ylabel('likelihood')
    plt.tight_layout()
    plt.savefig('model_likelihood.png')
    plt.close('all')


def plot_phases(tt, p, phase_list, tt_meas, sigma):
    nphase = len(phase_list)
    fig, axs = plt.subplots(nrows=nphase, ncols=1)

    for iax, ax in enumerate(axs):
        phase_mean = np.sum(tt[:, :, :, iax] * p, axis=None) / np.sum(p, axis=None)
        ax.hist(tt[:, :, :, iax].flatten(),
                weights=p[:, :, :].flatten(),
                bins=np.linspace(phase_mean - 50, phase_mean + 50, 100))
        ax.axvline(x=tt_meas[iax], c='r')
        ax.axvline(x=tt_meas[iax] - sigma[iax], c='r', ls='--')
        ax.axvline(x=tt_meas[iax] + sigma[iax], c='r', ls='--')
        ax.text(x=0.05, y=0.5, s=phase_list[iax],
                fontsize=14, weight='bold',
                transform = ax.transAxes)
    plt.tight_layout()
    plt.savefig('phase_misfits.png')
    plt.close('all')