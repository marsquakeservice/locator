# -*- coding: utf-8 -*-
import numpy as np
from os.path import join as pjoin
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_2D_with_marginals(x, y, z, x_aux=None, y_aux=None,
                           xlabel=None, ylabel=None, xunit='', yunit='',
                           scatter=False, **kwargs):
    fig = plt.figure(**kwargs)

    levels = np.sqrt(np.max(z)) * np.asarray((0.05, 0.2, 0.5, 0.75, 1.0))
    z_int = np.sum(z, axis=None)

    # Central 2D plot
    ax_2D = fig.add_axes([0.10, 0.10, 0.76, 0.72], label='2D')
    ax_x = fig.add_axes([0.10, 0.82, 0.76, 0.12], label='y_marginal', sharex=ax_2D)
    plt.axis('off')
    ax_y = fig.add_axes([0.86, 0.10, 0.11, 0.72], label='x_marginal', sharey=ax_2D)
    plt.axis('off')
    ax_cb = fig.add_axes([0.86, 0.84, 0.017, 0.12], label='colorbar')

    ax_x.get_xaxis().set_visible(False)
    ax_x.get_yaxis().set_visible(False)

    marg_x = np.nansum(z, axis=0)
    marg_x /= np.nanmax(marg_x)
    ax_x.plot(x, marg_x)
    marg_y = np.nansum(z, axis=1)
    marg_y /= np.nanmax(marg_y)
    l_p, = ax_y.plot(marg_y, y)
    if scatter:
        xx, yy = np.meshgrid(x, y)
        cf = ax_2D.scatter(xx, yy, c=np.sqrt(z), cmap='afmhot_r', marker='+')
    else:
        cf = ax_2D.contourf(x, y, np.sqrt(z), cmap='afmhot_r', levels=levels)

    ax_x.set_xlim(x[0], x[-1])
    ax_x.set_ylim(0, 1)
    ax_y.set_ylim(y[0], y[-1])
    ax_y.set_xlim(0, 1)

    # Calculate mean values and mark them
    mean_x = np.sum(marg_x * x / z_int)
    ax_x.axvline(x=mean_x, linestyle='dashed', color='black')
    ax_x.text(x=mean_x, y=max(marg_x)*1.1, s='%4.1f %s' % (mean_x, xunit),
              horizontalalignment='center')
    mean_y = np.sum(marg_y * y / z_int)
    ax_y.axhline(y=mean_y, linestyle='dashed', color='black')
    ax_y.text(x=max(marg_y) * 1.02, y=mean_y, s='%4.1f %s' % (mean_y, yunit),
              rotation=270.,
              verticalalignment='center')

    if x_aux is not None:
        prior = x_aux / max(x_aux) * max(marg_x)
        likelihood = marg_x / prior
        likelihood *= max(marg_x) * max(likelihood)
        l_p, = ax_x.plot(x, prior, 'k--')
        l_l, = ax_x.plot(x, likelihood, 'r:')
    if y_aux is not None:
        prior = y_aux / max(y_aux) * max(marg_y)
        likelihood = marg_y / prior
        likelihood *= max(marg_y) / max(likelihood)
        l_pi, = ax_y.plot(prior, y, 'k--', label='prior')
        l_l, = ax_y.plot(likelihood, y, 'r:', label='likelihood')

    ax_y.legend((l_p, l_l, l_pi), ('posterior', 'likelihood', 'prior'), loc=(0, -0.1))

    ax_2D.set_xlabel(xlabel)
    ax_2D.set_ylabel(ylabel)

    ax_2D.tick_params(bottom=True, top=True, left=True, right=True,
                      labelbottom=True, labelleft=True, labeltop=False, labelright=False)

    plt.colorbar(mappable=cf, cax=ax_cb)

    return fig, [ax_2D, ax_x, ax_y]


def plot(p, dis, dep, depth_prior=None, distance_prior=None):
    nmodel, ndepth, ndist = p.shape

    # Depth-distance matrix
    depthdist = np.sum(p, axis=(0)) / nmodel

    fig, axs = plot_2D_with_marginals(dis, dep, depthdist,
                                      x_aux=distance_prior,
                                      y_aux=depth_prior,
                                      xlabel='distance / degree',
                                      ylabel='depth / km',
                                      xunit='degree',
                                      yunit='km',
                                      figsize=(12,7.5))
    axs[0].set_ylim(150, 0)
    fig.savefig('depth_distance.png', dpi=200)
    plt.close('all')


def plot_models(p, files, tt_path):
    from h5py import File
    # Model likelihood plot
    models_p = np.sum(p, axis=(1, 2))
    plt.plot(models_p, '.')
    plt.xlabel('Model index')
    plt.ylabel('likelihood')
    plt.tight_layout()
    plt.savefig('model_likelihood.png')
    plt.close('all')

    # Plot with all mantle profiles
    models_p /= max(models_p)
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    for fnam, model_p in zip(files, models_p):
        with File(pjoin(tt_path, 'tt', fnam)) as f:
            radius = np.asarray(f['mantle/radius'])
            radius = (max(radius) - radius) * 1e-3
            for a in ax:
                lp,  = a.plot(f['mantle/vp'], radius, c='lightgrey',
                              alpha=0.4, zorder=2)
                ls,  = a.plot(f['mantle/vs'], radius,c='lightgrey',
                              alpha=0.4, zorder=2)
                lp,  = a.plot(f['mantle/vp'], radius, c='darkblue',
                              alpha=model_p**2, zorder=20)
                ls,  = a.plot(f['mantle/vs'], radius,c='darkred',
                              alpha=model_p**2, zorder=20)
    ax[0].set_ylim(2200, 0)
    ax[1].set_ylim(220, 0)
    ax[1].legend((lp, ls), ('vp', 'vs'))
    for a in ax:
        a.set_xlabel('velocity / m/s')
        a.set_ylabel('depth / km')

    plt.savefig('velocity_models.png', dpi=200)


def plot_phases(tt, p, phase_list, freqs, tt_meas, sigma):
    nphase = len(phase_list)
    fig, axs = plt.subplots(nrows=nphase, ncols=1, figsize=(6, 1.5 + nphase))

    for iax, ax in enumerate(axs):
        width = max(50., sigma[iax] * 2.0)
        phase_mean = np.sum(tt[:, :, :, iax] * p, axis=None) / np.sum(p, axis=None)
        ax.hist(tt[:, :, :, iax].flatten(),
                weights=p[:, :, :].flatten(),
                bins=np.linspace(tt_meas[iax] - width,
                                 tt_meas[iax] + width, 100))
        ax.axvline(x=tt_meas[iax], c='r')
        ax.axvline(x=tt_meas[iax] - sigma[iax], c='r', ls='--')
        ax.axvline(x=tt_meas[iax] + sigma[iax], c='r', ls='--')
        ax.axvline(x=phase_mean, c='darkgreen', lw=2)
        if phase_list[iax] in ['R1', 'G1']:
            phase_string = '%s %3ds'% (phase_list[iax],
                                       int(1./freqs[iax]))
        else:
            phase_string = phase_list[iax]
        ax.text(x=0.05, y=0.5, s=phase_string,
                fontsize=14, weight='bold',
                transform = ax.transAxes)
    plt.tight_layout()
    plt.savefig('phase_misfits.png')
    plt.close('all')
