# -*- coding: utf-8 -*-
import numpy as np
from os.path import join as pjoin
from os import environ

try:
    environ['DISPLAY']
except KeyError:
    import matplotlib
    matplotlib.use('Agg')

from matplotlib import pyplot as plt
from h5py import File

from locator.general_functions import calc_marginals_depdis

def plot_2D_with_marginals(x, y, z, x_aux=None, y_aux=None,
                           xlabel=None, ylabel=None, xunit='', yunit='',
                           scatter=False, **kwargs):
    fig = plt.figure(**kwargs)

    # Central 2D plot
    ax_2D = fig.add_axes([0.10, 0.10, 0.76, 0.72], label='2D')
    ax_x = fig.add_axes([0.10, 0.82, 0.76, 0.12], label='y_marginal',
                        sharex=ax_2D)
    plt.axis('off')
    ax_y = fig.add_axes([0.86, 0.10, 0.11, 0.72], label='x_marginal',
                        sharey=ax_2D)
    plt.axis('off')
    ax_cb = fig.add_axes([0.86, 0.84, 0.017, 0.12], label='colorbar')

    ax_x.get_xaxis().set_visible(False)
    ax_x.get_yaxis().set_visible(False)

    # # Calculate marginals and means
    mean_y, mean_x, marg_z, marg_y, marg_x = \
        calc_marginals_depdis(y, x, z)

    # normalize marginals (for plotting)
    marg_x /= np.nanmax(marg_x)
    marg_y /= np.nanmax(marg_y)

    # Integrate over all models
    z_sum = np.sum(z, axis=0)

    ax_x.plot(x, marg_x)
    l_p, = ax_y.plot(marg_y, y)
    if scatter:
        xx, yy = np.meshgrid(x, y)
        cf = ax_2D.scatter(xx, yy, c=np.sqrt(z_sum), cmap='afmhot_r',
                           marker='+')
    else:
        levels = np.sqrt(np.max(z_sum)) * \
                 np.asarray((0.05, 0.2, 0.5, 0.75, 1.0))
        cf = ax_2D.contourf(x, y, np.sqrt(z_sum), cmap='afmhot_r',
                            levels=levels)

    ax_x.set_xlim(x[0], x[-1])
    ax_x.set_ylim(0, 1)
    ax_y.set_ylim(y[0], y[-1])
    ax_y.set_xlim(0, 1)

    # mark mean values 
    ax_x.axvline(x=mean_x, linestyle='dashed', color='black')
    ax_x.text(x=mean_x, y=max(marg_x) * 1.1, s='%4.1f %s' % (mean_x, xunit),
              horizontalalignment='center')
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

    ax_y.legend((l_p, l_l, l_pi), ('posterior', 'likelihood', 'prior'),
                loc=(0, -0.1))

    ax_2D.set_xlabel(xlabel)
    ax_2D.set_ylabel(ylabel)

    ax_2D.tick_params(bottom=True, top=True, left=True, right=True,
                      labelbottom=True, labelleft=True, labeltop=False,
                      labelright=False)

    plt.colorbar(mappable=cf, cax=ax_cb)

    return fig, [ax_2D, ax_x, ax_y]


def plot(p, dis, dep, depth_prior=None, distance_prior=None):
    fig, axs = plot_2D_with_marginals(dis, dep, p,
                                      x_aux=distance_prior,
                                      y_aux=depth_prior,
                                      xlabel='distance / degree',
                                      ylabel='depth / km',
                                      xunit='degree',
                                      yunit='km',
                                      figsize=(12, 7.5))
    axs[0].set_ylim(300, 0)
    fig.savefig('depth_distance.png', dpi=200)
    plt.close('all')


def plot_model_density(p_model, prior, vp_all, vs_all):
    from matplotlib.pyplot import hist
    vp_min = 0.2e3
    vp_max = 9.0e3
    vs_min = 0.2e3
    vs_max = 5.5e3
    # hist_post = np.histogram(a=vp_all, bins=100, range=(vp_min, vp_max),
    #                          weights=p_model)
    sp = vp_all.shape
    p_model_mat = p_model.reshape((sp[0], 1)).repeat(sp[ 1], axis=1)
    prior_mat = prior.reshape((sp[0], 1)).repeat(sp[ 1], axis=1)
    nbins = 100
    vp_density, vp_bins, tmp = hist(vp_all, bins=nbins, range=(vp_min, vp_max),
                                    weights=p_model_mat)
    plt.close()
    vs_density, vs_bins, tmp = hist(vs_all, bins=nbins, range=(vs_min, vs_max),
                                    weights=p_model_mat)
    plt.close()
    vp_prior, vp_bins, tmp = hist(vp_all, bins=nbins, range=(vp_min, vp_max),
                               weights=prior_mat)
    plt.close()
    vs_prior, vs_bins, tmp = hist(vs_all, bins=nbins, range=(vs_min, vs_max),
                               weights=prior_mat)
    plt.close()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharey='all',
                           sharex='col')
    ax[0][0].set_title('P-wave velocity prior')
    ax[0][1].set_title('S-wave velocity prior')
    ax[1][0].set_title('P-wave velocity posterior')
    ax[1][0].set_title('P-wave velocity posterior')

    ax[0][0].pcolormesh(np.linspace(vp_min, vp_max, nbins),
                        np.arange(0, 200, 5),
                        np.asarray(vp_prior),
                        vmin=0., cmap='BuGn')
    ax[0][1].pcolormesh(np.linspace(vs_min, vs_max, nbins),
                        np.arange(0, 200, 5),
                        np.asarray(vs_prior),
                        vmin=0., cmap='afmhot_r')
    ax[1][0].pcolormesh(np.linspace(vp_min, vp_max, nbins),
                     np.arange(0, 200, 5),
                     np.asarray(vp_density),
                     vmin=0., cmap='BuGn')
    ax[1][1].pcolormesh(np.linspace(vs_min, vs_max, nbins),
                     np.arange(0, 200, 5),
                     np.asarray(vs_density),
                     vmin=0., cmap='afmhot_r')
    ax[0][0].set_ylim(200, 0)
    ax[0][0].set_ylabel('depth [km]')

    for a in ax:
        a[0].set_xlabel('Vp [m/s]')
        a[1].set_xlabel('Vs [m/s]')
    plt.tight_layout()
    plt.savefig('vel_hist.pdf')
    plt.close()


def plot_models(p, files, tt_path):
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
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    for fnam, model_p in zip(files, models_p):
        with File(pjoin(tt_path, 'tt', fnam)) as f:
            radius = np.asarray(f['mantle/radius'])
            radius = (max(radius) - radius) * 1e-3
            # for a in ax:
            #    lp,  = a.plot(f['mantle/vp'], radius, c='lightgrey',
            #                  alpha=0.4, zorder=2)
            #    ls,  = a.plot(f['mantle/vs'], radius,c='lightgrey',
            #                  alpha=0.4, zorder=2)
            #    lp,  = a.plot(f['mantle/vp'], radius, c='darkblue',
            #                  alpha=model_p**2, zorder=20)
            #    ls,  = a.plot(f['mantle/vs'], radius,c='darkred',
            #                  alpha=model_p**2, zorder=20)
            if not fnam[0] == 'C':
                lp, = ax[0][0].plot(f['mantle/vp'], radius, c='lightgrey',
                                    lw=0.5, alpha=0.4, zorder=2)
                ls, = ax[0][0].plot(f['mantle/vs'], radius, c='darkred',
                                    lw=0.5, alpha=0.4, zorder=2)
                if model_p > 0.3:
                    lp, = ax[0][1].plot(f['mantle/vp'], radius, c='darkblue',
                                        lw=0.5, alpha=model_p ** 2, zorder=20)
                    ls, = ax[0][1].plot(f['mantle/vs'], radius, c='darkred',
                                        lw=0.5, alpha=model_p ** 2, zorder=20)
                if model_p > 0.3:
                    lp, = ax[1][1].plot(f['mantle/vs'] - f['mantle/vs'][-7],
                                        radius, c='darkred', lw=0.5,
                                        alpha=model_p ** 2, zorder=20)
                lp, = ax[1][0].plot(f['mantle/vs'] - f['mantle/vs'][-7],
                                    radius, c='darkred', lw=0.5,
                                    alpha=0.4, zorder=2)

    for a in ax.flatten():
        a.set_ylim(600, 0)
    ax[0][0].set_xlim(3300, 4700)
    ax[0][1].set_xlim(3300, 4700)
    # ax[0][1].legend((lp, ls), ('vp', 'vs'))
    ax[1][0].set_xlim(-1000, 400)
    ax[1][1].set_xlim(-1000, 400)
    # ax[1][1].legend((lp, ls), ('vp', 'vs'))
    ax[0][0].set_title('V_S, all models')
    ax[0][1].set_title('V_S, allowed models')
    ax[1][0].set_title('V_S - V_S (low crust), all models')
    ax[1][1].set_title('V_S - V_S (low crust), allowed models')
    for a in ax[0]:
        a.set_xlabel('velocity, S-waves [m/s]')
        a.set_ylabel('depth [km]')
    for a in ax[1]:
        a.set_xlabel('reduced velocity, S-waves [m/s]')
        a.set_ylabel('depth [km]')
    plt.tight_layout()
    plt.savefig('velocity_models.png', dpi=200)


def plot_phases(tt, p, phase_list, freqs, tt_meas, sigma):
    nphase = len(phase_list)
    fig, axs = plt.subplots(nrows=nphase, ncols=1, figsize=(6, 1.5 + nphase))

    for iax, ax in enumerate(axs):
        width = max(50., sigma[iax] * 2.0)
        phase_mean = np.sum(tt[:, :, :, iax] * p, axis=None) / np.sum(p,
                                                                      axis=None)
        ax.hist(tt[:, :, :, iax].flatten(),
                weights=p[:, :, :].flatten(),
                bins=np.linspace(tt_meas[iax] - width,
                                 tt_meas[iax] + width, 100))
        ax.axvline(x=tt_meas[iax], c='r')
        ax.axvline(x=tt_meas[iax] - sigma[iax], c='r', ls='--')
        ax.axvline(x=tt_meas[iax] + sigma[iax], c='r', ls='--')
        ax.axvline(x=phase_mean, c='darkgreen', lw=2)
        if phase_list[iax] in ['R1', 'G1']:
            phase_string = '%s %3ds' % (phase_list[iax],
                                        int(1. / freqs[iax]))
        else:
            phase_string = phase_list[iax]
        ax.text(x=0.05, y=0.5, s=phase_string,
                fontsize=14, weight='bold',
                transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('phase_misfits.png')
    plt.close('all')
