# -*- coding: utf-8 -*-
import numpy as np
from os.path import join as pjoin
import matplotlib

#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from h5py import File

from locator.general_functions import _calc_marginals

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
        _calc_marginals(y, x, z)

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
    axs[0].set_ylim(600, 0)
    fig.savefig('depth_distance.png', dpi=200)
    plt.close('all')


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

def _write_model_density(p, files, tt_path):
    depths_target = np.arange(0.0, 1000.0, 2.0)
    vp_sums = np.zeros_like(depths_target)
    vp_sums2 = np.zeros_like(depths_target)
    vs_sums = np.zeros_like(depths_target)
    vs_sums2 = np.zeros_like(depths_target)
    p_model = np.sum(p, axis=(1, 2))
    p_model /= np.sum(p_model)
    nmodel = 0
    for fnam, model_p in zip(files, p_model):
        with File(pjoin(tt_path, 'tt', fnam)) as f:
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
    print('nmodel: ', nmodel)
    fnam = 'model_mean_sigma.txt'
    vp_mean = vp_sums
    vs_mean = vs_sums
    vp_sigma = np.sqrt(vp_sums2 - vp_mean**2)
    vs_sigma = np.sqrt(vs_sums2 - vs_mean**2)
    with open(fnam, 'w') as f:
        f.write('%6s, %8s, %8s, %8s, %8s\n' %
                ('depth', 'vp_mean', 'vp_sig', 'vs_mean', 'vs_sig'))
        for idepth in range(0, len(depths_target)):
            f.write('%6.1f, %8.2f, %8.2f, %8.2f, %8.2f\n' %
                    (depths_target[idepth],
                     vp_mean[idepth], vp_sigma[idepth],
                     vs_mean[idepth], vs_sigma[idepth]))


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
