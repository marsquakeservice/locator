import numpy as np
from matplotlib import pyplot as plt


def plot_2D_with_marginals(x, y, z, xlabel=None, ylabel=None, **kwargs):
    fig = plt.figure(**kwargs)

    levels = np.sqrt(np.max(z)) * np.asarray((0.0, 0.05, 0.2, 0.5, 0.75, 1.0))
    z_int = np.sum(z, axis=None)

    # Central 2D plot
    ax_2D = fig.add_axes([0.10, 0.10, 0.76, 0.72], label='2D')
    ax_x = fig.add_axes([0.10, 0.82, 0.76, 0.12], label='y_marginal', sharex=ax_2D)
    plt.axis('off')
    ax_y = fig.add_axes([0.86, 0.10, 0.12, 0.72], label='x_marginal', sharey=ax_2D)
    plt.axis('off')
    ax_cb = fig.add_axes([0.86, 0.84, 0.017, 0.15], label='colorbar')

    ax_x.get_xaxis().set_visible(False)
    ax_x.get_yaxis().set_visible(False)

    cf = ax_2D.contourf(x, y, np.sqrt(z), cmap='afmhot_r', levels=levels)
    marg_x = np.sum(z, axis=0)
    ax_x.plot(x, marg_x)
    marg_y = np.sum(z, axis=1)
    ax_y.plot(marg_y, y)
    ax_x.set_xlim(x[0], x[-1])
    ax_y.set_ylim(y[0], y[-1])

    # Calculate mean values and mark them
    mean_x = np.sum(marg_x * x / z_int)
    ax_x.axvline(x=mean_x, linestyle='dashed', color='black')
    mean_y = np.sum(marg_y * y / z_int)
    ax_y.axhline(y=mean_y, linestyle='dashed', color='black')

    ax_2D.set_xlabel(xlabel)
    ax_2D.set_ylabel(ylabel)

    plt.colorbar(mappable=cf, cax=ax_cb)
    ax_2D.set_ylim(150, 0)

    return fig, [ax_2D, ax_x, ax_y]

#def plot_origin_time()


def plot(p, dis, dep):
    nmodel, ndepth, ndist = p.shape

    # Depth-distance matrix
    depthdist = np.sum(p, axis=(0)) / nmodel

    fig, axs = plot_2D_with_marginals(dis, dep, depthdist,
                                      xlabel='distance / degree',
                                      ylabel='depth / km',
                                      figsize=(8,5))
    fig.savefig('depth_distance.png')

    #plt.contourf(dis, dep, depthdist)
    #plt.colorbar()
    #plt.xlabel('distance')
    #plt.ylabel('depth')
    #plt.tight_layout()
    #plt.savefig('depth_distance.png')
    plt.close('all')

    # Model likelihood plot
    plt.plot(np.sum(p, axis=(1, 2)), '.')
    plt.xlabel('Model index')
    plt.ylabel('likelihood')
    plt.tight_layout()
    plt.savefig('model_likelihood.png')
    plt.close('all')


def plot_phases(tt, p, phase_list, tt_meas, sigma):
    nphase = len(phase_list)
    fig, axs = plt.subplots(nrows=nphase, ncols=1, figsize=(6, 1.5 + nphase))

    for iax, ax in enumerate(axs):
        phase_mean = np.sum(tt[:, :, :, iax] * p, axis=None) / np.sum(p, axis=None)
        ax.hist(tt[:, :, :, iax].flatten(),
                weights=p[:, :, :].flatten(),
                bins=np.linspace(tt_meas[iax] - 50, tt_meas[iax] + 50, 100))
        ax.axvline(x=tt_meas[iax], c='r')
        ax.axvline(x=tt_meas[iax] - sigma[iax], c='r', ls='--')
        ax.axvline(x=tt_meas[iax] + sigma[iax], c='r', ls='--')
        ax.axvline(x=phase_mean, c='darkgreen', lw=2)
        ax.text(x=0.05, y=0.5, s=phase_list[iax],
                fontsize=14, weight='bold',
                transform = ax.transAxes)
    plt.tight_layout()
    plt.savefig('phase_misfits.png')
    plt.close('all')