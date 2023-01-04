import matplotlib.pyplot as plt
import numpy as np
from isosplit5_slow.isocut6_slow import jisotonic5_updown, compute_ks5, jisotonic5_downup

def isocut6_slow_demo(samples: np.array, *, xlim, output_fname: str, create_figure: bool):
    if create_figure:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 8))
        annotation_xy = (10, 100)

    N = len(samples)
    X = np.sort(samples)

    if create_figure:
        ax1.hist(X, 100, color='lightgray', edgecolor='black');
        ax1.set_xlim(xlim)
        ax1.set_xticks([])
        ax1.set_ylabel('Count')
        ax1.annotate(text='A', xy=annotation_xy, xycoords='axes pixels', fontsize=20)
    
    spacings = X[1:] - X[:(N - 1)]
    multiplicities = np.ones((N-1,))
    if np.min(spacings) == 0:
        raise Exception('Spacings are not allowed to be zero')
    log_densities = np.log(multiplicities / spacings)

    # Unimodal fit to densities
    log_densities_unimodal_fit = jisotonic5_updown(log_densities, weights=None)

    if create_figure:
        ax2.plot((X[1:] + X[0:(N - 1)]) / 2, log_densities, c='gray') 
        ax2.plot((X[1:] + X[0:(N - 1)]) / 2, log_densities_unimodal_fit, c='red') 
        ax2.set_xlim(xlim)
        ax2.set_xticks([])
        ax2.set_ylabel('Log density')
        ax2.annotate(text='B', xy=annotation_xy, xycoords='axes pixels', fontsize=20)

    densities_unimodal_fit_times_spacings = np.exp(log_densities_unimodal_fit) * spacings

    peak_index = np.argmax(log_densities_unimodal_fit)

    # dipscore_out
    dipscore_out, critical_range_min, critical_range_max = compute_ks5(multiplicities, densities_unimodal_fit_times_spacings, peak_index)

    log_densities_resid = log_densities - log_densities_unimodal_fit
    log_densities_resid_on_critical_range = log_densities_resid[critical_range_min:(critical_range_max + 1)]

    log_densities_resid_fit_on_critical_range = jisotonic5_downup(log_densities_resid_on_critical_range, weights=None)

    if create_figure:
        ax3.plot((X[:(N- 1)] + X[1:]) / 2, log_densities_resid, c='gray') 
        ax3.plot((X[critical_range_min + 1:critical_range_max + 2] + X[critical_range_min:critical_range_max + 1]) / 2, log_densities_resid_fit_on_critical_range, c='red') 
        ax3.set_xlim(xlim)
        # ax3.set_ylim([-500, 500])
        ax3.set_ylabel('Log density residual')
        ax3.annotate(text='C', xy=annotation_xy, xycoords='axes pixels', fontsize=20)
    
        plt.savefig(output_fname)
    
    cutpoint_index = np.argmin(log_densities_resid_fit_on_critical_range)

    # cutpoint_out
    cutpoint_out = 0.5 * (X[critical_range_min + cutpoint_index] + X[critical_range_min + cutpoint_index + 1])

    return dipscore_out, cutpoint_out

if __name__ == '__main__':
    np.random.seed(1)

    N1 = 500
    N2 = 1000
    N = N1 + N2
    separation = 4
    samples = np.zeros((N,), dtype=np.float64)
    for j in range(N):
        if j < N1:
            samples[j] = np.random.randn() - separation / 2
        else:
            samples[j] = np.random.randn() + separation / 2

    dipscore_out, cutpoint_out = isocut6_slow_demo(
        samples,
        xlim=[-5.5, 5.5],
        output_fname='results/isocut_demo.svg',
        create_figure=True
    )
    print(f'dipscore: {dipscore_out}; Cutpoint: {cutpoint_out}')