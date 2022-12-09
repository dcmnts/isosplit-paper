from typing import Union
import numpy as np

def isocut5(samples: np.array):
    N = len(samples)
    samples_sorted = np.sort(samples)

    ##############################################
    # SUBSAMPLING
    # num_bins is sqrt(N/2) * factor
    num_bins_factor = 1
    num_bins = int(np.ceil(np.sqrt(N * 1.0 / 2) * num_bins_factor))

    num_bins_1 = int(np.ceil(num_bins / 2)) # left bins
    num_bins_2 = num_bins - num_bins_1 # right bins

    # I guess this is the same as num_bins
    num_intervals = num_bins_1 + num_bins_2

    # what is this? the number of samples in each interval?
    intervals = np.zeros((num_intervals, ), dtype=samples.dtype)
    for i in range(num_bins_1):
        intervals[i] = i + 1
    for i in range(num_bins_2):
        intervals[num_intervals - 1 - i] = i + 1

    alpha = (N - 1) / np.sum(intervals)

    # now we scale this such that sum(intervals) = N - 1
    intervals = intervals * alpha

    # number of subsamples
    N_sub = num_intervals + 1

    # indices of the subsampled samples
    inds = np.zeros((N_sub,), dtype=np.int32)
    inds[0] = 0
    for i in range(num_intervals):
        inds[i + 1] = int(np.floor(inds[i] + intervals[i]))
    
    # These are the subsampled samples
    X_sub = np.zeros((N_sub,), dtype=samples.dtype)
    for i in range(N_sub):
        X_sub[i] = samples_sorted[inds[i]]
    ##################################################

    spacings = X_sub[1:] - X_sub[:(N_sub - 1)]
    multiplicities = inds[1:] - inds[:(N_sub - 1)]
    densities = multiplicities / spacings

    densities_unimodal_fit = jisotonic5_updown(densities, multiplicities)

    densities_resid = densities - densities_unimodal_fit

    densities_unimodal_fit_times_spacings = densities_unimodal_fit * spacings

    peak_index = np.argmax(densities_unimodal_fit)

    # dipscore_out
    dipscore_out, critical_range_min, critical_range_max = compute_ks5(multiplicities, densities_unimodal_fit_times_spacings, peak_index)

    critical_range_length = critical_range_max - critical_range_min + 1

    densities_resid_on_critical_range = densities_resid[critical_range_min:(critical_range_max + 1)]
    weights_for_downup = spacings[critical_range_min:(critical_range_max + 1)]

    densities_resid_fit_on_critical_range = jisotonic5_downup(densities_resid_on_critical_range, weights_for_downup)

    cutpoint_index = np.argmin(densities_resid_fit_on_critical_range)

    # cutpoint_out
    cutpoint_out = (X_sub[critical_range_min + cutpoint_index] + X_sub[critical_range_min + cutpoint_index + 1]) / 2

    return dipscore_out, cutpoint_out

def jisotonic5_updown(values: np.array, weights: Union[np.array, None]):
    N = len(values)
    values_reversed = values[::-1]
    if weights is not None:
        weights_reversed = weights[::-1]
    else:
        weights_reversed = None
    B1, MSE1 = jisotonic5(values, weights)
    B2, MSE2 = jisotonic5(values_reversed, weights_reversed)

    MSE = MSE1 + MSE2[::-1]
    best_ind = np.argmin(MSE)

    B1x, MSE1x = jisotonic5(values[:(best_ind + 1)], weights[:best_ind + 1])
    B2x, MSE2x = jisotonic5(values_reversed[:(N - best_ind - 1)], weights_reversed[:(N - best_ind - 1)])

    out = np.zeros((N,), dtype=values.dtype)
    out[:(best_ind + 1)] = B1x
    out[N-1:best_ind:-1] = B2x

    return out

def jisotonic5_downup(values: np.array, weights: Union[np.array, None]):
    return -jisotonic5_updown(-values, weights)

def jisotonic5(values: np.array, weights: Union[np.array, None]):
    N = len(values)
    if N == 0:
        return [], []
    
    unweighted_count0 = np.zeros((N,), dtype=np.int32)
    count0 = np.zeros((N,), dtype=values.dtype)
    sum0 = np.zeros((N,), dtype=values.dtype)
    sumsqr0 = np.zeros((N,), dtype=values.dtype)
    MSE0 = np.zeros((N,), dtype=values.dtype)
    BB = np.zeros((N,), dtype=values.dtype)

    last_index = 0
    
    if weights is not None:
        w0 = weights[0]
    else:
        w0 = 1
    
    count0[last_index] = w0
    unweighted_count0[last_index] = 1
    sum0[last_index] = values[0] * w0
    sumsqr0[last_index] = values[0] * values[0] * w0
    MSE0[0] = 0

    for j in range(1, N):
        last_index += 1
        unweighted_count0[last_index] = 1
        if weights is not None:
            w0 = weights[j]
        else:
            w0 = 1
        count0[last_index] = w0
        sum0[last_index] = values[j] * w0
        sumsqr0[last_index] = values[j] * values[j] * w0;
        MSE0[j] = MSE0[j - 1]
        while True:
            if last_index <= 0:
                break;
            if sum0[last_index - 1] / count0[last_index - 1] < sum0[last_index] / count0[last_index]:
                break;
            else:
                prevMSE = sumsqr0[last_index - 1] - sum0[last_index - 1] * sum0[last_index - 1] / count0[last_index - 1]
                prevMSE += sumsqr0[last_index] - sum0[last_index] * sum0[last_index] / count0[last_index]
                unweighted_count0[last_index - 1] += unweighted_count0[last_index]
                count0[last_index - 1] += count0[last_index]
                sum0[last_index - 1] += sum0[last_index]
                sumsqr0[last_index - 1] += sumsqr0[last_index]
                newMSE = sumsqr0[last_index - 1] - sum0[last_index - 1] * sum0[last_index - 1] / count0[last_index - 1]
                MSE0[j] += newMSE - prevMSE
                last_index -= 1

    ii = 0
    for k in range(last_index + 1):
        for cc in range(unweighted_count0[k]):
            BB[ii + cc] = sum0[k] / count0[k]
        ii += unweighted_count0[k]
    
    return BB, MSE0

def compute_ks5(counts1, counts2, peak_index):
    N = len(counts1)
    critical_range_min = 0
    critical_range_max = N - 1 # should get over-written!
    ks_best = -1

    # from the left
    counts1_left = np.zeros((peak_index + 1,), dtype=counts1.dtype)
    counts2_left = np.zeros((peak_index + 1,), dtype=counts1.dtype)
    for i in range(peak_index + 1):
        counts1_left[i] = counts1[i]
        counts2_left[i] = counts2[i]
    len0 = peak_index + 1
    while (len0 >= 4) or (len0 == peak_index + 1):
        ks0 = compute_ks4(counts1_left, counts2_left)
        if ks0 > ks_best:
            critical_range_min = 0
            critical_range_max = len0 - 1
            ks_best = ks0
        len0 = len0 / 2;

    # from the right
    counts1_right = np.zeros((N - peak_index,), dtype=counts1.dtype)
    counts2_right = np.zeros((N - peak_index,), dtype=counts1.dtype)
    for i in range(N - peak_index):
        counts1_right[i] = counts1[N - 1 - i]
        counts2_right[i] = counts2[N - 1 - i]
    len0 = N - peak_index
    while (len0 >= 4) or (len0 == N - peak_index):
        ks0 = compute_ks4(counts1_right, counts2_right)
        if ks0 > ks_best:
            critical_range_min = N - len0
            critical_range_max = N - 1
            ks_best = ks0
        len0 = len0 / 2

    return ks_best, critical_range_min, critical_range_max

def compute_ks4(counts1, counts2):
    N = len(counts1)

    sum_counts1 = np.sum(counts1)
    sum_counts2 = np.sum(counts2)

    cumsum_counts1 = 0
    cumsum_counts2 = 0

    max_diff = 0
    for i in range(N):
        cumsum_counts1 += counts1[i]
        cumsum_counts2 += counts2[i]
        diff = np.abs(cumsum_counts1 / sum_counts1 - cumsum_counts2 / sum_counts2)
        if (diff > max_diff):
            max_diff = diff

    return max_diff * np.sqrt((sum_counts1 + sum_counts2) / 2)