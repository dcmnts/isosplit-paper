from typing import Union
import numpy as np

def isocut6_slow(samples: np.array):
    N = len(samples)
    X = np.sort(samples)
    
    spacings = X[1:] - X[:(N - 1)]
    multiplicities = np.ones((N-1,))
    if np.min(spacings) == 0:
        raise Exception('Spacings are not allowed to be zero')
    log_densities = np.log(multiplicities / spacings)

    # Unimodal fit to densities
    log_densities_unimodal_fit = jisotonic5_updown(log_densities, weights=None)

    densities_unimodal_fit_times_spacings = np.exp(log_densities_unimodal_fit) * spacings

    peak_index = np.argmax(log_densities_unimodal_fit)

    # dipscore_out
    dipscore_out, critical_range_min, critical_range_max = compute_ks5(multiplicities, densities_unimodal_fit_times_spacings, peak_index)

    log_densities_resid = log_densities - log_densities_unimodal_fit
    log_densities_resid_on_critical_range = log_densities_resid[critical_range_min:(critical_range_max + 1)]

    log_densities_resid_fit_on_critical_range = jisotonic5_downup(log_densities_resid_on_critical_range, weights=None)
    
    cutpoint_index = np.argmin(log_densities_resid_fit_on_critical_range)

    # cutpoint_out
    cutpoint_out = 0.5 * (X[critical_range_min + cutpoint_index] + X[critical_range_min + cutpoint_index + 1])

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

    B1x, MSE1x = jisotonic5(values[:(best_ind + 1)], weights[:best_ind + 1] if weights is not None else None)
    B2x, MSE2x = jisotonic5(values_reversed[:(N - best_ind - 1)], weights_reversed[:(N - best_ind - 1)] if weights_reversed is not None else None)

    out = np.zeros((N,), dtype=np.float32)
    out[:(best_ind + 1)] = B1x
    out[N-1:best_ind:-1] = B2x

    return out

def jisotonic5_downup(values: np.array, weights: Union[np.array, None]):
    return -jisotonic5_updown(-values, weights)

def jisotonic5(values: np.array, weights: Union[np.array, None]):
    N = len(values)
    if N == 0:
        return np.array((0,), dtype=np.float32), np.array((0,), dtype=np.float32)
    
    unweighted_count0 = np.zeros((N,), dtype=np.int32)
    count0 = np.zeros((N,), dtype=np.float32)
    sum0 = np.zeros((N,), dtype=np.float32)
    sumsqr0 = np.zeros((N,), dtype=np.float32)
    MSE0 = np.zeros((N,), dtype=np.float32)
    BB = np.zeros((N,), dtype=np.float32)

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
    counts1_left = np.zeros((peak_index + 1,), dtype=np.float32)
    counts2_left = np.zeros((peak_index + 1,), dtype=np.float32)
    for i in range(peak_index + 1):
        counts1_left[i] = counts1[i]
        counts2_left[i] = counts2[i]
    len0 = peak_index + 1
    while (len0 >= 4) or (len0 == peak_index + 1):
        ks0 = compute_ks4(counts1_left[:len0], counts2_left[:len0])
        if ks0 > ks_best:
            critical_range_min = int(0)
            critical_range_max = int(len0 - 1)
            ks_best = ks0
        len0 = int(len0 / 2);

    # from the right
    counts1_right = np.zeros((N - peak_index,), dtype=np.float32)
    counts2_right = np.zeros((N - peak_index,), dtype=np.float32)
    for i in range(N - peak_index):
        counts1_right[i] = counts1[N - 1 - i]
        counts2_right[i] = counts2[N - 1 - i]
    len0 = N - peak_index
    while (len0 >= 4) or (len0 == N - peak_index):
        ks0 = compute_ks4(counts1_right[:len0], counts2_right[:len0])
        if ks0 > ks_best:
            critical_range_min = int(N - len0)
            critical_range_max = int(N - 1)
            ks_best = ks0
        len0 = int(len0 / 2)

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
        if sum_counts1 > 0 and sum_counts2 > 0:
            diff = np.abs(cumsum_counts1 / sum_counts1 - cumsum_counts2 / sum_counts2)
            if (diff > max_diff):
                max_diff = diff

    return max_diff * np.sqrt((sum_counts1 + sum_counts2) / 2)