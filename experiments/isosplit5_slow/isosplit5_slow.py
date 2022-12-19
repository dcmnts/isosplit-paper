import numpy as np
from isocut5_slow import isocut5_slow

def isosplit5_slow(
    X: np.ndarray, *,
    isocut_threshold=1,
    min_cluster_size=10,
    K_init=200,
    max_iterations_per_pass=500
):
    M = X.shape[0]
    N = X.shape[1]
    # compute the initial clusters
    target_parcel_size = min_cluster_size
    target_num_parcels = K_init
    # !! important not to do a final reassign because then the shapes will not be conducive to isosplit iterations -- hexagons are not good for isosplit!
    final_reassign = False
    labels = parcelate2(
        X,
        target_parcel_size=target_parcel_size,
        target_num_parcels=target_num_parcels,
        final_reassign=final_reassign
    )
    Kmax = int(np.max(labels))

    centroids = np.zeros((M, Kmax), dtype=np.float32)
    covmats = np.zeros((M, M, Kmax), dtype=np.float32)

    clusters_to_compute_vec = []
    for k in range(Kmax):
        clusters_to_compute_vec.append(True)
    centroids = compute_centroids(centroids, X, labels, clusters_to_compute_vec)
    covmats = compute_covmats(covmats, X, labels, centroids, clusters_to_compute_vec)

    # The active labels are those that are still being used -- for now, everything is active
    active_labels_vec = []
    active_labels = []
    for i in range(Kmax):
        active_labels_vec.append(True)
        active_labels.append(i + 1)

    # Repeat while something has been merged in the pass
    final_pass = False # plus we do one final pass at the end

    # Keep a matrix of comparisons that have been made in this pass
    comparisons_made = np.zeros((Kmax, Kmax), dtype=np.int32) 
    
    passnum = 0
    while True: # passes
        passnum += 1
        something_merged = False # Keep track of whether something has merged in this pass. If not, do a final pass.
        clusters_changed_vec_in_pass = [] # Keep track of the clusters that have changed in this pass so that we can update the comparisons_made matrix at the end
        for i in range(Kmax):
            clusters_changed_vec_in_pass.append(False)
        iteration_number = 0
        while True: # iterations
            clusters_changed_vec_in_iteration = []
            for i in range(Kmax):
                clusters_changed_vec_in_iteration.append(False)

            iteration_number += 1
            if iteration_number > max_iterations_per_pass:
                print("Warning: max iterations per pass exceeded.\n")
                break;

            if len(active_labels) > 0:
                # Create an array of active centroids and comparisons made, for determining the pairs to compare
                active_centroids = np.zeros((M, len(active_labels)), dtype=np.float32)
                for i in range(len(active_labels)):
                    for m in range(M):
                        active_centroids[m, i] = centroids[m, active_labels[i] - 1]
                active_comparisons_made = np.zeros((len(active_labels), len(active_labels)), dtype=np.int32)
                for i1 in range(len(active_labels)):
                    for i2 in range(len(active_labels)):
                        active_comparisons_made[i1, i2] = comparisons_made[active_labels[i1] - 1, active_labels[i2] - 1]

                # Find the pairs to compare on this iteration
                # These will be closest pairs of active clusters that have not yet
                # been compared in this pass
                inds1, inds2 = get_pairs_to_compare(active_centroids, active_comparisons_made)
                
                # remap the clusters to the original labeling
                inds1b = []
                inds2b = []
                for i in range(len(inds1)):
                    inds1b.append(active_labels[inds1[i] - 1])
                    inds2b.append(active_labels[inds2[i] - 1])

                # If we didn't find any, break from this iteration
                if len(inds1b) == 0:
                    break

                # Actually compare the pairs -- in principle this operation could be parallelized
                clusters_changed = []
                total_num_label_changes = 0

                # the labels are updated
                new_labels, clusters_changed, total_num_label_changes = compare_pairs(X=X, labels=labels, k1s=inds1b, k2s=inds2b, centroids=centroids, covmats=covmats, min_cluster_size=min_cluster_size, isocut_threshold=isocut_threshold)
                labels = new_labels

                for i in range(len(clusters_changed)):
                    clusters_changed_vec_in_pass[clusters_changed[i] - 1] = True
                    clusters_changed_vec_in_iteration[clusters_changed[i] - 1] = True

                # Update which comparisons have been made
                for j in range(len(inds1b)):
                    comparisons_made[inds1b[j] - 1, inds2b[j] - 1] = 1
                    comparisons_made[inds2b[j] - 1, inds1b[j] - 1] = 1

                # Recompute the centers for those that have changed in this iteration
                centroids = compute_centroids(centroids, X, labels, clusters_changed_vec_in_iteration)
                covmats = compute_covmats(covmats, X, labels, centroids, clusters_changed_vec_in_iteration)

                # For diagnostics
                # printf ("total num label changes = %d\n",total_num_label_changes);

                # Determine whether something has merged and update the active labels
                for i in range(Kmax):
                    active_labels_vec[i] = 0
                for i in range(N):
                    active_labels_vec[labels[i] - 1] = 1
                new_active_labels = []
                for i in range(Kmax):
                    if active_labels_vec[i]:
                        new_active_labels.append(i + 1)
                if len(new_active_labels) < len(active_labels):
                    something_merged = True
                active_labels = new_active_labels

        # zero out the comparisons made matrix only for those that have changed in this pass
        for i in range(Kmax):
            if clusters_changed_vec_in_pass[i]:
                for j in range(Kmax):
                    comparisons_made[i, j] = 0
                    comparisons_made[j, i] = 0

        if something_merged:
            final_pass = False
        if final_pass:
            break; # This was the final pass and nothing has merged
        if not something_merged:
            final_pass = True # If we are done, do one last pass for final redistributes

    # We should remap the labels to occupy the first natural numbers
    labels_map = []
    for i in range(Kmax):
        labels_map.append(0)
    for i in range(len(active_labels)):
        labels_map[active_labels[i] - 1] = i + 1
    for i in range(N):
        labels[i] = labels_map[labels[i] - 1]

    # If the user wants to refine the clusters, then we repeat isosplit on each
    # of the new clusters, recursively. Unless we only found only one cluster.
    K = np.max(labels)

    return labels

def parcelate2(X, *,
    target_parcel_size: int,
    target_num_parcels: int,
    final_reassign: bool
):
    M = X.shape[0]
    N = X.shape[1]
    parcels = []

    labels = []
    for i in range(N):
        labels.append(1)

    P_indices = []
    for i in range(N):
        P_indices.append(i)
    P_centroid = p2_compute_centroid(X, P_indices)
    P_radius = p2_compute_max_distance(P_centroid, X, P_indices)
    P = {
        'indices': P_indices,
        'centroid': P_centroid,
        'radius': P_radius
    }
    parcels.append(P)

    split_factor = 3 # split factor around 2.71 is in a sense ideal

    something_changed_0 = True
    while (len(parcels) < target_num_parcels) and (something_changed_0):
        something_changed = False
        candidate_found = False
        for i in range(len(parcels)):
            indices = parcels[i]['indices']
            if len(indices) > target_parcel_size:
                if parcels[i]['radius'] > 0:
                    candidate_found = True
        if not candidate_found:
            # nothing else will ever be split
            break

        target_radius = 0
        for i in range(len(parcels)):
            if len(parcels[i]['indices']) > target_parcel_size:
                tmp = parcels[i]['radius'] * 0.95
                if tmp > target_radius:
                    target_radius = tmp
        if target_radius == 0:
            print("Unexpected target radius of zero");
            break

        p_index = 0
        while p_index < len(parcels):
            inds = parcels[p_index]['indices']
            rad = parcels[p_index]['radius']
            sz = len(inds)
            if (sz > target_parcel_size) and (rad >= target_radius):
                assignments = []
                iii = p2_randsample(sz, split_factor)
                # iii[1] = iii[0]; //force failure for testing debug
                for i in range(len(inds)):
                    best_pt = -1
                    best_dist = 0
                    for j in range(len(iii)):
                        dist = 0
                        for m in range(M):
                            val = X[m, inds[iii[j]]] - X[m, inds[i]]
                            dist += val * val
                        dist = np.sqrt(dist)
                        if (best_pt < 0) or (dist < best_dist):
                            best_dist = dist
                            best_pt = j
                    assignments.append(best_pt)
                parcels[p_index]['indices'] = []
                for i in range(len(inds)):
                    if assignments[i] == 0:
                        parcels[p_index]['indices'].append(inds[i])
                        labels[inds[i]] = p_index + 1
                        labels[inds[i]] = p_index + 1
                parcels[p_index]['centroid'] = p2_compute_centroid(X, parcels[p_index]['indices'])
                parcels[p_index]['radius'] = p2_compute_max_distance(parcels[p_index]['centroid'], X, parcels[p_index]['indices'])
                for jj in range(1, len(iii)):
                    PP = {
                        'indices': [],
                        'centroid': None,
                        'radius': None
                    }
                    for i in range(len(inds)):
                        if assignments[i] == jj:
                            PP['indices'].append(inds[i])
                            labels[inds[i]] = len(parcels) + 1
                    PP['centroid'] = p2_compute_centroid(X, PP['indices'])
                    PP['radius'] = p2_compute_max_distance(PP['centroid'], X, PP['indices'])
                    if len(PP['indices']) > 0:
                        parcels.append(PP)
                    else:
                        print("Warning in isosplit5: new parcel has no points -- perhaps dataset contains duplicate points?")
                if len(parcels[p_index]['indices']) == sz:
                    print("Warning: Size did not change after splitting parcel.");
                    p_index += 1
                else:
                    something_changed = True
            else:
                p_index += 1

    # final reassign not yet implemented
    if (final_reassign):
        pass
        # centroids=get_parcel_centroids(parcels);
        # labels=knnsearch(centroids',X','K',1)';

    return labels

def compute_centroids(centroids: np.ndarray, X: np.ndarray, labels, clusters_to_compute_vec):
    M = X.shape[0]
    N = X.shape[1]
    Kmax = len(clusters_to_compute_vec)
    C = centroids
    counts = []
    for k in range(Kmax):
        counts.append(0)
        if clusters_to_compute_vec[k]:
            C[:, k] = 0
    for i in range(N):
        k0 = labels[i]
        i0 = k0 - 1
        if clusters_to_compute_vec[i0]:
            for m in range(M):
                C[m, i0] += X[m, i]
            counts[i0] += 1
    for k in range(Kmax):
        if clusters_to_compute_vec[k]:
            if counts[k]:
                for m in range(M):
                    C[m, k] /= counts[k]
    return C

def compute_covmats(covmats: np.ndarray, X: np.ndarray, labels, centroids, clusters_to_compute_vec):
    M = X.shape[0]
    N = X.shape[1]
    C = covmats
    Kmax = len(clusters_to_compute_vec)
    counts = []
    for k in range(Kmax):
        counts.append(0)
        if clusters_to_compute_vec[k]:
            C[:, :, k] = 0
    for i in range(N):
        i0 = labels[i] - 1
        if clusters_to_compute_vec[i0]:
            for m1 in range(M):
                for m2 in range(M):
                    C[m1, m2, i0] += (X[m1, i] - centroids[m1, i0]) * (X[m2, i] - centroids[m2, i0])
            counts[i0] += 1
    for k in range(Kmax):
        if clusters_to_compute_vec[k]:
            if counts[k]:
                for m1 in range(M):
                    for m2 in range(M):
                        C[m1, m2, k] /= counts[k]
    return C

def get_pairs_to_compare(active_centroids, active_comparisons_made):
    inds1 = []
    inds2 = []
    M = active_centroids.shape[0]
    K = active_comparisons_made.shape[0]
    dists = np.zeros((K, K), dtype=np.float32)
    for k1 in range(K):
        for k2 in range(K):
            if active_comparisons_made[k1, k2] or (k1 == k2):
                dists[k1, k2] = -1
            else:
                dist = 0
                for m in range(M):
                    val = active_centroids[m, k1] - active_centroids[m, k2]
                    dist += val * val
                dist = np.sqrt(dist)
                dists[k1, k2] = dist
    # important to only take the mutal closest pairs -- unlike how we originally did it
    # bool something_changed = true;
    # while (something_changed) {
    # something_changed = false;
    best_inds = []
    for k in range(K):
        best_ind = -1
        best_distance = -1
        for k2 in range(K):
            if dists[k, k2] >= 0:
                if (best_distance < 0) or (dists[k, k2] < best_distance):
                    best_distance = dists[k, k2]
                    best_ind = k2
        best_inds.append(best_ind)
    for j in range(K):
        if best_inds[j] > j:
            if best_inds[best_inds[j]] == j: # mutual!
                if dists[j, best_inds[j]] >= 0:
                    inds1.append(j + 1)
                    inds2.append(best_inds[j] + 1)
                    for aa in range(K):
                        dists[j, aa] = -1
                        dists[aa, j] = -1
                        dists[best_inds[j], aa] = -1
                        dists[aa, best_inds[j]] = -1
                    
                    # something_changed = true;
    return inds1, inds2

def compare_pairs(X, *, labels, k1s, k2s, centroids, covmats, min_cluster_size, isocut_threshold: float):
    N = len(labels)
    Kmax = np.max(labels)
    clusters_changed_vec = []
    for i in range(Kmax):
        clusters_changed_vec.append(False)
    new_labels = labels
    total_num_label_changes = 0
    for i1 in range(len(k1s)):
        k1 = k1s[i1]
        k2 = k2s[i1]
        inds1 = []
        inds2 = []
        for i in range(N):
            if labels[i] == k1:
                inds1.append(i)
            if labels[i] == k2:
                inds2.append(i)
        if (len(inds1) > 0) and (len(inds2) > 0):
            inds12 = []
            for aa in range(len(inds1)):
                inds12.append(inds1[aa])
            for aa in range(len(inds2)):
                inds12.append(inds2[aa])
            L12_old = []
            for i in range(len(inds1)):
                L12_old.append(1)
            for i in range(len(inds2)):
                L12_old.append(2)
            L12 = []
            for i in range(len(inds12)):
                L12.append(0)

            if (len(inds1) < min_cluster_size) or (len(inds2) < min_cluster_size):
                do_merge = True
            else:
                X1 = X[:, inds1]
                X2 = X[:, inds2]
                do_merge, L12 = merge_test(X1, X2, centroids[:, k1 - 1], centroids[:, k2 - 1], covmats[:, :, k1 - 1], covmats[:, :, k2 - 1], isocut_threshold=isocut_threshold)
            if do_merge:
                for i in range(len(inds2)):
                    new_labels[inds2[i]] = k1
                total_num_label_changes += len(inds2)
                clusters_changed_vec[k1 - 1] = True
                clusters_changed_vec[k2 - 1] = True
            else:
                # redistribute
                something_was_redistributed = False
                for i in range(len(inds1)):
                    if L12[i] == 2:
                        new_labels[inds1[i]] = k2
                        total_num_label_changes += 1
                        something_was_redistributed = True
                for i in range(len(inds2)):
                    if L12[len(inds1) + i] == 1:
                        new_labels[inds2[i]] = k1
                        total_num_label_changes += 1
                        something_was_redistributed = True
                if something_was_redistributed:
                    clusters_changed_vec[k1 - 1] = True
                    clusters_changed_vec[k2 - 1] = True
    clusters_changed = []
    for k in range(Kmax):
        if clusters_changed_vec[k]:
            clusters_changed.append(k + 1)
    return new_labels, clusters_changed, total_num_label_changes

def p2_compute_centroid(X, indices):
    M = X.shape[0]
    ret = np.zeros((M,), dtype=np.float32)
    count = 0
    for i in range(len(indices)):
        for m in range(M):
            ret[m] += X[m, indices[i]]
        count += 1
    if count > 0:
        for m in range(M):
            ret[m] /= count
    return ret

def p2_compute_max_distance(centroid, X, indices):
    M = X.shape[0]
    max_dist = 0
    for i in range(len(indices)):
        dist = 0
        for m in range(M):
            val = centroid[m] - X[m, indices[i]]
            dist += val * val
        dist = np.sqrt(dist)
        if dist > max_dist:
            max_dist = dist
    return max_dist

def p2_randsample(N: int, K: int):
    # Note we are not actually randomizing here. There's a reason, I believe.
    inds = []
    for a in range(K):
        inds.append(a)
    return inds

def merge_test(X1, X2, centroid1, centroid2, covmat1, covmat2, isocut_threshold: float):
    M = X1.shape[0]
    N1 = X1.shape[1]
    N2 = X2.shape[1]
    L12 = []
    for i in range(N1 + N2):
        L12.append(1)
    if (N1 == 0) or (N2 == 0):
        print("Error in merge test: N1 or N2 is zero.")
        return True

    # std::vector<float> centroid1 = compute_centroid(M, N1, X1);
    # std::vector<float> centroid2 = compute_centroid(M, N2, X2);

    V = np.zeros((M,), dtype=np.float32)
    for m in range(M):
        V[m] = centroid2[m] - centroid1[m]

    avg_covmat = (covmat1 + covmat2) / 2
    
    inv_avg_covmat = np.linalg.inv(avg_covmat)

    V = inv_avg_covmat @ V

    sumsqr = 0
    for m in range(M):
        sumsqr += V[m] * V[m]

    if sumsqr:
        for m in range(M):
            V[m] /= np.sqrt(sumsqr)
    
    projection1 = np.zeros((N1,), dtype=np.float32)
    projection2 = np.zeros((N2,), dtype=np.float32)
    projection12 = np.zeros((N1 + N2,), dtype=np.float32)

    for i in range(N1):
        tmp = 0
        for m in range(M):
            tmp += V[m] * X1[m, i]
        projection1[i] = tmp
        projection12[i] = tmp
    for i in range(N2):
        tmp = 0
        for m in range(M):
            tmp += V[m] * X2[m, i]
        projection2[i] = tmp
        projection12[N1 + i] = tmp
    
    dipscore, cutpoint = isocut5_slow(projection12)

    if dipscore < isocut_threshold:
        do_merge = True
    else:
        do_merge = False
    
    for i in range(N1 + N2):
        if projection12[i] < cutpoint:
            L12[i] = 1
        else:
            L12[i] = 2

    return do_merge, L12