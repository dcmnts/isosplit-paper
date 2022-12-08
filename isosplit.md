# ISO-SPLIT: A Non-Parametric Method for Unimodal Clustering

Note that isosplit_arxiv.tex is in this directory

## Abstract

A limitation of many clustering algorithms is that they require adjustable parameters to be tuned for each application or dataset, making them unsuitable for use in automated procedures that involve clustering as a processing step. Some techniques require an initial estimate of the number of clusters, while density-based techniques typically require a scale parameter. Other parametric methods, such as mixture modeling, make assumptions about the underlying cluster distributions. Here we introduce ISO-SPLIT, a non-parametric clustering method that does not require adjustable parameters nor parametric assumptions about the underlying cluster distributions. The only assumption is that clusters are unimodal and separated from one another by separating hyperplanes of relatively lower density. The technique uses a variant of Hartigan's dip statistic and isotonic regression as its kernel operation. Using simulations, we compare ISO-SPLIT with standard methods including k-means, DBSCAN, and Gaussian mixture methods. The method was developed to tackle the "spike sorting" problem in electrophysiology and is well-suited for low-dimensional datasets with many observations. ISO-SPLIT has been a key component of the MountainSort spike sorting algorithm and its source code is freely available.

## Introduction

<!-- rewritten -->
The purpose of unsupervised data clustering is to automatically partition a set of datapoints into clusters that reflect the underlying structure of the data. In this work, we focus on scenarios where the datapoints lie in a low-dimensional space, there are many observations, and each cluster is characterized by a dense core area surrounded by regions of lower density. This situation is common in our motivating application of spike sorting of neural firing events in electrophysiology, where this type of structure has been observed experimentally [@tiganj, @vargas].

<!-- rewritten -->
Most clustering algorithms have the limitation of requiring careful adjustment of parameters. For example, this is a problem for k-means [@kmeans] where the choice of the number of clusters ($K$) can be difficult. This is especially true for large datasets with many clusters, as in the spike sorting application. The output of k-means also depends heavily on the initialization of the algorithm, and it is often necessary to run it multiple times to find a globally optimal solution. K-means has other limitations as well, such as rigid assumptions about cluster populations and variances, and the tendency to artificially split larger clusters and merge smaller ones.

<!-- rewritten -->
Gaussian mixture modeling (GMM) is a flexible clustering algorithm that is usually solved using expectation-maximization (EM) [@em]. Many variations exist, some of which are outlined in Chapter 11 of [@murphy]. Unlike k-means, GMM allows each cluster to be modeled using a multivariate normal distribution. Some GMM implementations require knowledge of the number of clusters beforehand (see Chapter 8 of [@mixturemodels], while others consider this as a free variable [@roberts1998bayesian] . The main disadvantage of GMM is that it assumes clusters are well modeled by Gaussian distributions, which may not always be the case. Additionally, like k-means, GMM can be difficult to optimize when the number of clusters is large. Recently, mixture models with skew non-Gaussian components have been developed [@skewGMM, @skewGMM2], but these models are more complex with additional free parameters and may be even more difficult to optimize.

<!-- rewritten -->
Hierarchical clustering (see Chapter 15 of [@zaki-book]) does not demand that the number of clusters be specified beforehand. However, the output is a dendrogram rather than a partition, preventing it from being directly applicable to our motivating example. To obtain a partition from the dendrogram, a criteria for cutting the binary tree must be specified, similar to specifying $K$ in k-means. In addition, other parameters are typically required in agglomerative methods to determine which clusters are joined in each iteration. A further issue is that the time complexity of hierarchical clustering is at least $O(n^2)$, where $n$ is the number of observations (see Sec. 14.2.3 of [@zaki-book].

<!-- rewritten -->
Density-based clustering techniques such as DBSCAN [@dbscan] are useful due to their lack of assumptions about data distributions, allowing them to effectively identify clusters with non-convex shapes. However, these techniques also involve free parameters. DBSCAN requires two parameters to be adjusted depending on the application, including $\epsilon$, a scale parameter. The algorithm is particularly sensitive to the $\epsilon$ parameter in higher dimensions. Additionally, if clusters in the dataset have varying densities, no choice of the $\epsilon$ parameter will be able to work on the entire dataset.

<!-- rewritten -->
Other density-based techniques such as mean-shift [@mean-shift] involve the initial step of constructing a continuous non-parametric probability density function (Chapter 15 of [@zaki-book]. The basic version of the kernel density estimation method [@kernel-density-function-1, @kernel-density-function-2] requires specifying a spatial scale parameter (the bandwidth), which is subject to the same problem as DBSCAN. Variations of this method can automatically determine an optimal, spatially-dependent bandwidth [@silverman-density-estimation]. There is a variety of density-estimation methods to choose from [@rodriguez-clustering], however, they are often dependent on adjustable distance parameters. In general, these methods become computationally intractable in higher dimensions (even 4 dimensions is challenging).

<!--TODO: Other clustering methods -->
Affinity Propagation Clustering
K-medoids clustering
Spectral clustering
density-peak clustering (Rodriguez Laio)

<!-- rewritten -->
In this article, we present a density-based, scale-independent clustering technique that is suitable for situations where clusters are expected to be unimodal and can be separated from one another by hyperplanes. A cluster is considered unimodal if it is derived from a distribution that has a single peak of maximum density when projected onto any line. Thus, our assumption is that when any two adjacent clusters are projected onto the normal of a dividing hyperplane, they will form a 1D bimodal distribution with a split-point of lower density at the point of hyperplane intersection. Loosely speaking, this is the case when clusters are well-spaced and have convex shapes.

<!-- rewritten -->
In addition to being density-based, our technique has elements of both agglomerative hierarchical clustering and involves the EM-style iterative approach of k-means. It uses a non-parametric procedure to separate 1D distributions based on a modified Hartigan's dip statistic [@hartigan1985dip, @other_hartigan1985dip] and isotonic regression, which don't require any adjustable parameters (with the exception of a statistical significance threshold). In particular, no scale parameter is needed for density estimation. Moreover, since the core step of each iteration is 1D clustering applied to projections of data subsets onto lines, it overcomes the curse of dimensionality (the tradeoff being that we cannot handle clusters of arbitrary shape).

<!-- TODO: rewrite -->
This paper is organized as follows. First we describe an algorithm for splitting a 1D sample into unimodal clusters. This procedure forms the basis of the $p$-dimensional clustering technique, ISO-SPLIT, defined in Section 3. Simulation results are presented in Section 4, comparing ISO-SPLIT with three standard clustering techniques. In addition to quantitative comparisons using a measure of accuracy, examples illustrate situations where each algorithm performs best. The fifth section is an application of the algorithm to spike sorting of neuronal data. Next we discuss computational efficiency and scaling properties. Finally, Section 8 summarizes the results and discusses the limitations of the method. The appendices cover implementation details for isotonic regression, generation of synthetic datasets for simulations, and provide evidence for insensitivity to parameter adjustments.

## Clustering in one dimension

<!-- rewritten -->
Any approach overcoming the above limitations must at least be able to do so in the simplest, 1D case. Here we present a non-parametric approach to 1D clustering utilizing a statistical test for unimodality and isotonic regression. This procedure will then be used as the basis for the more general situation ($p\geq2$) described in Section [{isosplit-algorithm}].
  
<!-- rewritten -->
Clustering 1D data is special due to the fact that the input data can be sorted. The task then becomes finding the $K-1$ cut points (real numbers between adjacent datapoints) that determine the $K$ clusters. We assume that the clusters are unimodal, meaning that the density is lower between adjacent clusters. For simplicity we will describe an algorithm for deciding whether there is one cluster, or more than one cluster. In the latter case, a single cut point is determined representing the boundary separating one pair of adjacent clusters. Once the data have been split, the same algorithm may then be applied recursively on the left and right portions leading to further subdivisions, converging when no more splitting occurs. Thus, in addition to being the basis for higher dimensional clustering, the algorithm described here may also be used as a basis for general 1D clustering.

<!-- rewritten -->
We assume that two adjacent unimodal clusters are always separated by a region of lower density. This means that if $a_1$ and $a_2$ are the centers of two adjacent 1D clusters, there exists a cut point $a_1 < c < a_2$ such that the density near $c$ is significantly less than the densities near both $a_1$ and $a_2$. To determine this cut point, we must define the notion of density near a point. The common approach is to use either histogram binning or kernel density methods, but these require us to choose a length scale $\epsilon$. We aim to avoid this step.

Instead we use a variant of Hartigan's dip test for unimodality. First we will describe the Hartigan dip test. Let $x_1 < \dots < x_n$ be the sorted (assumed distinct) real numbers (input data samples). The null hypothesis is that the set $X=\{x_j\}$ is an independent sampling of a unimodal probability density $f(x)$, which by definition is increasing on $[-\infty,c]$ and decreasing on $[c,\infty]$. Let

$$S_X(x)=\#\{j:x_j\leq x\}$$

be the empirical distribution function of our data, and let F be a unimodal cumulative distribution function that approximates $S_X$. The normalized Kolmogorov-Smirnov distance between these two is given by

$$D_{X, F} = \frac{1}{\sqrt{n}}\sup_x{|F(x)-S_X(x)|}$$

In Hartigan's method, the dip statistic is
$$\tilde{D}_X=\inf_F{D_{X,F}},$$
which is used to either reject or accept the unimodality null hypothesis. Hartigan's original paper outlines a method for implementing this infimum, though it is far from straightforward. The time complexity has not been carefully evaluated, and there are very few, if any, open source software packages that implement it. But there are other reasons why we use a variant of Hartigan's test rather than the original algorithm. First, Hartigan's test only produces an accept/reject result, and does not supply an optimal cut point separating the clusters, which we need for the clustering. Second, there is a problem with Hartigan's test in the case where the number of points in one cluster (say on the far left) is very small compared with the total size $n$, as described in more detail below.

Here we define a slightly different statistic
$$D_X=D_{X,F_X}$$
where the approximation $F_X$  of $S_X$ is determined by down-up isotonic regression as described in Appendix [{appendixUpdown}]. Roughly speaking, $F_X$ results from a least-squares approximation of the emperical density function by a function that is monotonically increasing to left of the cutpoint, and monotonically decreasing to the right.

<!-- TODO: check if this paragraph is correct compared with the actual implementation -->
<!-- rewritten -->
As mentioned above, Hartigan's dip test has a flaw when the number of points in one cluster (say on the far left) is much smaller than the total size $n$. This is due to the fact that the absolute size of the dip in the emperical distribution only depends on the relatively small amount of data near the interface between the two cluster, whereas the test for rejection becomes more rigorous with increasing $n$. To address this, we perform a series of dip tests of sizes $4,8,16,32,\dots, n$. We compare two tests for each size, one starting from the left and one starting from the right. If the unimodality hypothesis is rejected in any one of these tests, the algorithm stops, the null hypothesis is rejected, and the cut point for that segment is returned. Otherwise, the unimodality hypothesis is accepted.

## Clustering in more than one dimension using 1D projections

In this section we address the $p$-dimensional situation ($p\geq2$) and describe an iterative procedure, termed ISO-SPLIT, in which the 1D routine is repeated as a kernel operation. The decision boundaries are less restrictive than $k$-means which always splits space into Voronoi cells with respect to the centroids, as illustrated in [{fig:decision_boundaries}].

The proposed procedure is outlined in Algorithm [{alg:main_algorithm}]. The input is a collection of $n$ points in $\mathbb{R}^p$, and the output is the collection of corresponding labels (or cluster memberships). The approach is similar to agglomerative hierarchical methods in that we start with a large number of clusters (output of *InitializeLabels* and iteratively reduce the number of clusters until convergence. However, in addition to merging clusters, the algorithm may also redistribute datapoints between adjacent clusters. This is in contrast to agglomerative hierarchical methods. At each iteration, the two [{closest}] clusters (that have not yet been handled) are selected and all datapoints from the two sets are projected onto a line orthogonal to the proposed hyperplane of separation. The 1D split test from the previous section is applied (see above) and then the points are redistributed based on the optimal cut point, or if no statistically significant cut point is found, the clusters are merged. This procedure is repeated until all pairs of clusters have been handled.

<!-- Algorithm goes here -->

The best line of projection may be chosen in various ways. The simplest approach is to use the line connecting the centroids of the two clusters of interest. Although this choice may be sufficient in most situations, the optimal hyperplane of separation may not be orthogonal to this line. Instead, the approach we used in our implementation is to estimate the covariance matrix of the data in the two clusters (assuming Gaussian distributions with equal variances) and use this to whiten the data prior to using the above method. The function *GetProjectionDirection* in [{alg:main_algorithm}] returns a unit vector $V$ representing the direction of the optimal projection line, and the function *Project* simply returns the inner product of this vector with each datapoint.

Similarly, there are various approaches for choosing the closest pair of clusters at each iteration *FindClosestPair*. One way is to minimize the distance between the two cluster centroids. Note, however, that we don't want to repeat the same 1D kernel operation more than once. Therefore, the closest pair that has not yet been handled is chosen. In order to avoid excessive iterations we used a heuristic for determining whether a particular cluster pair (or something very close to it) had been previously attempted.

The function *InitializeLabels* creates an initial labeling (or partitioning) of the data. This may be implemented using the $k$-means algorithm with the number of initial clusters chosen to be much larger than the expected number of clusters in the dataset, the assumption being that the output should not be sensitive once $K_\text{initial}$ is large enough (see Appendix [{appendixSensitivity}]). For our tests we used the minimum of $20$ and four times the true number of clusters. Since datasets may always be constructed such that our choice of $K_\text{initial}$ is not large enough, we will seek to improve this initialization step in future work.

The critical step is *ComputeOptimalCutpoint*, which is the 1D clustering procedure described in the previous section, using a threshold of $\tau_n=\alpha/\sqrt{n}$.

## Figuring out what the implemented algorithm actually does

```c++
dipscore_out, cutpoint_out = isocut5(samples)

// N = number of samples
samples_sorted = sort(samples)

//////////////////////////////////////////////////////////
// SUBSAMPLING
// num_bins is sqrt(N/2) * factor
num_bins_factor = 1
num_bins = ceil(sqrt(N * 1.0 / 2) * num_bins_factor)

num_bins_1 = ceil(num_bins / 2) // left bins
num_bins_2 = num_bins - num_bins_1 // right bins

// I guess this is the same as num_bins
num_intervals = num_bins_1 + num_bins_2

// what is this? the number of samples in each interval?
intervals[num_intervals]
intervals[i] = i + 1 for i < num_bins_1
intervals[num_intervals - 1 - i] = i + 1 for i < num_bins_2

alpha = (N - 1) / sum(intervals)

// now we scale this such that sum(intervals) = N - 1
intervals = intervals * alpha

// number of subsamples
N_sub = num_intervals + 1

// indices of the subsampled samples
inds[N_sub]
inds[0] = 0
inds[i + 1] = inds[i] + intervals[i]

// These are the subsampled samples
X_sub[N_sub]
X_sub[i] = samples_sorted[floor(inds[i])]
//////////////////////////////////////////////////////////

spacings[N_sub - 1]
spacings[i] = X_sub[i + 1] - X_sub[i]

multiplicities[N_sub - 1]
multiplicities[i] = inds[i + 1] - inds[i]

densities[N_sub - 1]
densities[i] = multiplicities[i] / spacings[i]

densities_unimodal_fit = jisotonic5_updown(densities, multiplicities)

densities_resid[N_sub - 1]
densities_resid[i] = densities[i] - densities_unimodal_fit[i]

densities_unimodal_fit_times_spacings = densities_unimodal_fit * spacings

peak_index = argmax(densities_unimodal_fit)

// dipscore_out
dipscore_out, critical_range_min, critical_range_max = compute_ks5(multiplicities, densities_unimodal_fit_times_spacings, peak_index)

critical_range_length = critical_range_max - critical_range_min + 1;

densities_resid_on_critical_range[critical_range_length]
densities_resid_on_critical_range[i] = densities_resid[critical_range_min + i];

weights_for_downup[critical_range_length]
weights_for_downup[i] = spacings[critical_range_min + i]

densities_resid_fit_on_critical_range
densities_resid_fit_on_critical_range = jisotonic5_downup(densities_resid_on_critical_range, weights_for_downup)

cutpoint_index = argmin(densities_resid_fit_on_critical_range);

// cutpoint_out
cutpoint_out = (X_sub[critical_range_min + cutpoint_index] + X_sub[critical_range_min + cutpoint_index + 1]) / 2
```

```c++
ks_best, critical_range_min, critical_range_max = compute_ks5(counts1, counts2, peak_index)

critical_range_min = 0
critical_range_max = N - 1

double ks_best = -1

// from the left
counts1_left[peak_index + 1]
counts1_left[i] = counts1[i]

counts2_left[peak_index + 1]
counts2_left[i] = counts2[i]

len = peak_index + 1
while (len >= 4) or (len == peak_index + 1)
	ks0 = compute_ks4(counts1_left[0..len-1], counts2_left[0..len-1])
	if ks0 > ks_best
		critical_range_min = 0
		critical_range_max = len - 1
		ks_best = ks0
	len = len / 2

// from the right
counts1_right[N - peak_index]
counts1_right[i] = counts1[N - 1 - i]

counts2_right[N - peak_index]
counts2_right[i] = counts2[N - 1 - i]

len = N - peak_index
while (len >= 4) or (len == N - peak_index)
	ks0 = compute_ks4(counts1_right[0..len-1], counts2_right[0..len-1])
	if ks0 > ks_best
		critical_range_min = N - len
		critical_range_max = N - 1
		ks_best = ks0
	len = len / 2
```

```c++
ks = compute_ks4(counts1, counts2)

sum_counts1 = sum(counts1);
sum_counts2 = sum(counts2);

cumsum_counts1 = 0;
cumsum_counts2 = 0;

max_diff = 0
for i = 0..N-1
	cumsum_counts1 += counts1[i]
	cumsum_counts2 += counts2[i]
	diff = abs(cumsum_counts1/sum_counts1 - cumsum_counts2 / sum_counts2)
	if diff > max_diff
		max_diff = diff
ks = max_diff * sqrt((sum_counts1 + sum_counts2) / 2)
```

Hartigan, J. A., & Hartigan, P. M. (1985). The Dip Test of Unimodality. The Annals of Statistics.

Hartigan, P. M. (1985). Computation of the Dip Statistic to Test for Unimodality. Journal of the Royal Statistical Society. Series C (Applied Statistics), 34(3), 320-325.