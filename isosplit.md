---
title: "Isosplit: A Non-Parametric Method for Unimodal Clustering"
citations-directive: 1
---

# Isosplit: A Non-Parametric Method for Unimodal Clustering

## Abstract

Many clustering algorithms require the tuning of adjustable parameters for each application or dataset, making them unsuitable for automated procedures that involve clustering. Some techniques require an initial estimate of the number of clusters, while density-based techniques typically require a scale parameter. Other parametric methods, such as mixture modeling, make assumptions about the underlying cluster distributions. Here we introduce Isosplit, a non-parametric clustering method that does not require adjustable parameters nor parametric assumptions about the underlying cluster distributions. The only assumption is that clusters are unimodal and separated from one another by hyperplanes of relatively lower density. The technique uses a variant of Hartigan's dip statistic and isotonic regression as its kernel operation. Using simulations, we compared Isosplit with standard methods including k-means, density-based techniques, and Gaussian mixture methods, and found that Isosplit overcomes many of the limitations of these techniques. Our algorithm was developed to tackle the "spike sorting" problem in electrophysiology and is well-suited for low-dimensional datasets with many observations. Isosplit has been in use as the clustering component of the MountainSort spike sorting algorithm and its source code is freely available.

[Introduction](#introduction) |
[Methods](#methods) |
[Results](#results) |
[Discussion](#discussion) |
[Conclusion](#conclusion)

## Introduction

The purpose of unsupervised data clustering is to automatically partition a set of datapoints into clusters that reflect the underlying structure of the data. In this work, we focus on scenarios where datapoints lie in a low-dimensional space, there are many observations, and each cluster is characterized by a dense core area surrounded by regions of lower density. This situation is common in our motivating application of spike sorting of neural firing events in electrophysiology, where this type of structure has been observed experimentally [@tiganj; @vargas].

Most clustering algorithms have the limitation of requiring careful adjustment of parameters. For example, the choice of the number of clusters ($K$) for k-means [@lloyd1982least] can be difficult, particularly for large datasets with many clusters, such as in the spike sorting application. The output of k-means also depends on the initialization of the algorithm, and it is often necessary to run it multiple times to find a globally optimal solution. K-means has other limitations as well, such as rigid assumptions about cluster populations and variances, and the tendency to artificially split larger clusters and merge smaller ones.

Gaussian mixture modeling (GMM) is a flexible clustering algorithm that is usually solved using expectation-maximization (EM) [@em]. Unlike k-means, GMM allows each cluster to be modeled using a multivariate normal distribution. Some GMM implementations require knowledge of the number of clusters beforehand (see Chapter 8 of [@mixturemodels], while others consider this as a free variable [@roberts1998bayesian] . The main disadvantage of GMM is that it assumes clusters are well modeled by Gaussian distributions, which may not always be the case. Additionally, like k-means, GMM can be difficult to optimize when the number of clusters is large. Recently, mixture models with skew non-Gaussian components have been developed [@skewGMM, @skewGMM2], but these models are more complex with additional free parameters and may be even more difficult to optimize.

Hierarchical clustering (see [@murtagh2012algorithms]) does not demand that the number of clusters be specified beforehand. However, the output is a dendrogram rather than a partition, preventing it from being directly applicable to our motivating example. To obtain a partition from the dendrogram, a criteria for cutting the binary tree must be specified, similar to specifying $K$ in k-means. In addition, other parameters are typically required in agglomerative methods to determine which clusters are joined in each iteration. A further issue is that the computation can be slow compared with other algorithms.

Density-based clustering techniques such as DBSCAN [@dbscan] are useful due to their lack of assumptions about data distributions, allowing them to effectively identify clusters with non-convex shapes. However, these techniques also require parameters. For example, DBSCAN requires two parameters to be adjusted depending on the properties of the dataset, including $\epsilon$, a scale parameter. The algorithm is particularly sensitive to $\epsilon$ in dimensions greater than two, and the algorithm is not well suited for more than four dimensions. Additionally, if clusters in the dataset have varying densities, no choice of the $\epsilon$ parameter will be able to work on the entire dataset. The mean-shift [@mean-shift] algorithm overcomes this limitation to an extent by automatically estimating the bandwidth, but it is very slow, and as we will see, suffers from some of the limitations of other algorithms.

Rodriguez-Laio (RL) clustering [@rodriguez-clustering], also known as density-peak clustering, is another density-based clustering technique that identifies representative datapoints based on their large distances to other points with greater density. This method has potential due to its ability to detect clusters without requiring a bandwidth parameter, making it suitable for datasets with varying cluster densities. However, the algorithm requires a threshold for determining which datapoints should be considered cluster representatives and is susceptible to oversplitting when clusters are highly anisotropic.

In this article, we present a new density-based, scale-independent clustering technique that is suitable for situations where clusters are expected to be unimodal and can be separated from one another by hyperplanes. A cluster is considered unimodal if it is derived from a distribution that has a single peak of maximum density when projected onto any line. Thus, our assumption is that when any two adjacent clusters are projected onto the normal of a dividing hyperplane, they will form a 1D bimodal distribution with a split-point of lower density at the point of hyperplane intersection. Loosely speaking, this is the case when clusters are well-spaced and have convex shapes.

In addition to being density-based, our technique has elements of agglomerative hierarchical clustering and involves the EM-style iterative approach of k-means. It uses a non-parametric procedure to separate 1D distributions based on a modified Hartigan's dip statistic [@hartigan1985dip, @hartigan1985algorithm] and isotonic regression [@barlow1972isotonic], which don't require any adjustable parameters (with the exception of a statistical significance threshold). In particular, no scale parameter is needed for density estimation. Moreover, since the core step of each iteration is 1D clustering applied to projections of data subsets onto lines, it overcomes the curse of dimensionality (the tradeoff being that we cannot handle clusters of arbitrary shape).

We first describe an algorithm for splitting a 1D sample into unimodal clusters. This forms the basis of the $p$-dimensional clustering technique, Isosplit. We present simulations comparing Isosplit with standard clustering techniques. We also address computational efficiency and scaling properties. Finally, we summarize the results and discusses the limitations of the method. The appendices cover implementation details for the algorithms and the generation of synthetic datasets.

**TODO: SpectralClustering**: Requires n_clusters. Seems that SC can be slow, esp on non-Gaussian example.

## Methods

### Clustering in one dimension

Any approach overcoming the above limitations must at least be able to do so in the 1D case. Here we present a non-parametric approach to 1D clustering utilizing a statistical test for unimodality and isotonic regression. This procedure will then be used as the basis for the more general situation ($p\geq 2$) described in Section [{isosplit-algorithm}].
  
Clustering of 1D data is special due to the fact that the input data can be sorted. The task then becomes finding the $K-1$ cut points (real numbers between adjacent datapoints) that determine the $K$ clusters. We assume that the clusters are unimodal, meaning that the density is lower between adjacent clusters. We will describe an algorithm for deciding whether there is one cluster, or more than one cluster. In the latter case, a single cut point is determined representing the boundary separating one pair of adjacent clusters. Once the data have been split, the same algorithm may then be applied recursively on the left and right portions leading to further subdivisions, converging when no more splitting occurs. Thus, in addition to being the basis for higher dimensional clustering, the algorithm described here could also be used as a basis for general 1D clustering.

We assume that two adjacent unimodal clusters are always separated by a region of lower density. This means that if $a_1$ and $a_2$ are the core points of two adjacent 1D clusters, there exists a cut point $a_1 < c < a_2$ such that the density near $c$ is significantly less than the densities near both $a_1$ and $a_2$. To determine this cut point, we must define the notion of density near a point. The common approach is to use either histogram binning or kernel density methods, but these require us to choose a length scale $\epsilon$. We aim to avoid this step.

Instead we use a variant of Hartigan's dip test for unimodality. First we will describe the Hartigan dip test. Let $x_1 < \dots < x_n$ be the sorted (assumed distinct) real numbers (input data samples). The null hypothesis is that the set $X=\{x_j\}$ is an independent sampling from a unimodal probability density $f(x)$, which by definition is increasing on $[-\infty,c]$ and decreasing on $[c,\infty]$ for some $c$. Let

$$S_X(x)=\#\{j:x_j\leq x\}$$

be the empirical distribution function of our data, and let F be a unimodal cumulative distribution function that approximates $S_X$. The normalized Kolmogorov-Smirnov distance between these two is given by

$$D_{X, F} = \frac{1}{\sqrt{n}}\sup_x{|F(x)-S_X(x)|}$$

In Hartigan's method, the dip statistic is
$$\tilde{D}_X=\inf_F{D_{X,F}},$$
which is used to either reject or accept the unimodality null hypothesis. Hartigan's original paper outlines a method for implementing this infimum, though it is far from straightforward. The time complexity has not been carefully evaluated, and there are very few, if any, open source software packages that implement it as described. But there are other reasons why we use a variant of Hartigan's test rather than the original algorithm. First, Hartigan's test only produces an accept/reject result, and does not supply an optimal cut point separating the clusters, which we need for the clustering. Second, there is a problem with Hartigan's test in the case where the number of points in one cluster (say on the far left) is very small compared with the total size $n$, as described in more detail below.

Here we define a different statistic
$$D_X=D_{X,F_X}$$
where the approximation $F_X$  of $S_X$ is determined by up-down isotonic regression as shown in [Algorithm 1](#algorithm-1) and described in [Appendix A](#appendix-a). Roughly speaking, $F_X$ results from an approximation of the emperical density function by a function that is monotonically increasing to the left of a critical point, and monotonically decreasing to the right (see [Figure A1](#figure-a1)).

As mentioned above, Hartigan's dip test has a flaw when the number of points in one cluster (say on the far left) is much smaller than the total size $n$. This is due to the fact that the absolute size of the dip in the empirical distribution only depends on the relatively small amount of data near the interface between the two cluster, whereas the test for rejection becomes more rigorous with increasing $n$ (note the normalizing factor of $\frac{1}{\sqrt{n}}$). To address this, we perform a series of dip tests of sizes $\lfloor n/2 \rfloor, \lfloor n/4 \rfloor, \lfloor n/8 \rfloor, \dots$. We compare two tests for each size, one starting from the left and one starting from the right. If the unimodality hypothesis is rejected in any one of these tests, the null hypothesis is rejected. Otherwise, the unimodality hypothesis is accepted. A more detailed description of this procedure is provided in Appendix XX. This procedure is encapsulated in the `ks_adj` function in [Algorithm 1](#algorithm-1).

In the case where the null hypothesis is rejected, a cutpoint must be found. This is obtained using down-up isotonic regression on the density residual as given in [Algorithm 1](#algorithm-1) and shown in Figurl [{ISOCUT}]. Further algorithmic details are provided in the appendix.

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-a1"></a>

![isocut_demo](https://user-images.githubusercontent.com/3679296/210560407-104e0bb3-ed4f-49d7-94f8-e31d85f647f6.svg)
<!--
name: isocut_demo.svg
-->
<figcaption>

Figure A1: Illustration of the Isocut algorithm for testing for unimodality in 1D and determining an optimal cutpoint. (A) histogram of a simulated bimodal distribution. (B) Estimated log density with unimodal fit obtained from up-down isotonic regression. (C) Residual log density with fit from down-up isotonic regression to determine the cutpoint at the minimum.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="algorithm-1"></a>

```python
# isocut algorithm
dipscore, cutpoint = isocut(samples):
    # define spacings, multiplicities
    s := diff(samples)
    m := [1, 1, ..., 1]

    # compute log densities
    d := log(m / s)

    # up-down isotonic regression
    d_uni_fit := isotonic_updown(d)

    # modified ks statistic
    peak_index := argmax(d_uni_fit)
    dipscore, critical_range := ks_adj(m, exp(d_uni_fit) * s, peak_index)

    # downup isotonic regression
    d_resid := d - d_uni_fit
    d_resid_fit := isotonic_downup(d_resid[crit_rng])

    # Determine cutpoint
    cp_ind := argmin(d_resid_fit)
    cutpoint :=
	    (samples[crit_rng][cp_ind] + samples[crit_rng][cp_ind + 1]) / 2
    
    return dipscore, cutpoint
```
<figcaption>

Algorithm 1: Isocut tests whether a 1D sampling of datapoints arises from a multi-modal distribution. In the case where the unimodality hypothesis is rejected, an optimal cutpoint is found that separates regions of relatively high density. Details on the `isotonic_updown`, `isotonic_downup`, and `ks_adj` functions are provided in the [Appendix A](#appendix-a) and [Appendix XXXXX](#appendix-xxxxx)).

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

### Clustering in more than one dimension using 1D projections

In this section we address the $p$-dimensional situation ($p\geq 2$) and describe an iterative procedure, termed Isosplit, in which the 1D routine is repeated as a kernel operation. The decision boundaries are less restrictive than k-means which always splits space into Voronoi cells with respect to the centroids, as illustrated in [Figure B1](#figure-b1).

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-b1"></a>

![decision_boundaries](https://user-images.githubusercontent.com/3679296/207963490-a9195e1e-88a3-4028-a7ac-a022cb0946cc.png)
<!--
name: decision_boundaries.png
-->
<figcaption>

Figure B1. Unlike k-means, the decision boundaries between Isosplit clusters occur at regions of lower density.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

The proposed procedure is outlined in [Algorithm 2](#algorithm-2). The input is a collection of $n$ points in $\mathbb{R}^p$, and the output is the collection of corresponding labels (or cluster memberships). The approach is similar to agglomerative hierarchical methods in that we start with a large number of clusters (output of `initial_parcellation`) and iteratively reduce the number of clusters until convergence. However, in addition to merging clusters, the algorithm may also redistribute datapoints between adjacent clusters. This is in contrast to agglomerative hierarchical methods. At each iteration, pairs of nearby clusters are selected and, for each pair, all datapoints from the two sets are projected onto a line orthogonal to the proposed hyperplane of separation. The 1D split test from the previous section is applied to the projected data (see above) and then the points are redistributed based on the optimal cut point, or if no statistically significant cut point is found, the clusters are merged. This procedure is repeated until all pairs of clusters have been handled. This process is illustrated in [Figure B2](#figure-b2).

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="algorithm-2"></a>

```python
clusters = isosplit(X):
    # initial clustering
    clusters := initial_parcellation(X)

    # significance threshold
    threshold := 2

    # iterate until nothing changes
    something_changed := True
    while something_changed:
        something_changed := False

        # determing pairs of clusters to compare
        # based on mutual closest neighbors
        # that have not already been compared
        pairs := pairs_to_compare(clusters) # uses comparison history
        for pair in pairs:
            # project onto a 1D subspace
            V = projection_direction(pair.C1, pair.C2)
            A1 := project(pair.C1, V)
            A2 := project(pair.C2, V)

            # test for merging/splitting using isocut
            dipscore, reassignments := merge_test(A1, A2)

            if dipscore > threshold:
                # reassign points if we have rejected the unimodality hypothesis
                clusters, changed := reassign_datapoints(clusters, pair, reassignments)
                if changed:
                    something_changed := True
            else:
                # merge if we have accepted the unimodality hypothesis
                clusters := merge_clusters(clusters, pair)
                something_changed := True

    return clusters
```
<figcaption>

Algorithm 2. Isosplit is a clustering approach that iteratively merges and splits nearby clusters based on unimodality tests along 1D directions of projection.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

With each cluster pair comparison, the direction of projection may be chosen in various ways. The simplest approach is to use the line connecting the centroids of the two clusters of interest. Although this choice may be sufficient in most situations, the optimal hyperplane of separation may not be orthogonal to this line. Instead, the approach we used in our implementation is to estimate the covariance matrix of the data in the two clusters (assuming Gaussian distributions with equal variances) and use this to whiten the data prior to using the above method. The function `projection_direction` in [Algorithm 2](#algorithm-2) returns a unit vector `V` representing the direction of the optimal projection line, and the function `project` simply returns the inner product of this vector with each datapoint.

There are also various choices for selecting pairs of clusters at each iteration (`pairs_to_compare` in [Algorithm 2](#algorithm-2)). One way is to select the two clusters with a minimal distance between the centroids, and then pick the next closest two, and so on. However, note that we don't want to repeat the same 1D kernel operation more than once. Therefore, the mutually closest pairs that have not yet been handled are chosen at each iteration. Further details are provided in the appendix.

The function `initial_parcellation` creates an initial labeling (or partitioning) of the data. This may be implemented using the $k$-means algorithm with the number of initial clusters chosen to be much larger than the expected number of clusters in the dataset, the assumption being that the output should not be sensitive once $K_\text{initial}$ is large enough (see Appendix [{appendixSensitivity}]). For our tests we used a method that partitioned the dataset into parcels of a target size of $10$ datapoints each without exceeding $K=200$ parcels.

The critical step is `merge_test`, which is the isocut procedure described in the previous section, using a threshold of 2 for the dipscore.

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-b2"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://ebcb6b4cda6f756f79dcc20c7c09f6d6b2ad0372&label=Bluster:%20Isosplit%20demo
<!--
height: 500
-->
<figcaption>

Figure B2. Iterations of the Isosplit algorithm on a dataset with four clusters. Use the controls on the left to step through the iterations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

## Results

To highlight scenarios where Isosplit overcomes the limitations of other methods, as well as scenarios that highlight its limitations, we evaluated the accuracy of Isosplit and various standard algorithms using simulated datasets. We selected optimal parameters for the non-Isosplit algorithms based on the known simulation parameters (e.g., the number of clusters for k-means). Isosplit, on the other hand, does not require any user-defined parameters.

The following standard algorithms were evaluated: Agglomerative clustering (Agg) from scikit learn with default parameters and known number of clusters; DBSCAN from scikit learn with optimal scale parameter corresponding to the simulated datasets; Gaussian Mixture Model (GMM) with `covariance_type='full'` and known number of clusters; K-means with known number of clusters; Rodriguez-Laio (RL) or density-peak clustering with implementation described in the appendix and known number of clusters (TODO: explain that more than just the number of clusters were provided to the algorithm); and Spectral clustering (Spect) with `assign_labels='discretize'` and known number of clusters.

We used the following formula when reporting the accuracy of a clustering compared with ground truth:

$$a = \frac{1}{K}\sum_{k=1}^K\max_{j}{\frac{\#(C_k \cap D_k^\prime)}{\#(C_k \cup D_j)}}$$

where $C_k$ is a ground-truth cluster and $D_j$ is a cluster in the clustering being evaluated.

[Unequal variances](#unequal-variances) |
[Anisotropic clusters](#anisotropic-clusters) |
[Non-Gaussian clusters](#non-gaussian-clusters) |
[Packed clusters](#packed-clusters) |<br />
[Higher dimensions](#higher-dimensions) |
[Small clusters](#small-clusters) |
[Non-unimodal examples](#non-unimodal-examples)


### Unequal variances

K-means clustering assumes equal variances for the clusters, which leads to incorrect decision boundaries when clusters have unequal variances ([Figure B1](#figure-b1)). The error is most pronounced when the variance mismatch is large and when clusters are partially overlapping.  Isosplit is less likely to suffer from this problem due to its use of a decision boundary at the hyperplane of lowest density between the clusters.

To illustrate this, we simulated two clusters drawn from spherical multivariate Gaussian distributions in 2D with varying separation distances between the clusters. In each case, the sizes of the two clusters matched, but the standard deviations differed by a factor of 10 ($\sigma_1=1$; $\sigma_2=\frac{1}{10}$). The results are shown in Figures [UV1](#figure-uv1) and [UV2](#figure-uv2).

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-uv1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://d764c70730a81d0e6fcd16eadcb2c4106352f3bc&label=Bluster:%20Unequal%20variances&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22Spect*%22],%22ds%22:26}
<!--
height: 700
-->
<figcaption>

Figure UV1: Performance of Isosplit compared with other algorithms for two clusters of unequal variance with varying separation distances. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-uv2"></a>

https://figurl.org/f?v=gs://figurl/vegalite-2&d=sha1://44257c70cfb3948bf51b1474855d97d925ddbff1&label=Accuracy%20vs.%20separation%20for%20unequal%20variances%20simulation
<!--
height: 550
-->
<figcaption>
Figure UV2. Average accuracies for the various clustering algorithms as a function of separation distance in the unequal variances simulation. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters).
</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

The results show that GMM performs best, as expected since the clusters were drawn from Gaussian distributions and the number of components was known. In the non-Isosplit cases, optimal parameters were used (e.g., K=2 for k-means), whereas Isosplit does not require any parameters to be set. Generally, Isosplit performed better than the non-GMM methods when the clusters overlapped to a moderate extent. The decision boundary for k-means was incorrect due to the unequal variances between the two clusters, and DBSCAN had trouble due to the varying densities of the clusters, making it difficult to choose an ideal scale parameter. As expected, Isosplit did not detect more than one distinct cluster for low separation distances.

### Anisotropic clusters

Another assumption of k-means is that clusters are spherical, or isotropic. Because its cost function is the sum of squared distances to the cluster centers, k-means can favor splitting anisotropic clusters in a direction of elongation rather than separating distinct clusters. Since it does not try to minimize any such metric, Isosplit does not have this problem as it will split along directions where there is a density dip, regardless of anisotropic shape.

To illustrate this, we simulated three Gaussian clusters in 2D, one spherical, and two having an anisotropy factor of 8:1. As in the unequal variances example, the separation distances were varied. The results are shown in [Figure AC1](#figure-ac1). Isosplit generally performed equal to or better than all other algorithms for sufficiently large separation distances. At a separation distance of 4.5, several of the algorithms (Agg, GMM, K-means, RL) favored splitting the anisotropic clusters along the direction of elongation. DBSCAN struggled for lower separation distances due to the variation in density in this example.

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-ac1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://9d253a4d61cb158a8148db15f6e6e81d11468376&label=Bluster:%20Anisotropic&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22SC*%22],%22ds%22:24}
<!--
height: 700
-->
<figcaption>

Figure AC1. Performance of clustering algorithms for three clusters, one spherical and two anisotropic, with varying separation distances. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-ac2"></a>

https://figurl.org/f?v=gs://figurl/vegalite-2&d=sha1://88518091b35c3dd0a94d981cd952b7f4975f8570&label=Accuracy%20vs.%20separation%20for%20anisotropic%20simulation
<!--
height: 550
-->
<figcaption>

Figure AC2. Average accuracies for the various clustering algorithms as a function of separation distance in the anisotropic clusters simulation. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters).

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

## Non-Gaussian clusters

Both k-means and GMM assume that clusters are Gaussian distributed. When a cluster comes from a skewed distribution, the representative points are pulled in the skewed direction which results in incorrect decision boundaries. Isosplit does not make the Gaussian assumption, and works with both skewed and symmetric distributions, provided they are unimodal. [Figure NG1](#figure-ng1) demonstrates this for two simulated clusters, with the cluster on the right being skewed right. In our simulations, Isosplit and RL performed much better than the other methods.

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-ng1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://07cd39063379e1f6a4ed5fe6d2238ba32d92e8a6&label=Bluster%3A%20Non-Gaussian
<!--
height: 700
-->
<figcaption>

Figure NG1: Performance of clustering algorithms for a pair of clusters, one of which is non-Gaussian and skewed right. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

## Packed clusters

To illustrate the performance of Isosplit and other algorithms for datasets with larger numbers of clusters, we simulated ten closely-packed Gaussian clusters with varying separation distances. The method for generating the cluster centroids and covariance matrices is described in [Appendix P](#appendix-p). The results are presented in Figures [PC1](#figure-pc1) and [PC2](#figure-pc2).

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-pc1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://2f0a7714888ff93f87c44841260f59c33dd7b285&label=Bluster:%20Many%20clusters&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22Spect*%22],%22ds%22:9}
<!--
height: 700
-->
<figcaption>

Figure PC1. Performance of clustering algorithms for simulations of ten closely-packed clusters, with varying separation distances. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-pc2"></a>

https://figurl.org/f?v=gs://figurl/vegalite-2&d=sha1://2c81b244396553b6b6c1f9479d9b3dc42c9b31b8&label=Accuracy%20vs.%20separation%20for%20the%20many%20clusters%20simulation
<!--
height: 550
-->
<figcaption>

Figure PC2. Average accuracies for the various clustering algorithms as a function of separation distance in the packed clusters simulation. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters).

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

### Higher dimensions

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-pc1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://4aa2b7b4d000262bbb09b7474815525b9a823a40&label=Bluster:%20Higher%20dimensions&s={%22algs%22:[%22Agg*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22],%22ds%22:12}
<!--
height: 700
-->
<figcaption>

Figure HD1. Performance of clustering algorithms for simulations of ten closely-packed clusters, with varying number of dimensions. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-hd2"></a>

https://figurl.org/f?v=gs://figurl/vegalite-2&d=sha1://b473ca9b355c8405816bf0aabe9987158c3dad0e&label=Accuracy%20vs.%20separation%20for%20the%20higher%20dimensions%20simulation
<!--
height: 550
-->
<figcaption>

Figure HD2. Average accuracies for the various clustering algorithms as a function of number of dimensions in the higher dimensions simulation. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters).

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

### Small clusters

Isosplit is designed to work best with relatively large clusters, or clusters with large numbers of datapoints. This is because the dip statistic of Isocut requires a sufficiently large number of samples. Figures [SC1](#figure-sc1) and [SC2](#figure-sc2) show the results of simulations of two spherical Gaussian clusters with moderate separation and varying cluster sizes. As expected, Isosplit is not able to separate the clusters for small sizes (under 50 datapoints or so).

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-sc1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://3002cb2d65c972848e2cffd19032d2b9e9ea985f&label=Bluster%3A%20Small%20clusters
<!--
height: 700
-->
<figcaption>

Figure SC1. Performance of clustering algorithms for simulations of two clusters with varying sizes (numbers of datapoints). Isosplit is not designed to handle clusters with many datapoints.Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-sc2"></a>

https://figurl.org/f?v=gs://figurl/vegalite-2&d=sha1://05de816255cffca1415bc6d493af93af0c8f9a3a&label=Accuracy%20vs.%20cluster%20size%20for%20small%20clusters%20simulation
<!--
height: 550
-->
<figcaption>

Figure SC2. Average accuracies for the various clustering algorithms as a function of clusters size. Isosplit is not designed to handle clusters with many datapoints. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters).

</figcaption>
</figure>
<!--------------------------------------------------------------------------------------------->

### Non-unimodal examples

To illustrate scenarios where Isosplit performs poorly, we applied Isosplit and other standard algorithms to a collection of datasets with non-unimodal clusters. The results are shown in [Figure NU1](#figure-nu1).

<!--------------------------------------------------------------------------------------------->
<figure>
<a name="figure-nu1"></a>

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://216953ae0558cfe0e5985f35da381b302dd36c68&label=Bluster:%20clustering-benchmark-datasets&s={%22algs%22:[%22Agg*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22],%22ds%22:0}
<!--
height: 700
-->
<figcaption>

Figure NU1. Miscellaneous non-unimodal examples. Isosplit is not designed to handle these cases.

</figcaption>
</figure>

## Discussion

We have shown that, for the target application, our new technique produces results that match or exceed the accuracy with those of standard clustering techniques. Most notably, it excels when clusters are non-Gaussian with varying populations, orientations, spreads, and anisotropies. Yet the key advantage of Isosplit is that it does not require selection of scale parameters nor the number of clusters. This is very important in situations where manual processing steps are to be avoided, for example when minimizing human bias or in order to increment repeatability Automation is also critical when hundreds of clustering runs must be executed within a single analysis, e.g., applications of spike sorting with large electrode arrays.

While Isosplit outperforms standard methods in situations satisfying the assumptions of the method, the algorithm has general limitations and is not suited for all contexts. Because Isosplit depends on statistically significant density dips between clusters, erroneous merging occurs when clusters are positioned close to one another (see the packed-clusters simulation). Certainly this is a challenging scenario for all algorithms, but k-means or mixture models are better suited to handle these cases. On the other hand, if the underlying density has dips which separate clusters, ISO-SPLIT will find them for sufficiently large $n$.

Our theory depends on the assumption that the data arise from a continuous probability distribution. While no particular noise model is assumed, we do assume that, after projection onto any 1D space, the distribution is locally well approximated by a uniform distribution. This condition is satisfied for any smooth probability distribution. In particular, it guarantees that no two samples have exactly the same value (which could lead to an infinite estimate of pointwise density). Situations where values are drawn from a discrete grid (e.g., an integer lattice) will fail to have this crucial property. One remedy for such scenarios could be to add random offsets to the datapoints to form a continuous distribution.

Clusters with non-convex shapes may be well separated in density but not separated by a hyperplane (Figure XYZ). In these situations, alternative methods such as DBSCAN are preferable.

While each iteration is efficient (essentially linear in a subset of the number of points of interest), computation time may be a concern since the number of iterations required to converge is unknown. Empirically, total computation time appears to increase linearly with the number of clusters, the number of dimensions, and the sample size.

As mentioned, a principal advantage of Isosplit is that it does not require parameter adjustments. Indeed, the core computational step is isotonic regression, which does not rely on any tunable parameters. One parameters is fixed once and for all, the threshold of rejecting the unimodality hypothesis for the 1D tests. In Appendix XYZ we argue that the algorithm is not sensitive to these values over reasonable ranges.

## Conclusion

A multi-dimensional clustering technique, Isosplit, based on density clustering of one-dimensional projections was presented. The algorithm was motivated by the electrophysiological spike sorting application. Unlike many existing techniques, the new algorithm does not depend on adjustable parameters such as scale or *a priori* knowledge of the number of clusters. Using simulations, Isosplit was compared with standard clustering algorithms and was shown to outperform these methods in situations where clusters were separated by regions of relatively lower density and where each pair of clusters could be largely split by a hyperplane. Isosplit was especially effective for non-Gaussian cluster distributions, anisotropic clusters, and for cases of unequal cluster variances.

## <a name="appendix-a"></a>Appendix A: Up-down isotonic regression

In this section we present *up-down isotonic regression*, which finds a least-squares best function that monotonically increases to a critical point and then monotonically decreases. This is a core step in Isocut, which is the kernal operation of Isosplit.

Isotonic regression is a non-parametric method for fitting an ordered set of real numbers by a monotonically increasing (or decreasing) function. Suppose we want to find the best least-squares approximation of the sequence $x_1,\dots,x_n$ by a monotonically increasing sequence. Considering the more general problem that includes weights, we want to minimize the cost function

$$F_w(y) = \sum_{i=1}^n w_i(y_i-x_i)^2,$$

subject to

$$y_1 \leq y_2 \leq \dots \leq y_n.$$

This may be solved in linear time using the pool adjacent violators algorithm (PAVA) [@pava]; we do not include the full pseudocode for this standard algorithm but note that it is essentially the same as the `mava_mse` function in [Algorithm AA1](#algorithm-aa1).

For up-down isotonic regression we need to find a turning point $y_b$ such that $y_1\leq y_2\leq\dots\leq y_b$ and $y_b\geq y_{b+1}\dots\geq y_n$. Again we want to minimize $F_w(y)$. One way to solve this is to use an exhaustive search for $b\in\{1,\dots,n\}$ with two runs of isotonic regression at each step. However, this would have $O(n^2)$ time complexity.

A modified PAVA that finds the optimal $b$ for the up-down case in linear time is presented in [Algorithm AA1](#algorithm-aa1). The idea is to perform isotonic regression from left to right and then right to left using a modified algorithm where the mean-squared error is recorded at each step. The turning point is then chosen to minimize the sum of the two errors.

In addition to up-down, *down-up isotonic regression* is also needed by Isocut. This procedure is a straightforward modification of isotonic_updown in [Algorithm AA1](#algorithm-aa1) by negating both the input and output.

<figure>
<a name="algorithm-aa1"></a>

```python
y = isotonic_updown(x):
    # isotonic regression for increasing followed by decreasing
    n := len(x)
    b := find_opimal_b(x)
    y1 := isotonic_increasing([x[1], ..., x[b]])
    y2 := isotonic_decreasing([x[b], ..., x[n]])
    y := [y1[1], ..., y1[b], y2[1], ..., y2[n-b+1]]
    return y

b = find_optimal_b(x):
    # find where to switch direction
    x1 := x
    x2 := -reverse(x) # negative reverse ordering
    m1 := pava_mse(x1)
    m2 := reverse(pava_mse(x2))
    b := argmin(m1[b] + m2[b])
    return  b

m = pava_mse(x, w):
    # modified PAVA to return MSE at every index
    i := 1
    j := 1
    count[i] := 1
    wcount[i] := w[j]
    sum[i] := w[j] * x[j]
    sumsqr[i] := w[j] * x[j]^2
    m[j] := 0

    for j := 2, ..., n:
        i := i + 1
        count[i] := 1
        wcount[i] := w[j]
        sum[i] := w[j] * x[j]
        sumsqr[i] := w[j] * x[j]^2
        m[j] := m[j - 1]
        while:
            if i = 1:
                break # first block
            if sum[i - 1]/count[i-1] < sum[i]/count[i]:
                break # criteria for stop merging in PAVA

            # merge the blocks
            m_before := sumsqr[i-1] - sum[i-1]^2/count[i-1]
            m_before := m_before + sumsqr[i] - sum[i]^2/count[i]
            count[i-1] := count[i-1] + count[i]
            wcount[i-1] := wcount[i-1] + wcount[i]
            sum[i-1] := sum[i-1] + sum[i]
            sumsqr[i-1] := sumsqr[i-1] + sumsqr[i]
            m_after := sumsqr[i-1] - sum[i-1]^2/count[i-1]
            m[j] := m[j] + m_after - m_before
            i := i - 1
```

<figcaption>

Algorithm AA1. Up-down isotropic regression.

</figcaption>
</figure>

## <a name="appendix-p"></a>Appendix P: Packing Gaussian clusters for simulations

Our simulations required automatic generation of synthetic datasets with fixed numbers of clusters of varying densities, populations, spreads, anisotropies, and orientations. The most challenging programming task was to determine the random locations of the cluster centers. If clusters were spaced out too much then the clustering would be trivial. On the other hand, overlapping clusters cannot be expected to be successfully separated. Here we briefly describe a procedure for choosing the locations such that clusters are tightly packed with the constraint that the solid ellipsoids corresponding to Mahalanobis distance $z_0$ do not intersect. Thus $z_0$, the separation parameter, controls the tightness of packing.

In the packed clusters example, the clusters are positioned iteratively, one at a time. Each cluster is positioned at the origin and then moved out radially in small increments of a random direction until the non-intersection criteria is satisfied. Thus we only need to determine whether two clusters defined by $(\mu_1,\Sigma_1)$ and $(\mu_2,\Sigma_2)$ are spaced far enough apart. Here $\mu_j$ are the cluster centers and $\Sigma_j$ are the covariance matrices. The problem boils down to determining whether two arbitrary $p$-dimensional ellipsoids intersect. Surprisingly this is a nontrivial task, especially in higher dimensions, but an efficient iterative solution is known [@ellipsoid-distance].

## References

<!--bibliography-->

\[@tiganj\] Tiganj, Zoran, and Mamadou Mboup. 2011. “A Non-Parametric Method for
Automatic Neural Spike Clustering Based on the Non-Uniform Distribution
of the Data.” *Journal of Neural Engineering* 8 (6): 066014.

\[@vargas\] Vargas-Irwin, Carlos, and John P Donoghue. 2007. “Automated Spike
Sorting Using Density Grid Contour Clustering and Subtractive Waveform
Decomposition.” *Journal of Neuroscience Methods* 164 (1): 1–18.

\[@lloyd1982least\] Lloyd, Stuart. 1982. “Least Squares Quantization in Pcm.” *IEEE
Transactions on Information Theory* 28 (2): 129–37.

\[@em\] Dempster, Arthur P, Nan M Laird, and Donald B Rubin. 1977. “Maximum
Likelihood from Incomplete Data via the EM Algorithm.” *Journal of the
Royal Statistical Society. Series B (Methodological)*, 1–38.

\[@mixturemodels\] McLachlan, Geoffrey, and David Peel. 2000. *Finite Mixture Models*.
Wiley-Interscience.

\[@roberts1998bayesian\] Roberts, Stephen J, Dirk Husmeier, Iead Rezek, and William Penny. 1998.
“Bayesian Approaches to Gaussian Mixture Modeling.” *IEEE Transactions
on Pattern Analysis and Machine Intelligence* 20 (11): 1133–42.

\[@skewGMM\] Frühwirth-Schnatter, S, and S Pyne. 2010. “Bayesian Inference for Finite
Mixtures of Univariate and Multivariate Skew-Normal and Skew-*t*
Distributions.” *Biostat.* 11 (2): 317–36.

\[@skewGMM2\] Browne, Ryan P., and Paul D. McNicholas. 2015. “A Mixture of Generalized
Hyperbolic Distributions.” *Canad. J. Stat.* 43 (2): 176–98.

\[@murtagh2012algorithms\] Murtagh, Fionn, and Pedro Contreras. 2012. “Algorithms for Hierarchical
Clustering: An Overview.” *Wiley Interdisciplinary Reviews: Data Mining
and Knowledge Discovery* 2 (1): 86–97.

\[@dbscan\] Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. “A
Density-Based Algorithm for Discovering Clusters in Large Spatial
Databases with Noise.” In *Proceedings of 2nd International Conference
on Knowledge Discovery and Data Mining (Kdd-96)*, 226–31. AAAI Press.

\[@mean-shift\] Cheng, Yizong. 1995. “Mean Shift, Mode Seeking, and Clustering.”
*Pattern Analysis and Machine Intelligence, IEEE Transactions on* 17
(8): 790–99.

\[@rodriguez-clustering\] Rodriguez, Alex, and Alessandro Laio. 2014. “Clustering by Fast Search
and Find of Density Peaks.” *Science* 344 (6191): 1492–6.

\[@hartigan1985dip\] Hartigan, John A, and PM Hartigan. 1985. “The Dip Test of Unimodality.”
*The Annals of Statistics*, 70–84.

\[@hartigan1985algorithm\] Hartigan, PM. 1985. “Algorithm as 217: Computation of the Dip Statistic
to Test for Unimodality.” *Journal of the Royal Statistical Society.
Series C (Applied Statistics)* 34 (3): 320–25.

\[@barlow1972isotonic\] Barlow, Richard E, and Hugh D Brunk. 1972. “The Isotonic Regression
Problem and Its Dual.” *Journal of the American Statistical Association*
67 (337): 140–47.

\[@pava\] Robertson, Tim, F T Wright, and Richard L Dykstra. 1988. *Order
Restricted Statistical Inference*. Wiley New York.

\[@ellipsoid-distance\] Lin, Anhua, and Shih-Ping Han. 2002. “On the Distance Between Two
Ellipsoids.” *SIAM Journal on Optimization* 13 (1): 298–308.