# Isosplit: A Non-Parametric Method for Unimodal Clustering

## Abstract

Many clustering algorithms require the tuning of adjustable parameters for each application or dataset, making them unsuitable for automated procedures that involve clustering. Some techniques require an initial estimate of the number of clusters, while density-based techniques typically require a scale parameter. Other parametric methods, such as mixture modeling, make assumptions about the underlying cluster distributions. Here we introduce Isosplit, a non-parametric clustering method that does not require adjustable parameters nor parametric assumptions about the underlying cluster distributions. The only assumption is that clusters are unimodal and separated from one another by hyperplanes of relatively lower density. The technique uses a variant of Hartigan's dip statistic and isotonic regression as its kernel operation. Using simulations, we compared Isosplit with standard methods including k-means, density-based techniques, and Gaussian mixture methods, and found that Isosplit overcomes many of the limitations of these techniques. Our algorithm was developed to tackle the "spike sorting" problem in electrophysiology and is well-suited for low-dimensional datasets with many observations. Isosplit has been in use as the clustering component of the MountainSort spike sorting algorithm and its source code is freely available.

## Introduction

The purpose of unsupervised data clustering is to automatically partition a set of datapoints into clusters that reflect the underlying structure of the data. In this work, we focus on scenarios where datapoints lie in a low-dimensional space, there are many observations, and each cluster is characterized by a dense core area surrounded by regions of lower density. This situation is common in our motivating application of spike sorting of neural firing events in electrophysiology, where this type of structure has been observed experimentally [@tiganj, @vargas].

Most clustering algorithms have the limitation of requiring careful adjustment of parameters. For example, the choice of the number of clusters ($K$) for k-means [@kmeans] can be difficult, particularly for large datasets with many clusters, such as in the spike sorting application. The output of k-means also depends on the initialization of the algorithm, and it is often necessary to run it multiple times to find a globally optimal solution. K-means has other limitations as well, such as rigid assumptions about cluster populations and variances, and the tendency to artificially split larger clusters and merge smaller ones.

Gaussian mixture modeling (GMM) is a flexible clustering algorithm that is usually solved using expectation-maximization (EM) [@em]. Many variations exist, some of which are outlined in Chapter 11 of [@murphy]. Unlike k-means, GMM allows each cluster to be modeled using a multivariate normal distribution. Some GMM implementations require knowledge of the number of clusters beforehand (see Chapter 8 of [@mixturemodels], while others consider this as a free variable [@roberts1998bayesian] . The main disadvantage of GMM is that it assumes clusters are well modeled by Gaussian distributions, which may not always be the case. Additionally, like k-means, GMM can be difficult to optimize when the number of clusters is large. Recently, mixture models with skew non-Gaussian components have been developed [@skewGMM, @skewGMM2], but these models are more complex with additional free parameters and may be even more difficult to optimize.

Hierarchical clustering (see Chapter 15 of [@zaki-book]) does not demand that the number of clusters be specified beforehand. However, the output is a dendrogram rather than a partition, preventing it from being directly applicable to our motivating example. To obtain a partition from the dendrogram, a criteria for cutting the binary tree must be specified, similar to specifying $K$ in k-means. In addition, other parameters are typically required in agglomerative methods to determine which clusters are joined in each iteration. A further issue is that the time complexity of hierarchical clustering is at least $O(n^2)$, where $n$ is the number of observations (see Sec. 14.2.3 of [@zaki-book].

Density-based clustering techniques such as DBSCAN [@dbscan] are useful due to their lack of assumptions about data distributions, allowing them to effectively identify clusters with non-convex shapes. However, these techniques also require parameters. For example, DBSCAN requires two parameters to be adjusted depending on the properties of the dataset, including $\epsilon$, a scale parameter. The algorithm is particularly sensitive to $\epsilon$ in dimensions greater than two, and the algorithm is not well suited for more than four dimensions. Additionally, if clusters in the dataset have varying densities, no choice of the $\epsilon$ parameter will be able to work on the entire dataset. The mean-shift [@mean-shift] algorithm overcomes this limitation to an extent by automatically estimating the bandwidth, but it is very slow, and as we will see, suffers from some of the limitations of other algorithms.

Rodriguez-Laio (RL) clustering [@rodriguez-clustering], also known as density-peak clustering, is another density-based clustering technique that identifies representative datapoints based on their large distances to other points with greater density. This method has potential due to its ability to detect clusters without requiring a bandwidth parameter, making it suitable for datasets with varying cluster densities. However, the algorithm requires a threshold for determining which datapoints should be considered cluster representatives and is susceptible to oversplitting when clusters are highly anisotropic.

In this article, we present a new density-based, scale-independent clustering technique that is suitable for situations where clusters are expected to be unimodal and can be separated from one another by hyperplanes. A cluster is considered unimodal if it is derived from a distribution that has a single peak of maximum density when projected onto any line. Thus, our assumption is that when any two adjacent clusters are projected onto the normal of a dividing hyperplane, they will form a 1D bimodal distribution with a split-point of lower density at the point of hyperplane intersection. Loosely speaking, this is the case when clusters are well-spaced and have convex shapes.

In addition to being density-based, our technique has elements of agglomerative hierarchical clustering and involves the EM-style iterative approach of k-means. It uses a non-parametric procedure to separate 1D distributions based on a modified Hartigan's dip statistic [@hartigan1985dip, @other_hartigan1985dip] and isotonic regression, which don't require any adjustable parameters (with the exception of a statistical significance threshold). In particular, no scale parameter is needed for density estimation. Moreover, since the core step of each iteration is 1D clustering applied to projections of data subsets onto lines, it overcomes the curse of dimensionality (the tradeoff being that we cannot handle clusters of arbitrary shape).

We first describe an algorithm for splitting a 1D sample into unimodal clusters. This forms the basis of the $p$-dimensional clustering technique, Isosplit. We present simulations comparing Isosplit with standard clustering techniques. We also address computational efficiency and scaling properties. Finally, we summarize the results and discusses the limitations of the method. The appendices cover implementation details for the algorithms and the generation of synthetic datasets.

**TODO: SpectralClustering**: Requires n_clusters. Seems that SC can be slow, esp on non-Gaussian example.

## Clustering in one dimension

Any approach overcoming the above limitations must at least be able to do so in the simplest, 1D case. Here we present a non-parametric approach to 1D clustering utilizing a statistical test for unimodality and isotonic regression. This procedure will then be used as the basis for the more general situation ($p\geq2$) described in Section [{isosplit-algorithm}].
  
Clustering 1D data is special due to the fact that the input data can be sorted. The task then becomes finding the $K-1$ cut points (real numbers between adjacent datapoints) that determine the $K$ clusters. We assume that the clusters are unimodal, meaning that the density is lower between adjacent clusters. For simplicity we will describe an algorithm for deciding whether there is one cluster, or more than one cluster. In the latter case, a single cut point is determined representing the boundary separating one pair of adjacent clusters. Once the data have been split, the same algorithm may then be applied recursively on the left and right portions leading to further subdivisions, converging when no more splitting occurs. Thus, in addition to being the basis for higher dimensional clustering, the algorithm described here may also be used as a basis for general 1D clustering.

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
where the approximation $F_X$  of $S_X$ is determined by down-up isotonic regression as described in Appendix [{appendixUpdown}]. Roughly speaking, $F_X$ results from a least-squares approximation of the emperical density function by a function that is monotonically increasing to left of the cutpoint, and monotonically decreasing to the right (see Figure [{ISOCUT}]).

As mentioned above, Hartigan's dip test has a flaw when the number of points in one cluster (say on the far left) is much smaller than the total size $n$. This is due to the fact that the absolute size of the dip in the emperical distribution only depends on the relatively small amount of data near the interface between the two cluster, whereas the test for rejection becomes more rigorous with increasing $n$. To address this, we perform a series of dip tests of sizes $\lfloor n/2 \rfloor, \lfloor n/4 \rfloor, \lfloor n/8 \rfloor, \dots$. We compare two tests for each size, one starting from the left and one starting from the right. If the unimodality hypothesis is rejected in any one of these tests, the algorithm stops, the null hypothesis is rejected, and the cut point for that segment is returned. Otherwise, the unimodality hypothesis is accepted. A more detailed description of this procedure is provided in Appendix XX.

In the case where the null hypothesis is rejected, a cutpoint must be found. This is obtained using down-up isotonic regression on the density residual as shown in Figurl [{ISOCUT}]. Algorithmic details are provided in the appendix.

![isocut_demo](https://user-images.githubusercontent.com/3679296/208520823-2a378ae7-68c8-4ce6-b1b4-2b2ba4dea168.svg)
> Figure ISOCUT: Illustration of the Isocut algorithm for testing for unimodality in 1D and determining an optimal cutpoint. (Top) histogram of simulated bimodal distribution. (Middle) Sub-sampled densities with unimodal fit obtained from up-down isotonic regression. (Bottom) Residual densities with fit from down-up isotonic regression to determine the cutpoint (minimum).

## Clustering in more than one dimension using 1D projections

In this section we address the $p$-dimensional situation ($p\geq2$) and describe an iterative procedure, termed Isosplit, in which the 1D routine is repeated as a kernel operation. The decision boundaries are less restrictive than k-means which always splits space into Voronoi cells with respect to the centroids, as illustrated in [{fig:decision_boundaries}].

The proposed procedure is outlined in Algorithm [{alg:main_algorithm}]. The input is a collection of $n$ points in $\mathbb{R}^p$, and the output is the collection of corresponding labels (or cluster memberships). The approach is similar to agglomerative hierarchical methods in that we start with a large number of clusters (output of *InitializeLabels* and iteratively reduce the number of clusters until convergence. However, in addition to merging clusters, the algorithm may also redistribute datapoints between adjacent clusters. This is in contrast to agglomerative hierarchical methods. At each iteration, the two [{closest}] clusters (that have not yet been handled) are selected and all datapoints from the two sets are projected onto a line orthogonal to the proposed hyperplane of separation. The 1D split test from the previous section is applied (see above) and then the points are redistributed based on the optimal cut point, or if no statistically significant cut point is found, the clusters are merged. This procedure is repeated until all pairs of clusters have been handled.

<!-- Algorithm goes here -->

The best line of projection may be chosen in various ways. The simplest approach is to use the line connecting the centroids of the two clusters of interest. Although this choice may be sufficient in most situations, the optimal hyperplane of separation may not be orthogonal to this line. Instead, the approach we used in our implementation is to estimate the covariance matrix of the data in the two clusters (assuming Gaussian distributions with equal variances) and use this to whiten the data prior to using the above method. The function *GetProjectionDirection* in [{alg:main_algorithm}] returns a unit vector $V$ representing the direction of the optimal projection line, and the function *Project* simply returns the inner product of this vector with each datapoint.

Similarly, there are various approaches for choosing the closest pair of clusters at each iteration *FindClosestPair*. One way is to minimize the distance between the two cluster centroids. Note, however, that we don't want to repeat the same 1D kernel operation more than once. Therefore, the closest pair that has not yet been handled is chosen. In order to avoid excessive iterations we used a heuristic for determining whether a particular cluster pair (or something very close to it) had been previously attempted.

The function *InitializeLabels* creates an initial labeling (or partitioning) of the data. This may be implemented using the $k$-means algorithm with the number of initial clusters chosen to be much larger than the expected number of clusters in the dataset, the assumption being that the output should not be sensitive once $K_\text{initial}$ is large enough (see Appendix [{appendixSensitivity}]). For our tests we used the minimum of $20$ and four times the true number of clusters. Since datasets may always be constructed such that our choice of $K_\text{initial}$ is not large enough, we will seek to improve this initialization step in future work.

The critical step is *ComputeOptimalCutpoint*, which is the 1D clustering procedure described in the previous section, using a threshold of $\tau_n=\alpha/\sqrt{n}$.

![decision_boundaries](https://user-images.githubusercontent.com/3679296/207963490-a9195e1e-88a3-4028-a7ac-a022cb0946cc.png)
> TODO: update this figure and describe it

## Results

To highlight the scenarios where Isosplit overcomes the limitations of other methods, we evaluated the accuracy of the various algorithms using simulated datasets. We selected optimal parameters for the non-Isosplit algorithms based on the known simulation parameters (e.g., the number of clusters for k-means). Isosplit, on the other hand, does not require any user-defined parameters.

### Unequal variances

K-means clustering assumes equal variances for the clusters, which leads to incorrect decision boundaries when clusters have unequal variances. The error is most pronounced when the variance mismatch is large and when the clusters are overlapping.  Isosplit is less likely to suffer from this problem due to its use of a decision boundary at the point of lowest density between the clusters.

To illustrate this, we simulated two clusters drawn from spherical multivariate Gaussian distributions in 2D with varying separation distances between the clusters. In each case, the sizes of the two clusters matched, but the standard deviations differed by a factor of 10 ($\sigma_1=1$; $\sigma_2=\frac{1}{10}$). The results are shown in Figures UV1 and UV2.

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://d672ac8b1994b0f02f4e19370f1de3a12077ac70&label=Bluster:%20Unequal%20variances&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22Spect*%22],%22ds%22:13}
<!--
height: 700
-->

> Figure UV1: Performance of Isosplit compared with other algorithms for two clusters of unequal variance with varying separation distances. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

![unequal_variances](https://user-images.githubusercontent.com/3679296/208480235-4873e081-3234-4953-a22d-e924052202f4.svg)
> Figure UV2: Average accuracies for the various clustering algorithms as a function of separation distance in the unequal variances simulation. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters).

The results of the comparison show that GMM performs best, as expected since the clusters were drawn from Gaussian distributions and the number of components was known. In the non-Isosplit cases, optimal parameters were used (e.g., K=2 for k-means), whereas Isosplit does not require any parameters to be set. Generally, Isosplit performed better than the non-GMM methods when the clusters overlapped to a moderate extent. The decision boundary for k-means was incorrect due to the unequal variances between the two clusters, and DBSCAN had trouble due to the varying densities of the clusters, making it difficult to choose an ideal scale parameter.

**TODO: Talk about MeanShift, SC, AC**

### Anisotropic clusters

Another assumption of k-means is that clusters are spherical, or isotropic. Since it tries to minimize the sum of squared distances to the cluster center, k-means can favor splitting anisotropic clusters in a direction of elongation rather than separating distinct clusters. Since it does not try to minimize any such metric, Isosplit does not have this problem as it will split along directions where there is a density dip, regardless of anisotropic.

K-means clustering assumes clusters are spherical, or isotropic. As it minimizes the sum of squared distances to the cluster center, k-means my favor splitting an elongated cluster along the direction of elongation rather than separating distinct anisotropic clusters. Isosplit, on the other hand, does not have this problem as it is designed to split clusters along directions where there is a density dip, regardless of anisotropic shape.

To illustrate this, we simulated three clusters in 2D, one spherical, and two having an anisotropic factor of 8:1. As in the unequal variances example, the separation distances were varied. The results are shown in interactive Figure AC. For separation distances around 4.5, Isosplit was more accurate than the other algorithms. Both k-means and GMM favored splitting the anisotropic clusters along the direction of elongation. DBSCAN struggled due to the variation in density in this example.

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://d22b563253cb5804e4a8df2877e9f06fabe1a414&label=Bluster:%20Anisotropic&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22SC*%22],%22ds%22:12}
<!--
height: 700
-->
> Figure AC1: Performance of clustering algorithms for three clusters, one spherical and two anisotropic, with varying separation distances. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

![anisotropic](https://user-images.githubusercontent.com/3679296/208479578-c1766b29-74a7-45b0-9e2a-91b518fb1fd3.svg)
> Figure AC2

**TODO: Talk about MeanShift, SC, AC**

## Non-Gaussian clusters

Both k-means and GMM assume that clusters are Gaussian distributed. When a cluster comes from a skewed distribution, the representative points are pulled in the skewed direction which can result in incorrect decision boundaries. Isosplit does not make the Gaussian assumption, and works with both skewed and symmetric distributions, provided they are unimodal. Figurl NG demonstrates this for two simulated clusters, with the one on the right being skewed right.

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://1d0cc25804a2153f103040599feaaca91f3f2155&label=Bluster:%20Non-Gaussian&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22Spect*%22],%22ds%22:0}
<!--
height: 700
-->

> Figure NG: Performance of clustering algorithms for a pair of clusters, one of which is non-Gaussian and skewed right. Algorithms with an asterisk have optimal parameters set based on known properties of the datasets (e.g., number of clusters). Use the interactive controls to explore all simulations.

**TODO: Talk about MeanShift, SC, AC**

## Many clusters

https://figurl.org/f?v=gs://figurl/bluster-views-1&d=sha1://dd517bd8d60a3d677eb615931a4794834b82bdca&label=Bluster:%20Many%20clusters&s={%22algs%22:[%22Agg*%22,%22DBSCAN*%22,%22GMM*%22,%22Isosplit%22,%22K-means*%22,%22RL*%22,%22Spect*%22],%22ds%22:9}
<!--
height: 700
-->

![many_clusters](https://user-images.githubusercontent.com/3679296/208481087-a7de34c9-7f0a-45ec-aa54-b811f1168f6a.svg)

## More than two dimensions

## Non-unimodal examples

![example_dbscan](https://user-images.githubusercontent.com/3679296/207963843-c3ffe463-e90e-4a6b-8021-2f8571033269.png)
> TODO: create a simulation for this type of example

## References

Hartigan, J. A., & Hartigan, P. M. (1985). The Dip Test of Unimodality. The Annals of Statistics.

Hartigan, P. M. (1985). Computation of the Dip Statistic to Test for Unimodality. Journal of the Royal Statistical Society. Series C (Applied Statistics), 34(3), 320-325.