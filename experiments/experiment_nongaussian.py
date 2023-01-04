import numpy as np
import bluster as bl
import matplotlib.pyplot as plt


class NonGaussianDistribution(bl.Distribution):
    def __init__(self, *, mu: np.array, sigma: np.ndarray) -> None:
        super().__init__()
        self._mu = np.array(mu)
        self._sigma = sigma
        self._ndims = len(self._mu)
    @property
    def ndims(self) -> int:
        return self._ndims
    def sample(self, num_samples: int) -> np.ndarray:
        samples = np.random.multivariate_normal(self._mu, self._sigma, num_samples)
        samples[:, 0] = self._mu[0] + (samples[:, 0] - self._mu[0])**2
        return samples.astype(np.float32)

def main():
    np.random.seed(1) # random seed of 0 caused spectral clustering to hang

    cluster_size = 500
    num_trials = 8
    dbscan_eps = (9) / np.sqrt(cluster_size)
    dbscan_min_samples = 3
    algorithms = [
        bl.AgglomerativeClusteringAlgorithm(name='Agg*', n_clusters=2),
        bl.DBSCANAlgorithm(eps=dbscan_eps, min_samples=dbscan_min_samples, name='DBSCAN*'),
        bl.GMMAlgorithm(n_components=2, name='GMM*'),
        bl.Isosplit6Algorithm(name='Isosplit'),
        bl.KMeansAlgorithm(n_clusters=2, name='K-means*'),
        # bl.MeanShiftAlgorithm(name='MeanShift'),
        bl.RodriguezLaioAlgorithm(name='RL*'),
        bl.SpectralClusteringAlgorithm(name='Spect*', n_clusters=2)
    ]

    study = bl.Study(name='Non-Gaussian')
    for sep in [3]:
        D = bl.CompositeDistribution([
            (bl.GaussianDistribution(mu=[0, 0], sigma=[[1, 0], [0, 5]]), 1),
            (NonGaussianDistribution(mu=[sep, 0], sigma=[[3, 0], [0, 5]]), 1)
        ])
        for itrial in range(1, num_trials + 1):
            samples, labels = D.sample_with_labels(cluster_size * 2)
            DS = study.add_dataset(bl.Dataset(
                samples=samples,
                labels=labels,
                name=f'Trial {itrial};',
                parameters={'trial': itrial}
            ))
            for alg in algorithms:
                if hasattr(alg, 'set_true_labels'):
                    alg.set_true_labels(labels)
                print(f'Trial {itrial}; Algorithm {alg.name};')
                labels2 = alg.run(samples)
                DS.add_clustering(bl.Clustering(
                    name=alg.name,
                    classname=alg.classname,
                    parameters=alg.parameters,
                    labels=labels2
                ))

    url = study.figurl()
    print(url)
    with open('results/nongaussian.figurl', 'w') as f:
        f.write(url)

if __name__ == '__main__':
    main()