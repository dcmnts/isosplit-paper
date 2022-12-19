import numpy as np
import bluster as bl
import matplotlib.pyplot
from plot_accuracies import plot_accuracies_vs_separation


def main():
    np.random.seed(0)

    separations = np.arange(2.5, 8.5, 0.5)

    sigma = 1
    anisotropy_factor = 10
    cluster_size = 500
    num_trials = 3
    dbscan_eps = (5 * sigma * 1.8) / np.sqrt(cluster_size)
    dbscan_min_samples = 3
    algorithms = [
        bl.AgglomerativeClusteringAlgorithm(name='Agg*', n_clusters=3),
        bl.DBSCANAlgorithm(eps=dbscan_eps, min_samples=dbscan_min_samples, name='DBSCAN*'),
        bl.GMMAlgorithm(n_components=3, name='GMM*', covariance_type='full'),
        bl.Isosplit5Algorithm(name='Isosplit'),
        bl.KMeansAlgorithm(n_clusters=3, name='K-means*'),
        # bl.MeanShiftAlgorithm(name='MeanShift'),
        bl.RodriguezLaioAlgorithm(name='RL*'),
        bl.SpectralClusteringAlgorithm(name='SC*', n_clusters=3)
    ]

    study = bl.Study(name='Anisotropic')
    for sep in separations:
        mu_1 = [0, 0]
        cov_1 = np.array([[1, 0], [0, anisotropy_factor]])
        mu_2 = [sep, 0]
        cov_2 = np.array([[1, 0], [0, anisotropy_factor]])
        mu_3 = [sep * 2, 0]
        cov_3 = np.array([[1, 0], [0, 1]])
        D = bl.CompositeDistribution([
            (bl.GaussianDistribution(mu=mu_1, sigma=cov_1), 1),
            (bl.GaussianDistribution(mu=mu_2, sigma=cov_2), 1),
            (bl.GaussianDistribution(mu=mu_3, sigma=cov_3), 1)
        ])
        for itrial in range(1, num_trials + 1):
            samples, labels = D.sample_with_labels(cluster_size * 3)
            DS = study.add_dataset(bl.Dataset(
                samples=samples,
                labels=labels,
                name=f'Separation {sep:.2f}; Trial {itrial};',
                parameters={'separation': sep, 'trial': itrial}
            ))
            for alg in algorithms:
                if hasattr(alg, 'set_true_labels'):
                    alg.set_true_labels(labels)
                print(f'Separation {sep}; Trial {itrial}; Algorithm {alg.name};')
                labels2 = alg.run(samples)
                DS.add_clustering(bl.Clustering(
                    name=alg.name,
                    classname=alg.classname,
                    parameters=alg.parameters,
                    labels=labels2
                ))

    plot_accuracies_vs_separation(
        study,
        'results/anisotropic.svg',
        title='Accuracy vs. separation for anisotropic simulation'
    )

    url = study.figurl()
    print(url)
    with open('results/anisotropic.figurl', 'w') as f:
        f.write(url)

if __name__ == '__main__':
    main()