import numpy as np
import bluster as bl
import matplotlib.pyplot as plt
from plot_accuracies import plot_accuracies_vs_separation


def main():
    np.random.seed(0)

    sigma1 = 1
    sigma2 = 1 / 10
    cluster_size = 500
    separations = np.arange(0, 6, 0.5)
    num_trials = 3
    dbscan_eps = (5 * sigma1) / np.sqrt(cluster_size)
    dbscan_min_samples = 3
    algorithms = [
        # bl.AffinityPropagationAlgorithm(name='AP') AP did not converge, took a long time, and gave bad results
        bl.AgglomerativeClusteringAlgorithm(name='Agg*', n_clusters=2),
        bl.DBSCANAlgorithm(eps=dbscan_eps, min_samples=dbscan_min_samples, name='DBSCAN*'),
        bl.GMMAlgorithm(n_components=2, name='GMM*'),
        bl.Isosplit5Algorithm(name='Isosplit'),
        bl.KMeansAlgorithm(n_clusters=2, name='K-means*'),
        # bl.MeanShiftAlgorithm(name='MeanShift'),
        bl.RodriguezLaioAlgorithm(name='RL*'),
        bl.SpectralClusteringAlgorithm(name='Spect*', n_clusters=2),
    ]

    study = bl.Study(name='Unequal variances')
    for sep in separations:
        D = bl.CompositeDistribution([
            (bl.GaussianDistribution(mu=[0, 0], sigma=np.eye(2) * sigma1 ** 2), 1),
            (bl.GaussianDistribution(mu=[sep, 0], sigma=np.eye(2) * sigma2 ** 2), 1)
        ])
        for itrial in range(1, num_trials + 1):
            samples, labels = D.sample_with_labels(cluster_size * 2)
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
        'results/unequal_variances.svg',
        title='Accuracy vs. separation for unequal variances simulation'
    )

    url = study.figurl()
    print(url)
    with open('results/unequal_variances.figurl', 'w') as f:
        f.write(url)
        
if __name__ == '__main__':
    main()