import numpy as np
import bluster as bl
import figurl as fig
from plot_accuracies import plot_accuracies_vs_parameter


def main():
    np.random.seed(0)

    sigma1 = 1
    sigma2 = 1
    cluster_sizes = np.arange(10, 201, 10)
    separation = 5
    num_trials = 5

    study = bl.Study(name='Small clusters')
    for cluster_size in cluster_sizes:
        D = bl.CompositeDistribution([
            (bl.GaussianDistribution(mu=[0, 0], sigma=np.eye(2) * sigma1 ** 2), 1),
            (bl.GaussianDistribution(mu=[separation, 0], sigma=np.eye(2) * sigma2 ** 2), 1)
        ])

        dbscan_eps = (5 * sigma1) / np.sqrt(cluster_size)
        dbscan_min_samples = 3
        algorithms = [
            # bl.AffinityPropagationAlgorithm(name='AP') AP did not converge, took a long time, and gave bad results
            bl.AgglomerativeClusteringAlgorithm(name='Agg*', n_clusters=2),
            bl.DBSCANAlgorithm(eps=dbscan_eps, min_samples=dbscan_min_samples, name='DBSCAN*'),
            bl.GMMAlgorithm(n_components=2, name='GMM*'),
            # bl.Isosplit5Algorithm(name='Isosplit5'),
            bl.Isosplit6Algorithm(name='Isosplit'),
            bl.KMeansAlgorithm(n_clusters=2, name='K-means*'),
            # bl.MeanShiftAlgorithm(name='MeanShift'),
            bl.RodriguezLaioAlgorithm(name='RL*'),
            bl.SpectralClusteringAlgorithm(name='Spect*', n_clusters=2),
        ]

        for itrial in range(1, num_trials + 1):
            samples, labels = D.sample_with_labels(cluster_size * 2)
            DS = study.add_dataset(bl.Dataset(
                samples=samples,
                labels=labels,
                name=f'Cluster size {cluster_size:.2f}; Trial {itrial};',
                parameters={'cluster_size': cluster_size, 'trial': itrial}
            ))
            for alg in algorithms:
                if hasattr(alg, 'set_true_labels'):
                    alg.set_true_labels(labels)
                print(f'Cluster size {cluster_size}; Trial {itrial}; Algorithm {alg.name};')
                labels2 = alg.run(samples)
                DS.add_clustering(bl.Clustering(
                    name=alg.name,
                    classname=alg.classname,
                    parameters=alg.parameters,
                    labels=labels2
                ))
    
    title='Accuracy vs. cluster size for small clusters simulation'
    chart = plot_accuracies_vs_parameter(
        study,
        title=title,
        param_name='cluster_size',
        param_title='Cluster size'
    )
    url = fig.Altair(chart).url(label=title)
    print(url)
    with open('results/small_clusters_plot.figurl', 'w') as f:
        f.write(url)

    url = study.figurl()
    print(url)
    with open('results/small_clusters.figurl', 'w') as f:
        f.write(url)
        
if __name__ == '__main__':
    main()