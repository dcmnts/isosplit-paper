from typing import List
import numpy as np
import bluster as bl
import figurl as fig
from generate_random_clusters.generate_random_clusters import generate_random_clusters, Cluster
from plot_accuracies import plot_accuracies_vs_parameter


def main():
    np.random.seed(0)

    num_clusters = 10
    cluster_size = 200
    separation = 3
    num_trials = 3
    ndimss = [2, 3, 4, 6, 10, 20]

    study = bl.Study(name='Higher dimensions')
    for ndims in ndimss:
        for itrial in range(1, num_trials + 1):
            clusters: List[Cluster] = generate_random_clusters(
                ndims=ndims,
                num_clusters=num_clusters,
                zdist=separation,
                pops=[cluster_size for i in range(num_clusters)],
                sigma_scale_factors=[1 + np.random.random() * 2 for i in range(num_clusters)],
                anisotropy_factor=2
            )
            D = bl.CompositeDistribution([
                (bl.GaussianDistribution(mu=c.mu.ravel(), sigma=c.covmat), c.pop)
                for c in clusters
            ])
            total_pop = np.sum([c.pop for c in clusters])

            samples, labels = D.sample_with_labels(total_pop)
            DS = study.add_dataset(bl.Dataset(
                samples=samples,
                labels=labels,
                name=f'Num. dims. {ndims}; Trial {itrial};',
                parameters={'ndims': ndims, 'trial': itrial}
            ))

            algorithms = [
                bl.AgglomerativeClusteringAlgorithm(name='Agg*', n_clusters=num_clusters),
                # bl.DBSCANAlgorithm(eps=dbscan_eps, min_samples=dbscan_min_samples, name='DBSCAN*'),
                bl.GMMAlgorithm(n_components=num_clusters, name='GMM*'),
                bl.Isosplit6Algorithm(name='Isosplit'),
                bl.KMeansAlgorithm(n_clusters=num_clusters, name='K-means*'),
                # bl.MeanShiftAlgorithm(name='MeanShift'),
                bl.RodriguezLaioAlgorithm(name='RL*', density_cutoff=None),
                # bl.SpectralClusteringAlgorithm(name='Spect*', n_clusters=num_clusters) # taking too long
            ]

            for alg in algorithms:
                if hasattr(alg, 'set_true_labels'):
                    alg.set_true_labels(labels)
                print(f'Num. dims. {ndims}; Trial {itrial}; Algorithm {alg.name};')
                labels2 = alg.run(samples)
                DS.add_clustering(bl.Clustering(
                    name=alg.name,
                    classname=alg.classname,
                    parameters=alg.parameters,
                    labels=labels2
                ))

    title='Accuracy vs. separation for the higher dimensions simulation'
    chart = plot_accuracies_vs_parameter(
        study,
        title=title,
        param_name='ndims',
        param_title='Num. dims.'
    )
    url = fig.Altair(chart).url(label=title)
    print(url)
    with open('results/higher_dimensions_plot.figurl', 'w') as f:
        f.write(url)

    url = study.figurl()
    print(url)
    with open('results/higher_dimensions.figurl', 'w') as f:
        f.write(url)

if __name__ == '__main__':
    main()