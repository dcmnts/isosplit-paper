import numpy as np
import bluster as bl
import figurl as fig
from plot_accuracies import plot_accuracies_vs_separation
import kachery_cloud as kcl


def main():
    np.random.seed(0)

    sigma1 = 1
    sigma2 = 1 / 10
    cluster_size = 500
    separations = np.arange(0, 6.01, 0.25)
    num_trials = 3
    dbscan_eps = (5 * sigma1) / np.sqrt(cluster_size)
    dbscan_min_samples = 3

    X = kcl.load_json('sha1://7e444d71a124e44995663d4e1b77e76b9c352039?label=clustering-benchmark-artificial-datasets.json')

    nonunimodal_examples = [
        'complex9', 'smile3', 'spiral', 'diamond9', 'disk-4000n', 'donutcurves', 'dartboard1', 'shapes', 'cuboids', '3-spiral', 'jain'
    ]
    # print([a['name'] for a in X])

    study = bl.Study(name='clustering-benchmark-datasets')
    for example_name in nonunimodal_examples:
        print(example_name)
        x = [a for a in X if a['name'] == example_name][0]
        name = x['name']
        datapoints = np.array(x['datapoints'], dtype=np.float32)

        # IMPORTANT!!! WE ARE PROJECTING DOWN TO TWO DIMENSIONS!
        datapoints = datapoints[:, :2]

        labels = np.array(x['labels'], dtype=np.int32)
        num_clusters = len(np.unique(labels))

        algorithms = [
            bl.AgglomerativeClusteringAlgorithm(name='Agg*', n_clusters=num_clusters),
            # bl.DBSCANAlgorithm(eps=dbscan_eps, min_samples=dbscan_min_samples, name='DBSCAN*'),
            bl.GMMAlgorithm(n_components=num_clusters, name='GMM*'),
            bl.Isosplit6Algorithm(name='Isosplit'),
            bl.KMeansAlgorithm(n_clusters=num_clusters, name='K-means*'),
            bl.RodriguezLaioAlgorithm(name='RL*'),
            # spectral clustering taking a long time!
            # bl.SpectralClusteringAlgorithm(name='Spect*', n_clusters=num_clusters), # having some trouble with complex9
        ]
        # algorithms = [
        #     bl.KMeansAlgorithm(n_clusters=num_clusters, name='K-means*')
        # ]

        DS = study.add_dataset(bl.Dataset(
            samples=datapoints,
            labels=labels,
            name=name,
            parameters={}
        ))
        for alg in algorithms:
            if hasattr(alg, 'set_true_labels'):
                alg.set_true_labels(labels)
            print(f'Dataset {name}; Algorithm {alg.name};')
            labels2 = alg.run(datapoints)
            DS.add_clustering(bl.Clustering(
                name=alg.name,
                classname=alg.classname,
                parameters=alg.parameters,
                labels=labels2
            ))

    url = study.figurl()
    print(url)
    with open('results/clustering_benchmark_datasets.figurl', 'w') as f:
        f.write(url)
        
if __name__ == '__main__':
    main()