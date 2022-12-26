from typing import List
import numpy as np
import bluster as bl
from generate_random_clusters.generate_random_clusters import generate_random_clusters, Cluster
from isosplit5_slow.isosplit5_slow import isosplit5_slow

def main():
    np.random.seed(0)

    sep = 1.8
    num_clusters = 4
    cluster_size = 500
    clusters: List[Cluster] = generate_random_clusters(
        ndims=2,
        num_clusters=num_clusters,
        zdist=sep,
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
    labels2, iterations = isosplit5_slow(
        samples.T,
        return_iterations=True
    )
    print(len(iterations))

    study = bl.Study(name='Isosplit demo')
    prev_labels = None
    for ii, it in enumerate(iterations):
        new_labels = it['labels']
        if prev_labels is None or (not np.all(np.array(prev_labels) == np.array(new_labels))):
            DS = study.add_dataset(bl.Dataset(
                samples=samples,
                labels=labels,
                name=f'Iteration {ii}',
                parameters={}
            ))
            DS.add_clustering(bl.Clustering(
                name='Isosplit',
                classname='Isosplit',
                parameters={},
                labels=new_labels
            ))
            study.add_dataset(DS)
            prev_labels = new_labels
    
    url = study.figurl()
    print(url)

if __name__ == '__main__':
    main()