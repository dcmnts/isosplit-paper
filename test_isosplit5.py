import numpy as np
from isosplit5 import isosplit5
from generate_random_clusters import generate_random_clusters

def main():
    clusters, samples, labels = generate_random_clusters(
        ndims=2,
        num_clusters=5,
        zdist=15,
        pops=[100, 100, 100, 100, 100],
        sigma_scale=1,
        spread_factor=0,
        anisotropy_factor=0,
        nongaussian=False
    )
    labels2 = isosplit5(samples)
    print(np.max(labels2))
    
if __name__ == '__main__':
    main()