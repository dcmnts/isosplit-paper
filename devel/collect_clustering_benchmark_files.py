# 1/6/23
# sha1://7e444d71a124e44995663d4e1b77e76b9c352039?label=clustering-benchmark-artificial-datasets.json

# preparation:
# git clone https://github.com/deric/clustering-benchmark

import os
import io
import numpy as np
from scipy.io import arff
import kachery_cloud as kcl

def main():
    dirname = 'clustering-benchmark/src/main/resources/datasets/artificial'
    # dirname = 'clustering-benchmark/src/main/resources/datasets/real-world'

    files = []
    for filename in os.listdir(dirname):
        # problematic examples
        if filename in ['cluto-t7-10k.arff', 'zelnik4.arff', 'insect.arff', 'cluto-t8-8k.arff', 'impossible.arff', 'dpc.arff', 'cure-t2-4k.arff', 'cluto-t4-8k.arff', 'banana.arff', 'dpb.arff', 'cluto-t5-8k.arff']:
            continue
        print(filename)
        if filename.endswith('.arff'):
            path = f'{dirname}/{filename}'
            with open(path, 'r') as f:
                arff_text = f.read()
            x = arff.loadarff(io.StringIO(arff_text))[0]
            ndims = len(x[0])
            labels = [_label_to_int(a[ndims - 1]) for a in x]
            if len(np.unique(labels)) <= 100:
                datapoints = [[_value_to_float(a[j]) for j in range(ndims)] for a in x]
                files.append({
                    'name': filename[:-len('.arff')],
                    'ndims': ndims,
                    'datapoints': datapoints,
                    'labels': labels,
                    'arff': arff_text
                })
            else:
                print(f'WARNING: skipping {filename} because number of unique labels is {len(np.unique(labels))} > 100')
    for f in files:
        print(f'{f["name"]}; ndims: {f["ndims"]}; n: {len(f["datapoints"])}; unique labels: {np.unique(f["labels"])}')
    
    uri = kcl.store_json(files, label='clustering-benchmark-artificial-datasets.json')
    print(uri)

def _value_to_float(x):
    return float(x)

def _label_to_int(x):
    if isinstance(x, np.float64):
        return int(x)
    elif x == b'noise' or x == 'noise':
        return -1000
    elif x == b'A':
        return 1
    elif x == b'B':
        return 2
    elif x == b'C':
        return 3
    elif x == b'Class 1':
        return 1
    elif x == b'Class 2':
        return 2
    elif x == b'Class 3':
        return 3
    else:
        return int(x)

if __name__ == '__main__':
    main()