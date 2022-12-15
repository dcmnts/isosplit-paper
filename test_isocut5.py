import numpy as np
from isocut5_slow import isocut5

def main():
    a = 0 + np.random.randn(1000)
    b = 5 + np.random.randn(1000)
    c = 10 + np.random.randn(1000)
    samples = np.concatenate([a, b, c])
    d, c = isocut5(samples)
    print(d, c)

if __name__ == '__main__':
    main()