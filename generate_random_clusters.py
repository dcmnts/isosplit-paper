from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Cluster:
    sigma: np.ndarray
    mu: np.ndarray
    pop: int


def generate_random_clusters(*,
    ndims: int,
    num_clusters: int,
    zdist: float,
    pops: List[float],
    sigma_scale: float,
    spread_factor: float,
    anisotropy_factor: float,
    nongaussian: bool
):
    clusters: List[Cluster] = []
    for j in range(num_clusters):
        mu = -1 + np.random.random((ndims, 1)) * 2
        mu = mu / np.linalg.norm(mu)
        min_mu_radius = 0
        max_mu_radius = np.Inf
        mu_radius = 0
        sigma = random_sigma(n=ndims, spread_factor=spread_factor, anisotropy_factor=anisotropy_factor) * sigma_scale
        done = False
        while True:
            if is_far_enough_away(mu=mu * mu_radius, sigma=sigma, clusters=clusters, zdist=zdist):
                if mu_radius - min_mu_radius < 0.1:
                    break
                else:
                    max_mu_radius = mu_radius
                    mu_radius = (mu_radius + min_mu_radius) / 2
            else:
                min_mu_radius = mu_radius
                if mu_radius == 0:
                    mu_radius = 1
                else:
                    mu_radius = np.minimum(mu_radius * 2, (mu_radius + max_mu_radius) / 2)
        clusters.append(Cluster(sigma=sigma, mu=mu * mu_radius, pop=pops[j]))
    
    samples_list = []
    labels_list = []
    for j in range(len(clusters)):
        sigma = clusters[j].sigma
        mu = clusters[j].mu
        pop = clusters[j].pop
        samples0 = generate_samples(mu=mu, sigma=sigma, num=pop, nongaussian=nongaussian)
        samples_list.append(samples0)
        labels_list.append(np.ones((1, pop)) * (j + 1))
    
    samples = np.concatenate(samples_list, axis=1)
    labels = np.concatenate(labels_list, axis=1)

    return clusters, samples, labels

def generate_samples(*, mu: np.ndarray, sigma: np.ndarray, num: int, nongaussian: bool):
    n = len(mu)
    if not nongaussian:
        samples = np.random.randn(n, num)
    else:
        samples = np.random.randn(n, num)
        for dd in range(n):
            tmp = np.log(np.abs(np.random.randn(1, num) + 3))
            tmp = (tmp - np.mean(tmp)) / np.sqrt(np.var(tmp)) * 1.1
            samples[dd, :] = tmp
        R = create_random_rotation(n)
        samples = R @ samples
    # TODO: is this the same as cholcov in matlab????????????
    T = np.linalg.cholesky(np.linalg.inv(sigma))

    samples = np.linalg.inv(T) @ samples

    for j in range(num):
        samples[:, j] = samples[:, j] + mu.transpose()
    return samples


def random_sigma(*,
    n: int,
    spread_factor: float,
    anisotropy_factor: float
):
    A = np.zeros((n, n))
    spread = np.exp(-(np.random.random() * 2 - 1) * spread_factor)
    for j in range(n):
        A[j, j] = spread * np.exp(-(np.random.random() * 2 - 1) * anisotropy_factor)
    R = create_random_rotation(n)
    A = R @ A @ R.transpose()

    return A

def create_random_rotation(n: int):
    R = np.eye(n, n)
    for j in range(n - 1):
        theta = np.random.random() * 2 * np.pi
        RR = np.eye(n, n)
        RR[j, j] = np.cos(theta)
        RR[j, j + 1] = np.sin(theta)
        RR[j + 1, j] = -np.sin(theta)
        RR[j + 1, j + 1] = np.cos(theta)
        R = RR @ R
    return R

def is_far_enough_away(*, mu: np.ndarray, sigma: np.ndarray, clusters: List[Cluster], zdist: float):
    for j in range(len(clusters)):
        if check_gaussian_intersection(mu, sigma, clusters[j].mu, clusters[j].sigma, zdist):
            return False
    return True

def check_gaussian_intersection(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, rr: float):
    A1 = 2 * np.linalg.inv(sigma1)
    b1 = -2 * np.linalg.inv(sigma1) @ mu1
    alpha1 = (mu1.transpose() @ np.linalg.inv(sigma1) @ mu1 - rr * rr).item()

    A2 = 2 * np.linalg.inv(sigma2)
    b2 = -2 * np.linalg.inv(sigma2) @ mu2
    alpha2 = (mu2.transpose() @ np.linalg.inv(sigma2) @ mu2 - rr * rr).item()

    dist0 = ellipsoid_distance(A1=A1, b1=b1, alpha1=alpha1, A2=A2, b2=b2, alpha2=alpha2)
    if dist0 == 0:
        return True
    else:
        return False

def ellipsoid_distance(*, A1: np.ndarray, b1: np.ndarray, alpha1: float, A2: np.ndarray, b2: np.ndarray, alpha2: float):
    # % Computes the Euclidean distance between two ellipsoids defined by
    # %   1/2*x'*A1*x + b1'*x + alpha1 <= 0
    # %   1/2*x'*A2*x + b2'*x + alpha2 <= 0
    # %
    # %   A_j is a nxn matrix (positive definite)
    # %   b_j is a nx1 vector
    # %   alpha_j is a scalar
    # %
    # % Reference: Lin, Anhua, and Shih-Ping Han. "On the distance between
    # %            two ellipsoids." SIAM Journal on Optimization 13.1 (2002):
    # %            298-308.
    
    tol0 = 1e-5

    # initiation
    c1 = -np.linalg.inv(A1) @ b1
    c2 = -np.linalg.inv(A2) @ b2

    # For now we use the L_infinity norm for ease
    gamma1 = 0.5 / np.max(np.abs(A1))
    gamma2 = 0.5 / np.max(np.abs(A2))

    max_iterations = 50000

    for it in range(max_iterations):
        # Step 1
        for passnum in [1, 2]:
            if passnum == 1:
                cc = c1
                dd = c2 - c1
                AA = A1
                bb = b1
                aalpha = alpha1
                sgn = 1
            else:
                cc = c1
                dd = c2 - c1
                AA = A2
                bb = b2
                aalpha = alpha2
                sgn = -1
            qa = (1/2 * dd.transpose() @ AA @ dd).item()
            qb = (1/2 * cc.transpose() @ AA @ dd + 1/2 * dd.transpose() @ AA @ cc + bb.transpose() @ dd).item()
            qc = (1/2 * cc.transpose() @ AA @ cc + bb.transpose() @ cc + aalpha).item()
            discr = qb * qb - 4 * qa * qc
            if discr < 0:
                print('WARNING: Discriminant is less than zero in ellipsoid_distance')
                return 0 # is this right?
            if qa == 0:
                return 0 # Is this right? Not in paper. I think it means c1=c2, which i guess means they intersect
            if passnum == 1:
                t1 = (-qb + sgn * np.sqrt(discr)) / (2 * qa)
            else:
                t2 = (-qb + sgn * np.sqrt(discr)) / (2 * qa)
        
        # Step 2
        if t2 <= t1:
            return 0
        xbar = c1 + t1 * (c2 - c1)
        ybar = c1 + t2 * (c2 - c1)
        distance = (np.sqrt((xbar - ybar).transpose() @ (xbar - ybar))).item()
        if distance < tol0:
            return 0
        
        # Step 3
        theta1 = angle_between(ybar - xbar, A1 @ xbar + b1)
        theta2 = angle_between(xbar - ybar, A2 @ ybar + b2)
        if (np.abs(theta1) < tol0) and (np.abs(theta2) < tol0):
            return distance
        
        # Step 4
        c1 = xbar - gamma1 * (A1 @ xbar + b1)
        c2 = ybar - gamma2 * (A2 @ ybar + b2)
    
    print(f'distance={distance}, theta1={theta1}, theta2={theta2}',distance,theta1,theta2);
    raise Exception('Maximum iterations exceeded in ellipsoid_distance')

def angle_between(v1: np.ndarray, v2: np.ndarray):
    numer = (v1.transpose() @ v2).item()
    denom = np.sqrt((v1.transpose() @ v1) * (v2.transpose() @ v2)).item()
    if denom == 0:
        return 0
    return np.arccos(np.clip(numer / denom, -1, 1))