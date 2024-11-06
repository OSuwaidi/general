# بسم الله الرحمن الرحيم

import numpy as np
from scipy import spatial

real_dist = np.array([1, 1, 4, 4, 10, 10, 35, 35])/100
approx_dist = np.array([25, 25, 12.5, 12.5, 6.25, 6.25, 3.125, 3.125])/100


def kl_div(true, pred):
    test = 0
    target = 0
    for t, p in zip(true, pred):
        test -= t*np.log2(p)  # Cross-entropy
        target -= t*np.log2(t)  # Entropy
    return test - target  # Cross-entropy >= Entropy


def swap_cross_entropy(true, pred):
    test = 0
    target = 0
    for t, p in zip(true, pred):
        test -= p*np.log2(t)
        target -= t * np.log2(t)
    return test - target


def sym_cross_entropy(true, pred):
    res = true @ pred
    target = true @ true
    return np.linalg.norm(res - target)


def sym_info_entropy(true, pred):
    target = np.array([-t*np.log2(t) for t in true])
    test = np.array([-p*np.log2(p) for p in pred])
    return np.linalg.norm(target - test)


print(f'{kl_div(real_dist, approx_dist) = }')
print(f'{swap_cross_entropy(real_dist, approx_dist) = }')
print(f'{sym_cross_entropy(real_dist, approx_dist) = }')
print(f'{sym_info_entropy(real_dist, approx_dist) = }\n')

cos_sim = spatial.distance.cosine
n = np.linalg.norm


def d1(x, y):
    return cos_sim(x, y) + np.cos(np.pi * np.sqrt(2 * (n(x) ** 2 + x @ y))/(2*n(x+y)+0.0001))  # Where "x" is the true entropy and "y" is the cross_entropy (predicted)


def d2(x, y):
    return cos_sim(x, y) + np.log(n(x + y) / np.sqrt(2 * (n(x) ** 2 + x @ y)) + 0.0001)


v = np.array([-t * np.log2(t) for t in real_dist])  # Entropy
w = np.array([-t * np.log2(p) for t, p in zip(real_dist, approx_dist)])  # Cross Entropy

print(f'Distance: {n(v - w)}')
print(f'Cos dist: {cos_sim(v, w)}')
print(f'New dist 1: {d1(v, w)}')
print(f'New dist 2: {d2(v, w)}')
