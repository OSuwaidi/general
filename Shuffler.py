# بسم الله الرحمن الرحيم
# Note: Shuffling rate needs to *decrease* for larger list sizes

import math
import random

import numpy as np


def shuffler(lis, sr, array=False, exc=None):  # "sr" = shuffling rate
    if not isinstance(lis, list):  # Make it compatible with shuffling (mxn) arrays
        arr = lis
        shape = arr.shape
        arr = arr.reshape(-1)
        lis = list(arr)
    lis = lis[:]  # such that any changes applied on "lis" won't affect original input list "x"
    prob = [1] * math.ceil(sr * 10) + [0] * math.ceil((1 - sr) * 10)  # Probability space
    indices = list(range(len(lis)))

    if exc:
        # Since deleting left elements from a list changes right elements' indices (-1), remove right indices first so that left indices wont get changed/affected
        for ele in sorted(exc, reverse=True):
            # Remove excluded indices (faster than using "sets")
            del indices[ele]

    for _ in range(math.ceil(sr * len(lis))):
        if random.choice(prob) == 1:
            i = random.choice(indices)
            j = random.choice(indices)
            temp = lis[i]
            lis[i] = lis[j]
            lis[j] = temp

    if array:
        return np.array(lis).reshape(shape)
    return lis
