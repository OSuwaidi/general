# بسم الله الرحمن الرحيم
# Note: Shuffling rate needs to *decrease* for larger list sizes

import random
import numpy as np
import math


def shuffler(lis, sr, array=False, exc=None):  # "sr" = shuffling rate
    if type(lis) != list:  # Make it compatible with shuffling (mxn) arrays
        arr = lis
        shape = arr.shape
        arr = arr.reshape(-1)
        lis = list(arr)
    lis = lis[:]  # Done, such that any changes applied on "lis" won't affect original input list "x"
    prob = ([1] * math.ceil(sr * 10)) + ([0] * math.ceil((1-sr) * 10))  # Probability space
    indices = list(range(len(lis)))

    if exc is not None:
        for ele in sorted(exc, reverse=True):  # Since deleting elements from a list changes its indices, we remove the higher indices first so that the lower indices wont get changed/affected
            del indices[ele]  # Remove excluded indices (faster than using "sets") {shorturl.at/byAP2}

    for _ in range(math.ceil(sr * len(lis))):
        if random.choice(prob) == 1:
            i = random.choice(indices)
            j = random.choice(indices)
            temp = lis[i]
            lis[i] = lis[j]
            lis[j] = temp

    if array is True:
        return np.array(lis).reshape(shape)
    return lis
