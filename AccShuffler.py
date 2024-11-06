# بسم الله الرحمن الرحيم

import random
import numpy as np


def acc_shuffle(lis, sr=1, array=False, exc=None, axis=None):  # "sr" = shuffling rate
    if axis == 1:  # To shuffle the number of batches (images) of an array --> Shuffles the first index of an array
        lis = list(lis)
    elif array is True:
        arr = lis
        shape = arr.shape
        lis = list(arr.reshape(-1))  # Will shuffle elements WITHIN the array

    lis = lis[:]  # Done, such that any changes applied on "lis" wont affect original input list "x"
    length = len(lis)
    indices = list(range(length))

    if exc is not None:
        for e in sorted(exc, reverse=True):  # Since deleting elements from a list changes its indices, we remove the higher indices first so that the lower indices wont get changed/affected
            del indices[e]  # Remove excluded indices (faster than using "sets")

    shuff_range = round(sr * length/2)  # How much to shuffle (depends on shuffling rate)
    if shuff_range < 1:
        shuff_range = 1  # "At least one shuffle (swap 2 elements)"

    for _ in range(shuff_range):
        i = random.choice(indices)
        indices.remove(i)
        j = random.choice(indices)
        indices.remove(j)
        lis[i], lis[j] = lis[j], lis[i]

    if axis == 1:
        return np.array(lis)
    if array is True:
        return np.array(lis).reshape(shape)
    return lis
