# بسم الله الرحمن الرحيم

import numpy as np  # Note: numpy is way faster than torch
import time

tp = time.perf_counter()
rng = np.random.default_rng()

start = np.array([[2, 8, 3],
                  [1, 6, 4],
                  [7, ' ', 5]])

finish = np.array([[8, 1, 3],
                   [2, ' ', 4],
                   [7, 6, 5]])


def crack(current, target):
    global n
    n = 0
    upper = target[0]  # Create row vectors for us to match, since it is more probable to match fewer number of tiles, rather than the whole set of tiles (9 in this case)
    mid = target[1]  # Need to match each row vector
    lower = target[2]  # * Reducing length of vector to match, increases the probability of matching with that vector *
    u_lock = m_lock = l_lock = 0
    while True:
        n += 1
        shuff = [rng.permutation(current.reshape(-1)) for _ in range(50)]  # Transform our current state into a (1xn) row vector to shuffle all the elements within
        for shuffled in shuff:
            shuffled.resize(current.shape)

            if u_lock < 1:  # Lock is used so that if we get our desired row vector, we don't need to search for it again
                U = (upper == shuffled).all(1)  # ".all(1)" collapses the column index into 1: It compares column entries against column entries("upper" against the "shuffled" matrix/array in this case)
                if True in U:
                    u_lock = 1
                    state_upper = shuffled[np.where(True == U)]  # Capture where the elements are matching ("True")
                    current = shuffled[np.where(False == U)]  # Capture where the elements do NOT match ("False"), so that we shuffle fewer elements
                    break  # Break the "for" loop and use the new *dimensionally reduced* "current" for shuffling

            if m_lock < 1:
                M = (mid == shuffled).all(1)
                if True in M:
                    m_lock = 1
                    state_mid = shuffled[np.where(True == M)]
                    current = shuffled[np.where(False == M)]
                    break

            if l_lock < 1:
                L = (lower == shuffled).all(1)
                if True in L:
                    l_lock = 1
                    state_lower = shuffled[np.where(True == L)]
                    current = shuffled[np.where(False == L)]
                    break

        if u_lock == m_lock == l_lock == 1:  # If all the locks are *locked* (=1), then we found all the pieces to the desired state
            return np.array((state_upper, state_mid, state_lower))


print(crack(start, finish))
print(f"Time = {time.perf_counter() - tp:.4} \nSteps = {n}")
