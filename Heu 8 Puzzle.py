# بسم الله الرحمن الرحيم
# Solving the puzzle using the "Shuffle Reduction" Algorithm

import numpy as np  # Note: numpy is way faster than torch
from AccShuffler import acc_shuffle


start = np.array([[5, 7, 2],
                  [3, 1, 6],
                  [4, 8, ' ']])  # Can start with even worst case initialization!

finish = np.array([[8, 1, 3],
                   [2, ' ', 4],
                   [7, 6, 5]])  # Has to be of type "numpy.ndarray" such that the heuristic can be computed element wise using the "np.sum()" operator!


def heuristic(state):
    return np.sum(state == finish)  # Returns a number, the higher, the more similar our "state" is to the goal (finish) state (adds the "Trues" as ones)


def shuff_reduc(current, finish):
    global steps
    steps = 0
    target_heu = heuristic(finish)
    sensitivity = 1  # How sensitive "shuffle rate" is to heuristic (the higher, the more you reward correct heuristic (shuffle), the less shuffles you do)
    while True:
        steps += 1
        curr_heu = heuristic(current)
        lock = np.where(current.reshape(-1) == finish.reshape(-1))[0]  # Finds where "current" is equal to "finish" (where it's "True") so that we won't move such places/locations
        sr = 1 - ((sensitivity * curr_heu) / target_heu)
        N = 100 * round(10 * (1 - sr) + 1)
        states = [acc_shuffle(current, sr, array=True, exc=lock) for _ in range(N)]
        states_heu = list(map(heuristic, states))
        max_heu = max(states_heu)
        state_max = states[states_heu.index(max_heu)]
        if max_heu == target_heu:
            return state_max
        elif max_heu > curr_heu:
            current = state_max


print(shuff_reduc(start, finish), f"Steps = {steps}")
