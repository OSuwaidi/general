# بسم الله الرحمن الرحيم
# Constrained optimization algorithm; finding either the minimum or maximum of a cost function under some set of constraints (usually doesn't find optimal (global) solution)

from AccShuffler import acc_shuffle
import numpy as np
import random
import time


def heuristic(lis):
    weights = [160, 2200, 350, 192, 333]
    tot_weight = 0
    for item, weight in zip(lis, weights):
        tot_weight += item*weight
    if (tot_weight/1000) <= 3:
        values = [150, 500, 60, 30, 40]
        tot_value = 0
        for item, value in zip(lis, values):
            tot_value += item*value
        return tot_value
    return 0


# Start with an equal distribution of 1's and 0's (if list is odd; have 1 be extra if maximizing, and have 0 be extra if minimizing)
x = [1, 0, 1, 0, 1]  # Can take values of 1's or 0's (exist or not)


def mutate(lis):
    i = random.choice(range(len(lis)))
    if lis[i] == 1:
        lis[i] = 0
    else:
        lis[i] = 1
    return lis


def best(current):
    sr = 1
    curr_heu = heuristic(current)
    while sr > 0:
        N = 5 * round(10 * (1 - sr) + 1)
        states = [acc_shuffle(current, sr) for _ in range(N)]  # The "sr" and "N" have to be increased together, increasing one only has no significant effect!
        sr -= 0.1  # Can decrease decay rate if we want more populations (more exploration in search space)
        states_heu = list(map(heuristic, states))
        max_index = states_heu.index(max(states_heu))
        max_heu = states_heu.pop(max_index)  # Remove (pop) the maximum heuristic such that state_max2 wont get the same heuristic index
        state_max = np.array(states.pop(max_index))  # Need to also remove the max state from "states" such that the indexes match, because we removed (-1) an index from "states_heu"
        state_max2 = states[states_heu.index(max(states_heu))]

        miss_match = np.where(state_max != state_max2)[0]  # Find where they're NOT equal (where they don't match)
        len_miss = len(miss_match)
        if len_miss > 1:
            rand = [random.choices([1, 0], k=len_miss) for _ in range(4)]
            crosses = [state_max.copy() for _ in range(4)]
            for i in range(len(crosses)):
                crosses[i][miss_match] = rand[i]
            cross_heu = list(map(heuristic, crosses))
            max_cross_heu = max(cross_heu)
            if max_cross_heu > max_heu:
                max_heu = max_cross_heu
                state_max = crosses[cross_heu.index(max_cross_heu)]
        state_max = list(state_max)

        # Apply mutation at a random index and invert the element/bit in that index:
        copies = [state_max[:] for _ in range(4)]
        mutations = list(map(mutate, copies))
        mut_heu = list(map(heuristic, mutations))
        max_mut_heu = max(mut_heu)
        if max_mut_heu > max_heu:
            max_heu = max_mut_heu
            state_max = mutations[mut_heu.index(max_mut_heu)]

        if max_heu > curr_heu:
            curr_heu = max_heu
            current = state_max

    return current, curr_heu


tp = time.perf_counter()
print(f"Optimal answer ≈ {best(x)} \nTime = {time.perf_counter()-tp:.4}")
