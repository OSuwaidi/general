# بسم الله الرحمن الرحيم
# Constrained optimization algorithm; finding either the minimum or maximum of a cost function under some set of constraints (usually doesn't find optimal (global) solution)

from AccShuffler import acc_shuffle
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


def best(current):
    sr = 1
    length = len(current)
    indices = range(length)
    curr_heu = heuristic(current)
    while sr > 0:
        N = 5 * round(10 * (1 - sr) + 1)
        states = [acc_shuffle(current, sr) for _ in range(N)]  # The "sr" and "N" have to be increased together, increasing one only has no significant effect!
        sr -= 0.1  # Can decrease decay rate if we want more populations (more exploration in search space)
        states_heu = list(map(heuristic, states))
        max_index = states_heu.index(max(states_heu))
        max_heu = states_heu.pop(max_index)  # Remove (pop) the maximum heuristic such that state_max2 wont get the same heuristic index
        state_max = states.pop(max_index)  # Need to also remove the max state from "states" such that the indexes match, because we removed (-1) an index from "states_heu"
        state_max2 = states[states_heu.index(max(states_heu))]
                                                                                                            # Cosine similarity?
        # Apply crossover between the two best results:  * Maybe try and shuffle crossovers *  (Ratio test: heu(2)/heu(1) should be >= 0.8 (sufficiently close))!!! --> Find where they don't match (False) and crossover the unmatched (False) values while preserving places where they match (True)
        cross_over1 = state_max[:(length//2)+1] + state_max2[(length//2)+1:]  # In a *non-constrained* problem, take *equal splits* of both states
        cross_over2 = state_max2[:(length//2)-1] + state_max[(length//2)-1:]
        cross_over3 = state_max[(length//2)-1:] + state_max2[:(length//2)-1]
        cross_over4 = state_max2[(length//2)+1:] + state_max[:(length//2)+1]
        crosses = [cross_over1, cross_over2, cross_over3, cross_over4]
        cross_heu = list(map(heuristic, crosses))
        max_cross_heu = max(cross_heu)
        if max_cross_heu > max_heu:
            max_heu = max_cross_heu
            state_max = crosses[cross_heu.index(max_heu)]

        # Apply mutation at a random index and invert the element/bit in that index:
        mutated = state_max[:]
        mutation_range = int(sr * length//3)  # More mutation when we are "far away" from optimal solution for exploratory purposes (explore new (initially far) areas in search space for potential expansion)
        if mutation_range < 1:
            mutation_range = 1
        for _ in range(mutation_range):
            i = random.choice(indices)
            if mutated[i] == 1:
                mutated[i] = 0
            else:
                mutated[i] = 1

        mut_heu = heuristic(mutated)
        if mut_heu > max_heu:
            max_heu = mut_heu
            state_max = mutated

        if max_heu > curr_heu:
            curr_heu = max_heu
            current = state_max

    return current, curr_heu


tp = time.perf_counter()
print(f"Optimal answer ≈ {best(x)} \nTime = {time.perf_counter()-tp:.4}")
