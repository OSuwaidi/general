import timeit

print("The time taken using comp ", timeit.timeit(setup='from test import heuristic; from AccShuffler import acc_shuffle', stmt='x= [1, 0, 1, 0, 1];x = [acc_shuffle(x, 1) for i in range(10)]; [heuristic(state) for state in x]'))
print("The time taken using map ", timeit.timeit(setup='from test import heuristic; from AccShuffler import acc_shuffle', stmt='x= [1, 0, 1, 0, 1];x = [acc_shuffle(x, 1) for i in range(10)]; list(map(heuristic, x))'))
