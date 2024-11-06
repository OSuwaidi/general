# بسم الله الرحمن الرحيم
import timeit

print("The time taken to copy arrays:", timeit.timeit(number=100000, setup='import numpy as np', stmt='arr=np.array([1, 0, 1, 0, 1]); [arr.copy() for i in range(10)]'))
print("The time taken using numpy's exp:", timeit.timeit(setup='import numpy as np', stmt='np.exp(500)'))
print("The time taken using math's exp:", timeit.timeit(setup='from math import exp', stmt='exp(500)'))
