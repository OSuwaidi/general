# بسم الله الرحمن الرحيم

import numpy as np


def fun(x):
    return x * 2  # If we had a print() here, we wouldn't be able to print it again below in text (Can only assign its text here)


a = list(range(0, 10))
fun_a = fun(a)
print(f"a = {a} \na_type = {type(a)} \nfun(a) = {fun_a} \n")  # Repeated the list "a" twice!

# To fix:
for n in a:
    print(fun(n), end=" ")  # Notice however that number 10 was not printed (no 20 produced, because range ignores last unit)

print("\n")

# OR:
fun_map = map(fun, a)
print(list(fun_map))

print("\n")

b = np.linspace(0, 10, 11)  # Start from 0, go to 10, (11 units)---> 11 steps (to give integers +1 at a time)
fun_b = fun(b)
print(f"b = {b} \nb_type = {type(b)} \nfun(b) = {fun_b}")  # "np.linspace" does not give a list, rather gives an array of iterable numbers!
