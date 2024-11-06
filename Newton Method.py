# بسم الله الرحمن الرحيم

import sympy as sym
import time

tp = time.perf_counter()

x, y = sym.symbols('x, y')


def newtonn(var, function, guess):
    f_prime = function.diff()
    f = sym.lambdify(var, function)
    f_prime = sym.lambdify(var, f_prime)
    while abs(f(guess)) > 0.001:  # Recall that "while (anything)" and "if (anything)" will always be True, unless (anything) is 0, empty or False
        guess -= (f(guess) / f_prime(guess))
    return guess


f1 = sym.exp(x) - 4 * x  # Must use sympy's methods/functions!
print(f"Root at x = {newtonn(x, f1, 2):.4} \nTime = {time.perf_counter() - tp:.4}\n")  # Give an educated guess so that the algorithm converges quickly
# Note that you can get different solutions (roots) depending on your initial guess, if the function has more than one root


'''Expanding Newton's Method for "n" number of variables:'''


def newton(function, **kwargs):  # Note: "**kwargs" must come last, after ordinary arguments
    num_vars = len(kwargs)
    vars = tuple(kwargs.keys())
    guesses = list(kwargs.values())
    f_diff = [function.diff(v) for v in vars]

    f = sym.lambdify(vars, function)
    f_diff = [sym.lambdify(vars, f_partial) for f_partial in f_diff]
    while abs(f(*guesses)) > 0.001:  # *Note*: The "*" operator unpacks the tuple (or any iterable) and passes them as positional arguments into the function
        for i in range(num_vars):
            guesses[i] -= (f(*guesses) / f_diff[i](*guesses))
    return guesses


f1 = sym.exp(x) - 4 * x + y ** 2  # Must use sympy's methods/functions!
print(f"Root at x = {newton(f1, x=2, y=1)[0]:.4}, and y = {newton(f1, x=2, y=1)[1]:.4}")
