# بسم الله الرحمن الرحيم

import time
from sympy import symbols, lambdify
import numpy

t_c = time.perf_counter()
a = 0.01  # The smaller the learning rate, the less noisy your reduction in cost will be (will be more smooth/gradual), and the lower the chance that your weights will bounce off and/or be oscillating
w1, w2, b = symbols("w1, w2, b")  # learning rate is weighting the “contribution” of the latest batch of data, The lower the learning rate, the lower the importance of the latest batch.
# Decreasing the learning rate increases the chance of convergence but at the cost of more iterations (epochs) needed to reach the optimum , it's epochs (how many times you pass your training data into the NN) that results in overfitting
# The main result of overfitting: Having a model which is too complex (too many parameters) compared to the amount/quantity of training data used.


# First input data
x11 = 5
x21 = 8
y1 = 12
yp1 = (w1 * x11 + w2 * x21 + b)

# Second input data
x12 = 2
x22 = 7
y2 = 13
yp2 = (w1 * x12 + w2 * x22 + b)

# Cost function
cost = (1 / 2) * ((yp1 - y1) ** 2 + (yp2 - y2) ** 2)
print(f"cost(w1, w2, b) = {cost}\n")

dw1 = cost.diff(w1)
dw2 = cost.diff(w2)
db = cost.diff(b)

dw1 = lambdify((w1, w2, b), dw1)
dw2 = lambdify((w1, w2, b), dw2)
db = lambdify((w1, w2, b), db)

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()
for _ in range(0, 1000):
    w1 = w1 - a * dw1(w1, w2, b)  # This range should be large enough such that when you run the script multiple times, you get almost equal values for your parameters each time
    w2 = w2 - a * dw2(w1, w2, b)  # If increasing the range by a lot still doesn't have the parameters converge on a particular value, you can reduce the value of alpha (a)
    b = b - a * db(w1, w2, b)  # Note: learning rate (alpha) and number of iterations/epochs (range) are inversely proportional (smaller alpha requires much more iterations, and a larger alpha requires fewer iterations)

print(f"w1 = {w1} \nw2 = {w2} \nb = {b}\n\ndw1 = {dw1(w1, w2, b)} \ndw1 = {dw2(w1, w2, b)} \ndb ={db(w1, w2, b)}")
print(f"Performance Time = {time.perf_counter()-t_c:.4} seconds")
