# بسم الله الرحمن الرحيم

import time
from sympy import symbols, lambdify
import numpy

t_c = time.perf_counter()
a = 0.01
w1, w2, b = symbols("w1, w2, b")
w = [w1, w2]

#  First input data
x1 = [5, 8]
y1 = 12
yp1 = b
for inputs, weights in zip(x1, w):
    yp1 += inputs * weights
print(f"yp1(w1, w2, b) = {yp1}")
c1 = (yp1 - y1) ** 2

#  Second input data
x2 = [2, 7]
y2 = 13
yp2 = b
for inputs, weights in zip(x2, w):
    yp2 += inputs * weights
print(f"yp2(w1, w2, b) = {yp2}")
c2 = (yp2 - y2) ** 2

# Cost function
cost = (1/2)*sum([c1, c2])
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
    w1 -= a * dw1(w1, w2, b)
    w2 -= a * dw2(w1, w2, b)
    b -= a * db(w1, w2, b)

print(f"w1 = {w1} \nw2 = {w2} \nb = {b}\n\ndw1 = {dw1(w1, w2, b)} \ndw1 = {dw2(w1, w2, b)} \ndb ={db(w1, w2, b)}\n")

print(f"y1_predicted = {w1*x1[0] + w2*x1[1] + b}\n")

print(f"Performance Time = {time.perf_counter()-t_c:.4} seconds")
