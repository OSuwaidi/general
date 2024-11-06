# بسم الله الرحمن الرحيم

from matplotlib import pyplot as plt
import sympy as sym
import numpy as np
import time


def sigmoid(z):
    return 1 / (1 + sym.exp(-z))


class Sample:
    w1, w2, b = sym.symbols('w1, w2, b')
    tot_error = 0

    def __init__(self, f1, f2, target):
        self.f1 = f1
        self.f2 = f2
        self.target = target

    def pred(self):
        return sigmoid(Sample.w1 * self.f1 + Sample.w2 * self.f2 + Sample.b)

    def error(self):
        error = (self.pred() - self.target) ** 2
        Sample.tot_error += error
        return error

    def deriv(self, var):
        return Sample.tot_error.diff(var)


data = [[3, 1.5, 1], [3.5, 0.5, 1], [4, 1.5, 1], [5.5, 1, 1], [1, 1, 0], [2, 1, 0], [2, 0.5, 0], [3, 1, 0]]
plottt = []
tp = time.perf_counter()


def evolution(data):
    for sample in data:
        s = Sample(*sample)
        s.error()
    tot_error = sym.lambdify(('w1', 'w2', 'b'), s.tot_error)
    dw1 = sym.lambdify(('w1', 'w2', 'b'), s.deriv('w1'))
    dw2 = sym.lambdify(('w1', 'w2', 'b'), s.deriv('w2'))
    db = sym.lambdify(('w1', 'w2', 'b'), s.deriv('wb'))

    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()
    i = 0.001
    f = 5
    n = 500
    while True:
        if dw1(w1, w2, b) < 0:
            w1 += np.random.uniform(i, f, n)
        else:
            w1 -= np.random.uniform(i, f, n)
        losses = tuple(tot_error(w1, w2, b))
        w1 = tuple(w1)[losses.index(min(losses))]

        if dw2(w1, w2, b) < 0:
            w2 += np.random.uniform(i, f, n)
        else:
            w2 -= np.random.uniform(i, f, n)
        losses = tuple(tot_error(w1, w2, b))
        w2 = tuple(w2)[losses.index(min(losses))]

        if db(w1, w2, b) < 0:
            b += np.random.uniform(i, f, n)
        else:
            b -= np.random.uniform(i, f, n)
        losses = tuple(tot_error(w1, w2, b))
        b = tuple(b)[losses.index(min(losses))]

        e = tot_error(w1, w2, b)
        plottt.append(e)
        if e < 0.1:
            return w1, w2, b


w1, w2, b = evolution(data)
plt.plot(plottt, c='m')
plt.grid()

x_p1 = [3, 1.25]  # This point is barely in the "red" zone, it's very close to the decision boundary (~1)
y_p1 = (w1 * x_p1[0] + w2 * x_p1[1] + b)
print(f"Our prediction for red says = {sigmoid(y_p1)}")

x_p2 = [3, 0.75]  # This point is barely in the "blue" zone, it's very close to the decision boundary (~0)
y_p2 = (w1 * x_p2[0] + w2 * x_p2[1] + b)
print(f"Our prediction for blue says = {sigmoid(y_p2)} \n")
print(f'Time = {time.perf_counter()-tp}')
plt.show()
