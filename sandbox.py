# بسم الله الرحمن الرحيم و به نستعين

import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
import sympy as sym
from Bouncing_GD import bouncy_gd, gd

mem = Memory("./mycache")
plt.style.use('seaborn')
plt.rcParams['figure.autolayout'] = True

x, y = sym.symbols('x, y')
plot = True

fact1 = sym.sin(x)*sym.sin(y)
fact2 = sym.exp(sym.sqrt((100 - sym.sqrt(x**2+y**2)/sym.pi)**2))
f = -0.0001 * (sym.sqrt((fact1*fact2)**2)+1)**0.1

# *** Hard one ***:
# term1 = 100 * sym.sqrt(sym.sqrt((x - 0.01*y**2)**2))
# term2 = 0.01 * sym.sqrt((x+10)**2)
# f = term1 + term2

# frac1 = 1 + sym.cos(12*sym.sqrt(x**2+y**2))
# frac2 = 0.5*(x**2+y**2) + 2
# f = -frac1/frac2

# term1 = -(y+47) * sym.sin(sym.sqrt(sym.sqrt((y+x/2+47)**2)))
# term2 = -x * sym.sin(sym.sqrt(sym.sqrt((x-(y+47))**2)))
# f = term1 + term2

# fact1 = (sym.cos(sym.sin(sym.sqrt((x**2-y**2)**2))))**2 - 0.5
# fact2 = 1 + 0.001*(x**2+y**2)
# f = 0.5 + fact1/fact2

# fact1 = -sym.cos(x)*sym.cos(y)
# fact2 = sym.exp(-(x-sym.pi)**2-(y-sym.pi)**2)
# f = fact1*fact2

# term1 = sym.sin(x + y)
# term2 = (x - y)**2
# term3 = -1.5*x
# term4 = 2.5*y
# f = term1 + term2 + term3 + term4 + 1

# f = -20 * sym.exp(-0.2*sym.sqrt((x**2 + y**2)/2)) - sym.exp((sym.cos(2*sym.pi*x) + sym.cos(2*sym.pi*y))/2) + 20 + sym.exp(1)

grad_x = f.diff(x)
grad_y = f.diff(y)

f = sym.lambdify((x, y), f)
grad_x = sym.lambdify((x, y), grad_x)
grad_y = sym.lambdify((x, y), grad_y)


if plot:
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    X, Y = np.meshgrid(x, y)
    z = f(X, Y)

    space = plt.axes(projection='3d')
    space.plot_surface(X, Y, z, cmap='summer', antialiased=False)
    plt.xlabel('$w1$')
    plt.ylabel('$w2$')
    space.set_zlabel('$Loss$', rotation='horizontal')
    plt.show()


def loss(w):
    return f(*w)


def grad(w):
    return np.array([grad_x(*w), grad_y(*w)])


c1 = 0
c2 = 0
seed = np.arange(3000)
for i in range(100):
    LR = 1
    weight1, losses1 = bouncy_gd(2, loss, grad, lr=LR, epochs=100, TH=0.7, seed=seed[i])
    weight2, losses2 = bouncy_gd(2, loss, grad, lr=LR, epochs=100, TH=0.7, seed=seed[i], mom=0.999)
    if losses1[-1] < losses2[-1]:
        c1 += 1
    else:
        c2 += 1
    # plt.semilogy(losses1, label='GD')
    # plt.ylabel('Loss', rotation='horizontal')
    # plt.semilogy(losses2, label='Bouncy GD')
    # plt.xlabel('Iterations')
    # print(f'Bouncy GD Loss cond1: {losses1[-1]}')
    # print(f'Bouncy GD Loss cond2: {losses2[-1]}')
    # plt.legend()
    # plt.show()
print(c1)
print(c2)
