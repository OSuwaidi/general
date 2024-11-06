# بسم الله الرحمن الرحيم
# We have "n" flowers that are described by 2 features (inputs); some length "l" and width "w", and depending on these 2 features we are to predict whether it's a blue or a red flower
# Since we have 2 outputs (blue or red), this implies that this is a binary classification problem (cannot be solved linearly), therefore we need to introduce nonlinearity (either add hidden layers or activation functions or both)
# We set red flowers = 1, and blue flowers = 0, and that is possible by applying the sigmoid activation function to our predicted output
# 2 leaf/input nodes, and one output node (from 0 to 1)

import time
import numpy as np
import sympy
from sympy import symbols, lambdify
from matplotlib import pyplot as plt


y_b = 0
y_r = 1
             # The smaller the alpha, the less noisy your change in parameters and cost will be (more smooth/gradual)

w1, w2, b, wh = symbols('w1, w2, b, wh')  # Adding that single hidden layer "wh" greatly increases the amount of nonlinearity in our modelling function


def sigmoid(output):
    return 1/(1+sympy.exp(-output))


# Data samples:
data = [[3, 1.5, 1], [3.5, 0.5, 1], [4, 1.5, 1], [5.5, 1, 1], [1, 1, 0], [2, 1, 0], [2, 0.5, 0], [3, 1, 0]]  # 8 elements in the list "data", each element is a list itself: [length, width, color]
cost = 0

plt.style.use('seaborn')
# Construct a scatter plot of the decision space:
for sample in data:
    color = "r"
    if sample[2] == 0:
        color = "b"
    plt.scatter(sample[0], sample[1], c=color)

plt.axis([0, 6, 0, 2])  # axis() takes the form: [xmin, xmax, ymin, ymax]
plt.title('Decision Space')
plt.xlabel('Length')
plt.ylabel('Width')
plt.tight_layout()
plt.show()

t_p = time.process_time()
t_c = time.perf_counter()
t = time.time()

# Calculate the loss of this function:
for sample in data:
    cost += (sigmoid(wh*(w1*sample[0] + w2*sample[1] + b)) - sample[2])**2  # Here we are taking the cost of ALL our samples, in practice you would divide the number of samples into "batches", and then pass each batch into the cost to process gradient descent, then repeat for all the different batches till you complete the entire sample set (cover all the batches = 1 epoch)
print(f"cost = {cost} \n")

dw1 = cost.diff(w1)
dw2 = cost.diff(w2)
db = cost.diff(b)
dwh = cost.diff(wh)

dw1 = lambdify((w1, w2, b, wh), dw1)
dw2 = lambdify((w1, w2, b, wh), dw2)
db = lambdify((w1, w2, b, wh), db)
dwh = lambdify((w1, w2, b, wh), dwh)
cost = lambdify((w1, w2, b, wh), cost)


def tanh(output):
    return np.tanh(output)/10


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
wh = np.random.randn()
costs = []
for i in range(500):  # Notice: we decreased the number of epochs because we were able to increase our learning parameter instead
    w1 -= tanh(dw1(w1, w2, b, wh))
    w2 -= tanh(dw2(w1, w2, b, wh))
    b -= tanh(db(w1, w2, b, wh))
    wh -= tanh(dwh(w1, w2, b, wh))
    costs.append(cost(w1, w2, b, wh))

print(f"w1 = {w1} \nw2 = {w2} \nb = {b}\n\n∂c/∂w1 = {dw1(w1, w2, b, wh)} \n∂c/∂w2 = {dw2(w1, w2, b, wh)} \n∂c/∂b = {db(w1, w2, b, wh)} \n∂c/∂wh = {dwh(w1, w2, b, wh)} \n")

print(f"Final cost value = {costs[-1]} \n")

plt.plot(costs, c="g")
plt.title("Gradient Descent")
plt.xlabel("Epochs")  # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
plt.ylabel("Cost", rotation=0, labelpad=20)
plt.tight_layout()


# Try to predict if these never-seen before samples are red or blue flowers:
x_p1 = [3, 1.25]  # This point is barely in the "red" zone, it's very close to the decision boundary (~1)
y_p1 = wh*(w1*x_p1[0] + w2*x_p1[1] + b)
print(f"Our prediction for red says = {sigmoid(y_p1)}")

x_p2 = [3, 0.75]  # This point is barely in the "blue" zone, it's very close to the decision boundary (~0)
y_p2 = wh*(w1*x_p2[0] + w2*x_p2[1] + b)
print(f"Our prediction for blue says = {sigmoid(y_p2)} \n")

print(f"CPU Time = {time.process_time()-t_p:.4} seconds")  # The sum of the system and user CPU time of the current process (for special situations only)
print(f"Real Time = {time.time()-t:.4} seconds")  # Not a good measure at all (relatively speaking). It’s not reliable, because it’s adjustable
print(f"Performance Time = {time.perf_counter()-t_c:.4} seconds")  # BEST measure of performance when comparing between different versions of a code/algorithm (The higher the "Tick Rate" (lower Resolution), the more precise the value is). Tick Rate: refers to the number of ticks per second, thus if it ticks faster, it measures time elapsed more accurately!
                                                                   # It also measures time elapsed during sleep, and it's system-wide
plt.show()


def npsigmoid(output):
    return 1/(1+np.exp(-output))


# Draw Decision Boundary:
space = plt.axes(projection="3d")
length = np.linspace(0, 6, 30)
width = np.linspace(0, 2, 30)
L, W = np.meshgrid(length, width)  # Gives every possible combination of "length" and "width" values combined on a grid space
z = npsigmoid(wh*(w1*L + w2*W + b))
s = space.plot_surface(L, W, z, cmap='cool')  # "cool" color map shows contrast between large-valued outputs and low-valued outputs. Whereas "hsv" color map shows the intensity at the rates of change of the function (highlights slopes)
plt.colorbar(s, shrink=0.7, aspect=20)  # "aspect" is inverse of thickness
for i in data:
    color = 'r' if i[2] == 1 else 'b'
    space.scatter3D(i[0], i[1], npsigmoid(wh*(w1*i[0] + w2*i[1] + b)), s=100, c=color)  # "a.scatter3D(x, y, z, size=100, color='r')"

plt.title("3D Decision Space + Boundary")
plt.xlabel('Length')
plt.ylabel('Width')
plt.gca().set_zlim(0, 1)
plt.gca().set_zlabel('Prediction')
plt.show()
