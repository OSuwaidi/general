# بسم الله الرحمن الرحيم
# We have "n" flowers that are described by 2 features (inputs); some length "l" and width "w", and depending on these 2 features we are to predict whether it's a blue or a red flower
# Since we have 2 outputs (blue or red), this implies that this is a binary classification problem (cannot be solved linearly [noncontinuous], therefore we need to introduce nonlinearity (either add hidden layers or activation functions or both))
# We set red flowers = 1, and blue flowers = 0, and that is possible by applying the sigmoid activation function to our predicted output
# 2 leaf/input nodes, and one output node (from 0 to 1)

import time
import numpy as np
import sympy
from sympy import symbols as syms, lambdify
from matplotlib import pyplot as plt

y_b = 0
y_r = 1
alpha = 0.1  # The learning rate is the MOST IMPORTANT hyperparamter to tune when training a neural network (notice that we increased initial alpha to 1.4 for it to converge faster at the beginning, but then toned it down a bit, so that it wont cause the weights to bounce-off/oscillate back and forth around the minimum of the cost surface/curvature)
              # The smaller the alpha, the less noisy your change in parameters and cost will be (more smooth/gradual)

w1, w2, b1, b2, c1, c2, k = syms('w1, w2, b1, b2, c1, c2, k')


def sigmoid(output):
    return 1/(1+sympy.exp(-output))


# Data samples:
data = [[3, 1.5, 1], [3.5, 0.5, 1], [4, 1.5, 1], [5.5, 1, 1], [1, 1, 0], [2, 1, 0], [2, 0.5, 0], [3, 1, 0]]  # 8 elements in the list "data", each element is a list itself: [length, width, color]
cost = 0

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
plt.grid()
plt.show()

t_p = time.process_time()
t_c = time.perf_counter()
t = time.time()

# Calculate the loss of this function:
for sample in data:
    cost += (sigmoid(w1*(sympy.sin(c1*sample[0] + b1)) + w2*(sympy.cos(c2*sample[1] + b2)) + k) - sample[2])**2  # Here we are taking the cost of ALL our samples, in practice you would divide the number of samples into "batches", and then pass each batch into the cost to process gradient descent, then repeat for all the different batches till you complete the entire sample set (cover all the batches = 1 epoch)
print(f"cost = {cost} \n")

dw1 = cost.diff(w1)
dw2 = cost.diff(w2)
db1 = cost.diff(b1)
db2 = cost.diff(b2)
dc1 = cost.diff(c1)
dc2 = cost.diff(c2)
dk = cost.diff(k)

dw1 = lambdify((w1, w2, b1, b2, c1, c2, k), dw1)
dw2 = lambdify((w1, w2, b1, b2, c1, c2, k), dw2)
db1 = lambdify((w1, w2, b1, b2, c1, c2, k), db1)
db2 = lambdify((w1, w2, b1, b2, c1, c2, k), db2)
dc1 = lambdify((w1, w2, b1, b2, c1, c2, k), dc1)
dc2 = lambdify((w1, w2, b1, b2, c1, c2, k), dc2)
dk = lambdify((w1, w2, b1, b2, c1, c2, k), dk)
cost = lambdify((w1, w2, b1, b2, c1, c2, k), cost)

w1 = np.random.randn()
w2 = np.random.randn()
b1 = np.random.randn()
b2 = np.random.randn()
c1 = np.random.randn()
c2 = np.random.randn()
k = np.random.randn()
costs = []
for i in range(1000):  # Notice: we decreased the number of epochs because we were able to increase our learning parameter instead
    w1 -= alpha * dw1(w1, w2, b1, b2, c1, c2, k)
    w2 -= alpha * dw2(w1, w2, b1, b2, c1, c2, k)
    b1 -= alpha * db1(w1, w2, b1, b2, c1, c2, k)
    b2 -= alpha * db2(w1, w2, b1, b2, c1, c2, k)
    c1 -= alpha * dc1(w1, w2, b1, b2, c1, c2, k)
    c2 -= alpha * dc2(w1, w2, b1, b2, c1, c2, k)
    k -= alpha * dk(w1, w2, b1, b2, c1, c2, k)
    costs.append(cost(w1, w2, b1, b2, c1, c2, k))
    if i == 100:  # Ladder technique
        alpha = 0.04  # This allows us to train our model/algorithm for less epochs, by starting out with a high lr, thus converging faster initially, then decreasing it to avoid exploding gradient. This saves time during training (allows us to have higher initial alpha values, then we can decrease it here)
    elif i == 500:
        alpha = 0.02  # Decreased alpha by a tiny bit when our gradients started to diminish, to push our gradients further and minimize the cost; by having more impact/change on our weights (This number has to always be less than the initial learning rate) (If this alpha was to remain the same as the initial alpha (1) or even = 0.9, the cost function would blow up!)


print(f"Final cost value = {costs[-1]} \n")

plt.plot(costs, c="g")
plt.title("Gradient Descent")
plt.xlabel("Epochs")  # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
plt.ylabel("Cost", rotation=0)
plt.grid()


# Try to predict if these never-seen before samples are red or blue flowers:
x_p1 = [3, 1.25]  # This point is barely in the "red" zone, it's very close to the decision boundary (~1)
y_p1 = w1*(np.sin(c1*x_p1[0] + b1)) + w2*(np.cos(c2*x_p1[1] + b2)) + k
print(f"Our prediction for red says = {sigmoid(y_p1)}")

x_p2 = [3, 0.75]  # This point is barely in the "blue" zone, it's very close to the decision boundary (~0)
y_p2 = w1*(np.sin(c1*x_p2[0] + b1)) + w2*(np.cos(c2*x_p2[1] + b2)) + k
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
z = npsigmoid(w1*(np.sin(c1*L + b1)) + w2*(np.cos(c2*W + b2)) + k)
space.plot_surface(L, W, z, cmap='cool')  # "summer" color map shows contrast between large-valued outputs and low-valued outputs. Whereas "hsv" color map shows the intensity in the rate of change of the function (highlights slopes)
for i in data:
    color = 'r' if i[2] == 1 else 'b'
    space.scatter3D(i[0], i[1], npsigmoid(w1*(np.sin(c1*i[0] + b1)) + w2*(np.cos(c2*i[1] + b2)) + k), s=100, c=color)  # "a.scatter3D(x, y, z, size=100, color='r')"


plt.title("3D Decision Space + Boundary")
plt.xlabel('Length')
plt.ylabel('Width')
space.set_zlim(0, 1)
space.set_zlabel('Prediction')
plt.show()
