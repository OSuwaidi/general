# بسم الله الرحمن الرحيم
# We have "n" flowers that are described by 2 features (inputs); some length "l" and width "w", and depending on these 2 features we are to predict whether it's a blue or a red flower
# Since we have 2 outputs (blue or red), this implies that this is a binary classification problem (cannot be solved linearly), therefore we need to introduce nonlinearity (either add hidden layers or activation functions or both)
# We set red flowers = 1, and blue flowers = 0, and that is possible by applying the sigmoid activation function to our predicted output

import time
import numpy
import sympy
from sympy import symbols as syms, lambdify
from matplotlib import pyplot as plt

y_b = 0
y_r = 1
alpha = 1  # The learning rate is the MOST IMPORTANT hyperparamter to tune when training a neural network (notice that we increased initial alpha to 1 for it to converge faster at the beginning, but then toned it down a bit to 0.7, so that it wont cause the weights to bounce-off/oscillate back and forth around the minimum of the cost surface/curvature)

w1, w2, b = syms('w1, w2, b')


def sigmoid(output):
    return 1/(1+sympy.exp(-output))


# Data samples:
data = [[3, 1.5, 1], [3.5, 0.5, 1], [4, 1.5, 1], [5.5, 1, 1], [1, 1, 0], [2, 1, 0], [2, 0.5, 0], [3, 1, 0]]  # 8 elements in list "data", each element is a list itself: [length, width, color]
cost = 0
for i in range(len(data)):
    sample = data[i]
    cost += (sigmoid(w1*sample[0] + w2*sample[1] + b) - sample[2])**4  # Here we are taking the cost of ALL our samples, in practice you would divide the number of samples into "batches", and then pass each batch into the cost to process the gradient descent, then repeat for all the different batches till you complete the entire sample set (cover all the batches = 1 epoch)
print(f"cost = {cost} \n")

for i in range(len(data)):
    sample = data[i]
    color = "r"
    if sample[2] == 0:
        color = "b"
    plt.scatter(sample[0], sample[1], c=color)

plt.axis([0, 6, 0, 2])  # axis() takes the form: [xmin, xmax, ymin, ymax]
plt.title('Decision Space')
plt.xlabel('Length')
plt.ylabel('Width')
plt.grid()
plt.show()  # A scatter plot of the decision space

t_p = time.process_time()
t_c = time.perf_counter()
t = time.time()

dw1 = cost.diff(w1)
dw2 = cost.diff(w2)
db = cost.diff(b)

dw1 = lambdify((w1, w2, b), dw1)
dw2 = lambdify((w1, w2, b), dw2)
db = lambdify((w1, w2, b), db)
cost = lambdify((w1, w2, b), cost)

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()
costs = []
for i in range(1000):  # Notice: we decreased the number of epochs because we were able to increase our learning parameter instead
    w1 -= alpha * dw1(w1, w2, b)
    w2 -= alpha * dw2(w1, w2, b)
    b -= alpha * db(w1, w2, b)
    costs.append(cost(w1, w2, b))
    if db(w1, w2, b) < 0.1:  # This allows us to train our model/algorithm for less epochs, therefore saves time during training (allows us to have higher initial alpha values, then we can decrease it here)
        alpha = 0.6  # Decreased alpha by a tiny bit only when "db" started to diminish, to push our gradients further and minimize the cost; by having more impact/change on our weights (This number has to always be less than the initial learning rate) (If this alpha was to remain the same as the initial alpha (1) or even = 0.9, the cost function would blow up!)

print(f"w1 = {w1} \nw2 = {w2} \nb = {b}\n\ndw1 = {dw1(w1, w2, b)} \ndw2 = {dw2(w1, w2, b)} \ndb = {db(w1, w2, b)} \n")

print(f"Final cost value = {costs.pop()} \n")

plt.plot(costs, c="g")
plt.title("Gradient Descent")
plt.xlabel("Epochs")  # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
plt.ylabel("Cost")
plt.grid()

# Try to predict if this sample is a red or a blue flower
x_pred1 = [3, 1.5]  # This point is barely in the "red" zone, it's very close to the decision boundary (~1)
y_pred1 = w1*x_pred1[0] + w2*x_pred1[1] + b
print(f"Our prediction says for red = {sigmoid(y_pred1)}")

x_pred2 = [3, 1]  # This point is barely in the "blue" zone, it's very close to the decision boundary (~0)
y_pred2 = w1*x_pred2[0] + w2*x_pred2[1] + b
print(f"Our prediction says for blue = {sigmoid(y_pred2)} \n")

print(f"CPU Time = {time.process_time()-t_p:.4} seconds")  # The sum of the system and user CPU time of the current process (for special situations only)
print(f"Real Time = {time.time()-t:.4} seconds")  # Not a good measure at all (relatively speaking). It’s not reliable, because it’s adjustable
print(f"Performance Time = {time.perf_counter()-t_c:.4} seconds")  # BEST measure of performance when comparing between different versions of a code/algorithm (The higher the "Tick Rate" (lower Resolution), the more precise the value is). Tick Rate: refers to the number of ticks per second, thus if it ticks faster, it measures time elapsed more accurately!
plt.show()                                                         # It also measures time elapsed during sleep, and it's system-wide
