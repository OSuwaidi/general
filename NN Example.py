# بسم الله الرحمن الرحيم
# We have "n" flowers that are described by 2 features (inputs); some length "l" and width "w", and depending on these 2 features we are to predict whether it's a blue or a red flower
# Since we have 2 outputs (blue or red), this implies that this is a binary classification problem (cannot be solved linearly), therefore we need to introduce nonlinearity (either add hidden layers or activation functions or both)
# We set red flowers = 1, and blue flowers = 0, and that is possible by applying the sigmoid activation function to our predicted output

import numpy
import sympy
import time
from sympy import symbols as syms, lambdify
from matplotlib import pyplot as plt

t_p = time.process_time()
t_c = time.perf_counter()
t = time.time()

y_red = 1
y_blue = 0
alpha = 0.1
w1, w2, b = syms("w1, w2, b")
w = [w1, w2]

# Sets of inputs for different samples/data_point(s) [l, th] that correspond to red flowers:
r1 = [3, 1.5]  # Set of inputs 1
r2 = [3.5, 0.5]  # Set of inputs 2
r3 = [4, 1.5]  # Set of inputs 3
r4 = [5.5, 1]  # Set of inputs 4
red_inputs = [r1, r2, r3, r4]  # "red_inputs" represents a list that contains all the sets of inputs fed from different samples/data_point(s)

yr1 = yr2 = yr3 = yr4 = b
yr = [yr1, yr2, yr3, yr4]

i = 0
for r in red_inputs:
    for x, y in zip(r, w):
        yr[i] += x * y
    i += 1

# "yr" represents a list of the output data point(s), formed by each corresponding set of inputs from the different samples/data_point(s) BEFORE activation


def sigmoid(list):
    for i in range(0, len(list)):
        list[i] = 1/(1+sympy.exp(-list[i]))  # Note: you can't raise powers or take logs of expressions using normal methods: exp(), log(), rather you need to use special methods that are explicitly for symbols (expressions): sympy.exp(), sympy.log()


sigmoid(yr)

# Now "yr" represents the list of ACTIVATED outputs, which correspond to the set of inputs from multiple data points

cost_red = [(i - y_red)**2 for i in yr]  # Each predicted output from each sample point, is subtracted by the actual output for that data point (square diff)
cost_red = ((1/2)*sum(cost_red))  # Taking the average sum of errors (total amount of error divided by number of samples)
Cost_R = cost_red

# Sets of inputs for different samples/data_point(s) [l, th] that correspond to blue flowers:
b1 = [1, 1]
b2 = [2, 1]
b3 = [2, 0.5]
b4 = [3, 1]
blue_inputs = [b1, b2, b3, b4]

yb1 = yb2 = yb3 = yb4 = b
yb = [yb1, yb2, yb3, yb4]  # Set of outputs corresponding to each set of inputs

for i in range(0, len(blue_inputs)):  # Repeats for however many sample/data points we have for training
    for x, y in zip(blue_inputs[i], w):  # Looks at each index of list "blue_inputs", then calls each at a time to zip it with corresponding values of list "w"
        yb[i] += x*y  # Multiplies each element of list b1, b2, b3, b4 with corresponding/matching element in list "w"


sigmoid(yb)

cost_blue = [(i - y_blue)**2 for i in yb]
cost_blue = ((1/2)*sum(cost_blue))
Cost_B = cost_blue

# Now we optimize our parameters (w1, w2, b):
total_cost = Cost_R + Cost_B
print(f"Cost sum = {total_cost} \n")

dw1 = total_cost.diff(w1)
dw2 = total_cost.diff(w2)
db = total_cost.diff(b)

# Make our derivatives (gradients) "real" callable functions:
dw1 = lambdify((w1, w2, b), dw1)
dw2 = lambdify((w1, w2, b), dw2)
db = lambdify((w1, w2, b), db)
total_cost = lambdify((w1, w2, b), total_cost)

# Initialize/randomize your parameters:
w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()
costs = []
for i in range(10000):
    w1 -= alpha * dw1(w1, w2, b)
    w2 -= alpha * dw2(w1, w2, b)
    b -= alpha * db(w1, w2, b)
    costs.append(total_cost(w1, w2, b))

print(f"w1 = {w1} \nw2 = {w2} \nb = {b}\n\ndw1 = {dw1(w1, w2, b)} \ndw2 = {dw2(w1, w2, b)} \ndb = {db(w1, w2, b)} \n")
print(f"Final cost = {costs.pop()}")

plt.plot(costs, c="g")
plt.xlabel("Epochs")  # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
plt.ylabel("Cost")
plt.title("Gradient Descent")
plt.grid()


# Try to predict if this sample is a red or a blue flower:
def sigmoid1(output):
    return 1 / (1 + sympy.exp(-output))


x_pred1 = [3, 1.5]  # This point is barely in the "red" zone, it's very close to the decision boundary (~1)
y_pred1 = w1*x_pred1[0] + w2*x_pred1[1] + b
print(f"Our prediction says for red = {sigmoid1(y_pred1)} \n")

x_pred2 = [3, 1]  # This point is barely in the "blue" zone, it's very close to the decision boundary (~0)
y_pred2 = w1*x_pred2[0] + w2*x_pred2[1] + b
print(f"Our prediction says for blue = {sigmoid1(y_pred2)} \n")

print(f"CPU Time = {time.process_time()-t_p:.4} seconds")  # The sum of the system and user CPU time of the current process (for special situations only)
print(f"Real Time = {time.time()-t:.4} seconds")  # Not a good measure at all (relatively speaking). It’s not reliable, because it’s adjustable
print(f"Performance Time = {time.perf_counter()-t_c:.4} seconds")  # BEST measure of performance when comparing between different versions of a code/algorithm (The higher the "Tick Rate" (lower Resolution), the more precise the value is). Tick Rate: refers to the number of ticks per second, thus if it ticks faster, it measures time elapsed more accurately!
plt.show()
