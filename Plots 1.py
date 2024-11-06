# بسم الله الرحمن الرحيم

import numpy
from matplotlib import pyplot as plt

i = [1, 2, 3]
data = [i] * 8  # A list of lists
print(f"Data = {data}")
print(f"Data length ={len(data)} \n")

random = numpy.random.rand()  # "rand()" gives a random float from 0 to 1 (only positive values)
print(f"Random = {random}")
gauss = numpy.random.randn()  # "randn()" gives a random float from the normal distribution "Gaussian" (can be negative values)
print(f"Gauss = {gauss}")
r = numpy.random.randint(0, 8)  # Gives a random integer from 0 to 7 (1/8 chance to get any integer)
print(f"Random integer = {r}")
print(f"Data[rand] = {data[r]} \n")  # Gives me the set/list in the parent list "data" by grabbing index "r" (a random list from "data" will be printed each time, depending on value of "r")

################################
x = []
y = []
z = []

for i in range(10):
    x.append(numpy.random.rand())
    y.append(numpy.random.rand())
    z.append(numpy.random.rand())

print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
plt.plot(x, label="x")  # When you plot a range/list only, without another axis, it will plot the lists values on the y-axis and have the number of values on the x-axis (Good for looking at changes per run/iteration)
plt.plot(y, label="y")
plt.plot(z, label="z")
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Random Generator Graph")
plt.legend()  # Adds a "legend" box that contains all the labels you assigned for each curve
plt.show()
