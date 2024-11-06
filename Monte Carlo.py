# بسم الله الرحمن الرحيم
# Monte Carlo Simulation; is used to give the best estimate of a population, by using random sampling from that population, and it utilizes the Theory For Large Numbers
# The larger/greater the number of random samples (the bigger sample distribution) collected/gathered from the population, the more accurate the estimate of the population is (less variance).

import numpy as np
from matplotlib import pyplot as plt
import time
t_c = time.perf_counter()
t_p = time.process_time()

# Let's try to use Monte Carlo to estimate the area under a curve (integral):

# Recall that the average height of a function (average of a function) is equal to the integral of a function from a to b, divided by the interval between a and b --> (b-a)
#    f_avg = i=a to f=b: Σ(f_i) / (b-a)/Δx == Σ(f_i) / n:  Sum of (samples/points) / Number of (samples/points) *** n=(b-a)/Δx (number of samples), where "Δx" = spacing/interval (width of rectangles)***
#    f_avg = a-->b: ∫(f) dx / (b-a)  :  average_height = area/width  (Note: Δx --> dx as Δx --> 0)
#                   Cross multiply:
#    a-->b: ∫(f) dx = (b-a) * f_avg
#    a-->b: ∫(f) dx = (b-a) * (i=a to f=b: Σ(f_i)/n)  --> We used "n" here instead of "(b-a)/dx", because we can't practically take infinitely many samples (dx)
#                                                              --> Therefore we use a finite number of samples (Δx) --> n=(b-a)/Δx

# Now: Estimate the integral of the sine function from (limits of integration) 0 to pi:
#     0-->pi: ∫(sin(x)) dx = 2


def f(theta):
    return np.sin(theta)


a = 0
b = np.pi

N = 100  # Number of samples

areas_avg = []
for i in range(10000):  # Take 10000 different average areas produced by Monte Carlo estimation; plot them on a histogram to see the mean/average of area means
    # Important: "x" here is type "array", thus it can be iterated over!!!
    x = np.random.uniform(low=0, high=np.pi, size=N)  # Gives "N" number of random *floats* (samples) from 0 to pi
    tot_area = (b - a) * (sum(f(x)) / N)  # Note: "f(x)" here returns an array of outputs/results
    areas_avg.append(tot_area)

plt.hist(areas_avg, color='y', ec='black')  # "ec" = edge color between histogram rectangles
plt.title('Distribution of Areas using Monte Carlo')
plt.xlabel('Average Areas')


print(f"Time = {time.perf_counter()-t_c}")
print(f"Time = {time.process_time()-t_p}")

plt.show()  # "bins" = number of bars/buckets; where if the values produced were in between a specific range (x and y) they will all get stacked on the same bin
