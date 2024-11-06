# بسم الله الرحمن الرحيم

from matplotlib import pyplot as plt
import numpy as np

'''
To ***globally*** change the default properties of matplotlib plots, use: "plt.rcParams['property_name']"; a dictionary that contains plot properties
eg:
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
'''
plt.style.use('seaborn')


def sq(num):
    return num**2


# Bar graph/plot:
x = range(0, 10)
y = range(2, 22, 2)
plt.bar(x, y, color='black')
plt.show()


# Another bar graph/plot:
methods = ['AvgPool', 'FlexPool', 'FlexPool + Reg', 'FlexPool + Reg + DO']
datasets = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
y_avg = [91.38, 91.63, 63.29, 46.5]
y_flex = [92.35, 91.97, 63.55, 47.03]
y_flexreg = [93.48, 92.05, 63.83, 48.11]
y_flexregdo = [93.98, 92.32, 64.06, 48.55]

bar_width = 0.2

# Set position of bar on X axis:
r1 = range(len(datasets))  # Each bar will be separated by 1 unit
r2 = [x + bar_width for x in r1]  # Next bar will be "bar_width" units away from first bar
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Make the plot
colors = ['#7f6d5f', '#557f2d', '#2d7f5e', 'royalblue']

plt.bar(r1, y_avg, color=colors[0], width=bar_width, edgecolor='white', label=methods[0])  # Each bar will be separated by 1 unit
plt.bar(r2, y_flex, color=colors[1], width=bar_width, edgecolor='white', label=methods[1])
plt.bar(r3, y_flexreg, color=colors[2], width=bar_width, edgecolor='white', label=methods[2])
plt.bar(r4, y_flexregdo, color=colors[3], width=bar_width, edgecolor='white', label=methods[3])

# Add xticks in the middle of the group bars
plt.xticks([r + (bar_width*(len(methods)-1))/2 for r in range(len(datasets))], datasets, weight='bold')  # delta = (width*(n-1))/2

plt.ylim(44, 94)
plt.yticks(np.arange(44, 95, 2.5))
plt.legend(frameon=False)  # Removes frame/border from legend
plt.box(False)  # Removes background/grid from entire plot
plt.show()


# Another bar plot using ***object-oriented methods***:
plt.style.use('ggplot')

fig, ax = plt.subplots(nrows=1, ncols=1)
methods = ['Baseline', 'Proposed Method']
classes = ['crab', 'eel', 'fish', 'shells', 'starfish', 'animal_etc', 'plant', 'rov', 'fabric', 'fishing_gear', 'metal', 'paper', 'plastic', 'rubber', 'wood', 'trash_etc']
acc_baseline = [18, 37, 21, 13, 25, 16, 25, 59, 38, 12, 30, 34, 44, 22, 42, 13]
acc_proposed = [12, 42.3, 17.3, 27, 31.1, 28.003, 30, 69, 44.3, 26, 38.33, 35.03, 40, 27, 43, 21]

bar_width = 0.3

# Set position of bar on X axis:
r1 = range(len(classes))  # Each bar will be separated by 1 unit
r2 = [x + bar_width for x in r1]  # Next bar will be "bar_width" units away from first bar

# Make the plot
colors = [np.array([0.33238445, 0.40178976, 0.52039438, 0.50799573]), np.array([0.01965365, 0.47951259, 0.62485501, 0.78194779])]

ax.bar(r1, acc_baseline, color=colors[0], width=bar_width, edgecolor='white', label=methods[0])  # Each bar will be separated by 1 unit
ax.bar(r2, acc_proposed, color=colors[1], width=bar_width, edgecolor='white', label=methods[1])

# Add xticks in the middle of the group bars
plt.xticks([r + (bar_width*(len(methods)-1))/2 for r in range(len(classes))], classes, weight='bold', rotation=90)  # delta = (width*(n-1))/2

plt.ylim(10, 70)
plt.yticks(np.arange(10, 75, 5))
ax.legend(frameon=False)  # Removes frame/border from legend
ax.grid(color='#E6E6E6')
ax.set_facecolor('white')
fig.tight_layout()  # Makes everything more *fit*
fig.subplots_adjust(bottom=0.15)  # Add 15% margin on the bottom of the plot
plt.show()


# Pie chart/plot:
plt.style.use('seaborn')
data = [23, 17, 35, 29, 12, 41]  # Will be normalized such that they sum to 100 (or 1 in %)
cars = ['audi', 'toyota', 'tata', 'bmw', 'nissan', 'kia']
plt.pie(data, labels=cars)  # Matches corresponding elements from "data" to "cars" together
plt.show()

test = {'first': 25, 'second': 25, 'third': 50}
plt.pie(test.values(), labels=test.keys())  # Matches corresponding elements from "values" to "keys" together
plt.show()


# Line/Curve plots:
                            # Give me 100 values/elements, starting from 0 and ending at 5 (inclusive)
z = np.linspace(0, 5, 100)  # Since im starting at 0, I need 99 more elements to reach 5, hence: (5-0)/99 --> (b-a)/(n-1)
                            # The greater the number of intermediate values, the smoother the curve
w = sq(z)

plt.plot(z, w, ls=':', c="green", label='z graph')  # Will place a ':' on each (x, y) coordinate
plt.plot(x, y, ls='--', marker='d', c='pink', label='xy graph')  # Will place a 'diamond' on each plotted (x, y) based on equation
plt.axis([0, 10, 0, 25])  # "plt.axis([x-low, x-high, y-low, y-high])"
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.title("Graphs")
plt.show()


# sin graph:
plt.subplot(1, 2, 1)  # Number of rows = 1, number of columns = 2, subplot order = 1 (firsT)
x = np.linspace(0, 101, 10000)
y = np.sin(x)  # Note: Importing this "sin" function HAS TO BE FROM NUMPY, other "sin" functions don't take arrays as arguments
plt.plot(x, y, c='y')  # ***Every attribute below this "plt.plot()" will be assigned to itself, until a new "plt.plot()" is assigned***
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Sin Graph")


# cos graph
plt.subplot(122)  # Same thing as "plt.subplot(1, 2, 2)"
x = np.linspace(0, 101, 10000)
y = np.cos(x)
plt.plot(x, y, c='orange')  # New "plt.plot" assignment, thus define its attributes below
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Cos Graph")

plt.subplots_adjust(wspace=0.5)  # Width space between the 2 subplots
plt.tight_layout()
plt.show()
