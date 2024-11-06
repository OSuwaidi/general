# بسم الله الرحمن الرحيم

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np

'''
To ***globally*** change the default properties of matplotlib plots, use: "plt.rcParams['property_name']"; a dictionary that contains plot properties
eg:
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
'''

plt.style.use('ggplot')
plt.rcParams["figure.autolayout"] = True

# Define datapoints for plotting:
x = np.linspace(0, 10, 101)
y = np.linspace(0, 10, 101)
x, y = np.meshgrid(x, y)


def fun(a, b, F=1, M=1):
    return M*np.sin(F*np.sqrt(a**2 + b**2))


def plotter(F=1, M=1):
    z = fun(x, y, F, M)
    sb.heatmap(z)
    plt.contour(z, extend='both', linewidths=3)
    plt.gca().invert_yaxis()  # "gca" means: "get current axes"
    plt.xticks(range(0, 101, 5), labels=range(0, 101, 5), rotation=0)  # Recall: "plt.xticks(ticks=number_of_ticks, labels=the_label_at_each_tick)"
    plt.yticks(range(0, 101, 5), labels=range(0, 101, 5), rotation=0)
    plt.show()


# Plot "z" using a heatmap from seaborn:
plotter()
'''
from ipywidgets import interactive
interactive_plot = interactive(plotter, F=(0, 10, 1), M=(1, 5, 1))  # "F(start, end, increment)"
interactive_plot  # Works for notebooks
'''

# Plot "z" using a 3D surface:
z = fun(x, y)
surface = plt.axes(projection='3d')
surface.plot_surface(x, y, z, cmap='magma')
plt.show()


# Draw animated plots:
fig = plt.figure()
graph = plt.plot([], marker='o')[0]  # The "[0]" is there because "plt.plot([], ...)" returns a *list* containing "Line3D" object
colors = iter([np.random.rand(4) for _ in range(6)])  # Will trigger once in every 21 elements

# Define datapoints:
x = np.linspace(-10, 10, 101)
y = x**2


def anim_fun(frame):
    if not frame % 21:  # "if not 0 ==> True"
        graph.set_color(next(colors))  # Generator object, such that it will get the next element on every "next" call automatically (no need for indexing)!
    graph.set_data(x[:frame], y[:frame])
    return graph


anim = FuncAnimation(fig, func=anim_fun, frames=101, interval=10, repeat=False, blit=False)  # Number of frames needs to be at least equal to the number of sample points to get a complete graph (if frames > num_samples: the graph will freeze at last frame for some time)

plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.show()


# Animating the "z" function from earlier:

# Define the datapoints again:
num_points = 50
x = np.linspace(0, 10, num_points)
y = np.linspace(0, 10, num_points)
x, y = np.meshgrid(x, y)
z = fun(x, y)

x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)  # Flatten the datapoints to apply list indexing for frame motion
dims = x.shape[0]  # Shape: (2500,)
length = num_points  # The length should match the number of points, such that you don't connect the last point in the meshgrid with the next starting point in the meshgrid (unwanted line)


# plt.style.use('seaborn') has "set_prop_cycle" containing only 6 colors (you can access the color cycle via: "plt.rcParams['axes.prop_cycle']"), and since we have 50 different lines, we want 50 different colors
fig = plt.figure()
graph = plt.axes(projection='3d')

# The "property cycle" controls the style properties such as: color, marker, and linestyle of future plot commands:
data = [[x[k:k+length], y[k:k+length], z[k:k+length]] for k in range(0, dims, length)]
graph.set_prop_cycle(color=[np.random.rand(4) for _ in range(length)], lw=[1.5] * length)  # Adjust "set_prop_cycle" to contain more colors (50)
graph = [plt.plot([], [], marker='.')[0] for _ in range(length)]  # Creates 50 "Line" objects (instances of matplotlib.3d class)


def anim_fun(frame):
    for sample, line in zip(data, graph):
        line.set_data_3d(sample[0][:frame], sample[1][:frame], sample[2][:frame])  # This will be applied to all 50 "line" objects we created
    return graph


anim = FuncAnimation(fig, func=anim_fun, frames=length+30, interval=10, blit=False)  # Number of frames needs to be at least equal to the number of sample points to get a complete graph (if frames > num_samples: the graph will freeze at last frame for some time: that's why we added "+30")

plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_zlabel('Z')
plt.gca().set_zlim(-1, 1)
plt.show()
