
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def randline_gen(length, dims=3):
    """
    Create a line using a random walk algorithm

    "length" is the number of points for the line.
    "dims" is the number of dimensions the line has.
    """
    lineData = np.zeros((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for i in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = (np.random.rand(dims) - 0.5) * 0.15
        lineData[:, i] = lineData[:, i-1] + step
    return lineData


def update_lines(frame, line_objects, line_data):
    for line, data in zip(line_objects, line_data):
        print(data[0:2, :frame])
        line.set_data(data[0:2, :frame])  # The x, y values in data must be as a *column* vector: "sample = [[x], [y]]"
        line.set_3d_properties(data[2, :frame])
    return line_objects


# Attaching 3D axis to the figure
fig = plt.figure()
graph = plt.axes(projection='3d')

# Generate 50 lines of random 3-D lines:
data = [randline_gen(25, 3) for index in range(50)]

# Creating 50 line objects:
lines = [graph.plot([], [])[0] for _ in data]

# Setting the axes properties:
plt.axis([0, 1, 0, 1])
plt.gca().set_zlim(0, 1)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_zlabel('z')
plt.title('Random Walk')

# Creating the Animation object
line_ani = FuncAnimation(fig, update_lines, frames=30, fargs=(lines, data), interval=50, blit=False)  # --> "update_lines(frame, line_objects, line_data)" and "fargs=(frames, lines, data)"
plt.show()
