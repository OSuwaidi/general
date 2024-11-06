import cv2
import numpy as np
from math import ceil, sin, cos, pi
from matplotlib import pyplot as plt

# Load the image in grayscale:
img = cv2.imread('circle.jpg', 0)
r, c = int(img.shape[0] / 2), int(img.shape[1] / 2)  # center row, center column
diameter = len([i for i in img[r, :] if np.all(i == 0)])  # Count black pixels in color image (all channels 0)
radius = ceil(diameter / 2)

# Convert grayscale image to BGR (3-channel) color image:
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Parameters for drawing lines:
spacing = pi/5  # Adjust for desired spacing
thetas = np.arange(0, 2 * pi, spacing)

# Draw red lines from the center of the circle:
for theta in thetas:
    end_x = round(c + radius * cos(theta))
    end_y = round(r + radius * sin(theta))
    cv2.line(img, (c, r), (end_x, end_y), color=(0, 255, 255), thickness=1)

# Display the image with matplotlib:
plt.scatter(c, r, c='y', s=50, marker='o')
plt.imshow(img)
# plt.axis('off')
plt.show()
