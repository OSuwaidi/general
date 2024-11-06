# بسم الله الرحمن الرحيم
"""
* OpenCV is NOT used to train the neural networks, you should do that with a framework like PyTorch, and then export the model to run in OpenCV.
* OpenCV IS used to take a trained neural network model, prepare and preprocess images for it, apply it on the images and output the results.
   You can also use it to combine neural networks with other computer vision algorithms available in OpenCV.

* OpenCV deep learning execution process:
- Load a model from disk.
- Pre-process images to serve as inputs to the neural network.
- (run other computer vision algorithms on the input images if necessary)
- Pass the image through the network and obtain output classifications.
- (run other computer vision algorithms on the outputs if necessary)

* OpenCV changes its behavior according to the type of the image array passed:
- If the image is 8-bit unsigned, it is displayed as is.
- If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the value range [0,255*256] is mapped to [0,255].
- If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255].
"""

import numpy as np
from matplotlib import image, pyplot as plt
import cv2

# Using plots:
cat = image.imread('Cat.jpg')
print(f"Cat dimensions = {cat.shape}")  # The 3 at the end represents the 3 channels of RGB (row, column, channel)  --> (width, length, depth)
print(f"Number of pixels in Cat image = {cat.size}")  # Includes the 3 channels for RGB
width, length, dimensions = cat.shape  # Tuple unpacking!
print(f"Height = {width} \nWidth = {length} \nDimensions/Depth = {dimensions} \n")

plt.imshow(cat)
plt.show()  # When you hover over the image, you get a list of vales [R, G, B], each representing the intensity of their respective colors

gray_cat = cv2.cvtColor(cat, cv2.COLOR_RGB2GRAY)  # Convert to gray scale
print(f"Gray scale cat dimensions = {gray_cat.shape} \n")  # If you want to get the array/matrix of the image, print the object (img) itself without ".shape"

plt.imshow(gray_cat, cmap='gray')
plt.show()


# Using OpenCV:
print(f"CV version: {cv2.__version__} \n")  # Check your installed OpenCV version
CV_cat = cv2.imread('Cat.jpg')  # Note: "cv2.imread() takes another argument (1 or 0 or -1) [1: colored image | 0: gray scale image | -1: image with alpha channel]  --> Alpha channel: makes the pixel transparent/opaque
cv2.imshow('Color cat', CV_cat)
cv2.waitKey(0)  # 0 makes it such that your image window wont close until a  key is pressed

CV_cat_gray = cv2.imread('Cat.jpg', 0)  # Number 0 makes it gray scale
cv2.imshow('Gray cat', CV_cat_gray)
cv2.waitKey(1500)  # 1500 milliseconds = 1.5 seconds

cv2.destroyAllWindows()  # Will automatically close all opened image windows (instead of manually exiting each one by one)


cat = cv2.imread("Cat.jpg")  # CV reads the image in BGR format instead of RGB!!!
print(f"cat shape = {cat.shape} \n\n")
cv2.imshow('BGR', cat)
plt.imshow(cat)  # "plt.imshow()" expects the image in RGB format
plt.show()

# To read the image as RGB using "plt.imshow()":
plt.imshow(cat[:, :, ::-1])
plt.show()

# To flip the image across the vertical (y) axis, across the columns:
plt.imshow(cat[:, ::-1])
plt.show()

# To flip the image across the horizontal (x) axis, across the rows:
plt.imshow(cat[::-1])
plt.show()


# To get the pixel value of an image at a certain location:
r = 300  # Row
c = 600  # Column
value = cat[r, c]
print(f"RGB pixel value = {value} \n")  # Returns 3 values, one for red, one for green, one for blue

value = gray_cat[r, c]
print(f"Gray pixel value = {value}")  # Returns one value representing the shade of gray. Thus, gray scale images take less memory and have less complexity

"""
# Giving functions/actions to different mouse clicks:

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Event: is when you double click your left mouse button
        cv2.circle(img, (x, y), 50, (255, 0, 0), -1)  # Action: is to draw a circle on "img" centered at (x, y), with radius = 50, and color (255, 0, 0) and thickness = -1 (None)
                                                                                                  # Where (x, y) is the location your mouse is pointing at


img = np.zeros((512, 512, 3), dtype=np.uint8)  # Create a completely black image (zeros) of dimensions 512 x 512
cv2.namedWindow('drawing')

# Call the method that activates the new mouse action:
cv2.setMouseCallback('drawing', draw_circle)  # "drawing" is the window of our image that we want to act upon, while the second parameter is the function to give action (to be applied)

while True:
    cv2.imshow('drawing', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
"""