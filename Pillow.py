# بسم الله الرحمن الرحيم

# import stepic as stepic  # Encoding/Decoding images
from PIL import Image, ImageDraw, ImageFont  # "Pillow" package
from PIL.ImageFilter import BLUR, SHARPEN, CONTOUR, SMOOTH
import numpy as np

# Making "cat" = object      # "bpp" = bits per pixel: number of bits needed to hold one unit/element (pixel) of an image
cat = Image.open('Cat.jpg')  # "Cat.jpg" has 24 bpp; that is: each pixel needs 3 bytes (1 byte = 8 bits) of data to be stored in that image
                             # Can only call image name without path, because image is in the same folder

dimensions = cat.size  # "img.size" returns a tuple of (length, width) (we put it in a list to be able to use tuple unpacking to unpack its contents)
print(f"Dimensions = {dimensions}")
length, width = dimensions
print(f"Resolution = {length * width}")

# Show normal image:
cat.show()

# Rotate the image:
cat_r = cat.rotate(180)  # "img.rotate()" is a generative method, thus it needs to be stored in a variable (does not affect the original image)


# Add text to the image:
draw = ImageDraw.Draw(cat, "RGBA")  # RGBA is a 4-channel format containing data for Red, Green, Blue, and Alpha. There is no use to the Alpha channel other than making the color at each pixel transparent/opaque (or partially transparent; translucent)
text = "Meow"
font_type = ImageFont.truetype("arial.ttf", 18)
draw.text((100, 100), text, (255, 255, 0), font=font_type)  # "draw.text(LOCATION, TEXT, COLOR, FONT_TYPE)"


# Blur the image:
cat_blur = cat.filter(BLUR)
cat_blur.show()

# Contour the image:
cat_cont = cat.filter(CONTOUR)


# Smoothen the image:
cat_smooth = cat.filter(SMOOTH)
cat_smooth.show()


# Sharpen the image (works by exaggerating the brightness difference along the edges within an image):
cat_sharp = cat.filter(SHARPEN)  # "Sharpening" an image gives more details/definition on the edges of that image
# The higher the resolution of the image, the more pixels it has, the sharper it can be (Because a patch of the image now can be defined with more # of pixels, thus giving better quality as each pixel can generate its own shade/intensity/opaqueness)
# The sharpening process works by utilizing a slightly blurred version of the original image; this is then subtracted away from the original to detect the presence of edges!
cat_sharp.show()
norm = np.array(cat)
blur = np.array(cat_blur)
sharp = norm - blur
sharp = Image.fromarray(sharp)
sharp.show()

blur = Image.fromarray(blur)
blur.show()


# To remove the outlines/borders around the image:
contour = np.array(cat_cont)
fade = norm - contour
fade = Image.fromarray(fade)



# Gray scale the image:
cat_gray = cat.convert('L')


# Convert to HSV (Hue, Saturation, Value=brightness)
cat_HSV = cat.convert('HSV')  # HSV is better than RGB in CV applications, because HSV separates "luma" (intensity) from "chroma" (color). Such that intensity is not coupled with color, whereas in RGB, a more intense image is a more red/green/blue image (0 to 255)
# These colors are better perceived by the human vision. H = Hue (color), S = Saturation (inverse of "whiteness"), V = Value/Intensity (brightness)
# Due to these properties, HSV is more robust against lighting changes and shadows (only intensity will be affected not the color)
    # eg: if you take 2 images of a single-color plane, one with shadow on it and one without. In RGB colorspace, the shadowed image will have very different characteristics than the normal one. However, in HSV colorspace, the "hue" (shade of color) of the 2 images will be likely the same, but they will differ in "value/luma" component (brightness)


# New image
ele = Image.open('Elephant.jpg')

new_size = (500, 500)  # (length, width)
ele = ele.resize(new_size)  # Resized elephant image


cat.paste(ele, (200, 50))  # Insert the elephant image into the cat image as watermark at location (200, 200) --> (l, w)


# Now representing images using arrays/matrices:
v = np.array([[1, 2, 3],
              [2, 3, 1]])
w = np.array([[2, 9, 0],
              [1, 3, 4]])
print(f"v = {v} \n")
print(f"w = {w} \n")


RGB = np.array(cat)
print(f"RGB size = {RGB.shape} \n")  # 549 rows, 976 columns, and 3 dimensions for RGB channels (the 3 represents depth)

x = Image.fromarray(RGB)  # Forms/constructs an image when given the array/matrix of that image


GS = np.array(cat_gray)
print(f"GS size = {GS.shape} \n")  # 549 rows, 976 columns, and 1 dimension only, where 0 represents black, and 255 represents white (in between = shades of gray)
print(GS, "\n\n")  # Prints the matrix of "cat" image

'''
# Hide messages inside images:
msg = stepic.encode(cat, b"Hello?")  # Embeds/encodes a "secret" message into an image in a way that doesn't affect how humans perceive the image, but actually changes the image slightly
msg.save("Encoded.png", 'PNG')

encoded = Image.open('Encoded.png')
dec = stepic.decode(encoded)  # Brings out the hidden message in the image
print(f"Hidden message = {dec} \n")

RGB_encoded = np.array(encoded)
print(f"cat - cat_hidden = {RGB - RGB_encoded}")  # This is a proof that the images actually differ slightly, because the difference in their arrays/matrices is not exactly 0!
secret_image = RGB - RGB_encoded
secret = Image.fromarray(secret_image)
secret.show()
'''
