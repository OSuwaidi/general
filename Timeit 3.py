import timeit

print("The time taken using list_comp ", timeit.timeit(number=10, setup='import cv2', stmt='img=cv2.imread("Cat.jpg"); images=[img.copy() for _ in range(1000)]; blurred = [cv2.GaussianBlur(images[i], (3,3), 0) for i in range(len(images))]'))

print("The time taken using map ", timeit.timeit(number=10, setup='import cv2; import functools', stmt='img=cv2.imread("Cat.jpg"); images=[img.copy() for _ in range(1000)]; blurrer = functools.partial(cv2.GaussianBlur, ksize=(3,3), sigmaX=0); blurs = list(map(blurrer, images))'))
