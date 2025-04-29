import sobel_demo as nd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time


def rgb_2_gray(img, mode='lut'):
    if mode == 'lut':
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722)
    else:
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)


img = io.imread("lena.jpg")
gray = rgb_2_gray(img).astype("float64")

# TODO: define filters in x in y direction
filter_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]], dtype=float)

filter_y = np.array([[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]], dtype=float)

def compute_gradient_magnitude(s1, s2):
    height, width = s1.shape

    res = np.zeros((height, width))

    for i in range(height):
        for j in range(width): 
            res[i,j] = np.sqrt(np.square(s1[i,j]) + np.square(s2[i,j]))

    return res

def flip_filter(filter_matrix):
    return np.flip(filter_matrix, axis=(0, 1))

fx = flip_filter(filter_x)
fy = flip_filter(filter_y)
start = time.time()
# TODO: filter image in x direction (nd.sobel(gray, filter_x))
t1 = nd.sobel(gray, fx)
end = time.time()
duration = end-start
plt.imshow(t1, cmap='gray')
plt.show()
print("Duration in milliseconds: ", duration*1000)

start = time.time()
# TODO: filter image in y direction (nd.sobel(gray, filter_y))
t2 = nd.sobel(gray, fy)
end = time.time()
duration = end-start
print("Duration in milliseconds: ", duration*1000)
plt.imshow(t2, cmap='gray')
plt.show()
# TODO compute Gradient magnitude
grad = compute_gradient_magnitude(t1, t2)

plt.imshow(grad, cmap='gray')
plt.show()