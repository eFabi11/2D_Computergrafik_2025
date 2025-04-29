import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time

def rgb_2_gray(img, mode='lut'):
    if mode == 'lut':
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722)
    else:
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)
    

def sobel(img, filter):
    # TODO: implement sobel filtering e.g. with 4 foor loops
    image_height, image_width = img.shape
    filter_size = filter.shape[0]
    out_height = (image_height - filter_size) // 1 + 1
    out_width = (image_width - filter_size) // 1 + 1
    filtered_img = np.zeros((out_height, out_width), dtype=np.float64)
    #filtered_img = np.zeros((gray.shape[0]-2, gray.shape[1]-2))
    for i in range(out_height):
        for j in range(out_width):
            region = img[i:i+filter_size, j:j+filter_size]
            filtered_img[i,j] = np.sum(region*filter) // 8
    
    return filtered_img

def compute_gradient_magnitude(s1, s2):
    height, width = s1.shape

    res = np.zeros((height, width))

    for i in range(height):
        for j in range(width): 
            res[i,j] = np.sqrt(np.square(s1[i,j]) + np.square(s2[i,j]))

    return res


def flip_filter(filter_matrix):
    return np.flip(filter_matrix, axis=(0, 1))

img = io.imread("lena.jpg")
gray = rgb_2_gray(img)

height, width = gray.shape

# TODO: define filter in x in y direction

filter_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]], dtype=float)

filter_y = np.array([[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]], dtype=float)

flipped_filter_x = flip_filter(filter_x)
flipped_filter_y = flip_filter(filter_y)

start = time.time()
# TODO: filter image in x direction (sobel(gray, filter_x))
sobel_x = sobel(gray, flipped_filter_x)
end = time.time()
duration = end-start
print("Duration in milliseconds: ", duration*1000)
plt.imshow(sobel_x, cmap='gray')
plt.show()
start = time.time()
# TODO: filter image in y direction (sobel(gray, filter_y))
sobel_y = sobel(gray, flipped_filter_y)
end = time.time()
duration = end-start
print("Duration in milliseconds: ", duration*1000)
plt.imshow(sobel_y, cmap='gray')
plt.show()

# TODO compute Gradient magnitude

grad = compute_gradient_magnitude(sobel_x, sobel_y)

plt.imshow(grad, cmap='gray')
plt.show()