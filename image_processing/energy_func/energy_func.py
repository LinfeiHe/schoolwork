'''
this is a demo about using energy function for image denoising
'''

import cv2
import random
import matplotlib.pyplot as plt


def salt_pepper_noise(image, probability=0.05):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    [width, height] = gray.shape
    for i in range(width):
        for j in range(height):
            if random.random() <= probability:
                p = round(random.random())
                gray[i, j] = p * 255
    return gray

'''
def laplacian(matrix):
    [width, height] = matrix.shape
    temp = np.zeros((width + 2, height + 2), np.float64)
    out = np.zeros((width + 2, height + 2), np.float64)
    temp[1:width+1, 1:height+1] = matrix
    for i in range(1, width):
        temp[i, 0] = temp[i, 1]
        temp[i, height + 1] = temp[i, height]
    for i in range(1, height):
        temp[0, i] = temp[1, i]
        temp[width + 1, i] = temp[width, i]
    for i in range(1, width):
        for j in range(1, height):
            out[i, j] = temp[i + 1, j] + temp[i - 1, j] + temp[i, j + 1] + temp[i, j - 1] - 4 * temp[i, j]
    return out[1:width+1, 1:height+1]
'''

# main
img = cv2.imread('test.tif')
img = salt_pepper_noise(img)

K = 50
alpha = 0.25
temp = img.astype('int16')
'''temp = img.astype('float64')'''

plt.ion()
for i in range(K):
    temp += (alpha * cv2.Laplacian(temp, cv2.CV_16S, ksize=1)).astype('int16')
    '''temp += alpha * laplacian(temp)'''
    plt.imshow(temp, 'gray', interpolation='bicubic')
    plt.pause(0.5)
    s = 'K=%s'%i
    plt.title(s)
    #time.sleep(0.5)
