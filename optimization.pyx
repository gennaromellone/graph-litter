import time 
import cv2
import numpy as np

import cython

@cython.boundscheck(False)
cpdef unsigned int sigmoid_fast(int x):
    if x > 3:
        return 2
    elif x < -3:
        return 1
    else:
        return 0 

@cython.boundscheck(False)
cpdef list histogramCSLTP_fast(unsigned char [:,:] image, unsigned int hist_size):
    # set the variable extension types
    cdef int x, y, image_width, image_height

    # grab the image dimensions
    image_height = image.shape[0]
    image_width = image.shape[1]
    img_grey = image
    #img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zeroHorizontal = np.zeros(image_width + 2).reshape(1, image_width + 2)
    zeroVertical = np.zeros(image_height).reshape(image_height, 1)

    img_grey = np.concatenate((img_grey, zeroVertical), axis = 1)
    img_grey = np.concatenate((zeroVertical, img_grey), axis = 1)
    img_grey = np.concatenate((zeroHorizontal, img_grey), axis = 0)
    img_grey = np.concatenate((img_grey, zeroHorizontal), axis = 0)

    pattern_img = np.zeros((image_height + 1, image_width + 1))
    
    
    for x in range(1, image_height -2):
        for y in range(1, image_width -2):
            
            s1 = sigmoid_fast(img_grey[x-2, y-2] - img_grey[x+2, y+2])
            s3 = sigmoid_fast(img_grey[x-2, y+2] - img_grey[x+2, y-2])*3
    
            s = s1 + s3
        
            pattern_img[x, y] = s
    start = time.time()
    pattern_img = pattern_img[1:(image_height+1), 1:(image_width+1)].astype(int)
    
    histogram = np.histogram(pattern_img, bins = np.arange(hist_size +1))[0]
    histogram = histogram.reshape(1, -1)
    
    #print("Time elapsed:", time.time() - start)
    return histogram[0].tolist()

def histogramCSLTP(image, hist_size):

    def sigmoid(x):
        if x > 3:
            return 2
        elif x < -3:
            return 1
        else:
            return 0 
    
    
    image_height = image.shape[0]
    image_width = image.shape[1]

    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zeroHorizontal = np.zeros(image_width + 2).reshape(1, image_width + 2)
    zeroVertical = np.zeros(image_height).reshape(image_height, 1)

    img_grey = np.concatenate((img_grey, zeroVertical), axis = 1)
    img_grey = np.concatenate((zeroVertical, img_grey), axis = 1)
    img_grey = np.concatenate((zeroHorizontal, img_grey), axis = 0)
    img_grey = np.concatenate((img_grey, zeroHorizontal), axis = 0)

    pattern_img = np.zeros((image_height + 1, image_width + 1))
    
    
    for x in range(1, image_height -2):
        for y in range(1, image_width -2):
            
            s1 = sigmoid(img_grey[x-2, y-2] - img_grey[x+2, y+2])
            s3 = sigmoid(img_grey[x-2, y+2] - img_grey[x+2, y-2])*3
    
            s = s1 + s3
        
            pattern_img[x, y] = s
    start = time.time()
    pattern_img = pattern_img[1:(image_height+1), 1:(image_width+1)].astype(int)
    
    histogram = np.histogram(pattern_img, bins = np.arange(hist_size +1))[0]
    histogram = histogram.reshape(1, -1)
    
    #print("Time elapsed:", time.time() - start)

    return histogram[0].tolist()