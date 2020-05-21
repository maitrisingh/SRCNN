#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:24:39 2020

@author: code
"""

import sys
import keras
import cv2
import numpy
import matplotlib
import skimage

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Keras: {}'.format(keras.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('OpenCV: {}'.format(cv2.__version__))
print('Skimage: {}'.format(skimage.__version__))


#import necessary packages
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os

#define A function for peak signal to noise ration(PSNR)
def psnr(target, ref):
    #assume RGB/BGR image
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
     
    rmse = math.sqrt(np.mean(diff ** 2))
    
    return 20*math.log10(255. / rmse)


#define function for mean Squared error(MSE)
def mse(target, ref):
    #mse is the sum pf the squared difference between the two image
        
    err = np.sum((target.astype('float'))** 2)
    err /= float(target.shape[0] *target.shape[1])
        
    return err
    
#define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel = True))
    
    return scores

#prepare degraded images by introducing quality distortions via resizing
    
def prepare_images(path, factor):
    
    #loop throgh filesin the directory
    for file in os.listdir(path):
        
        #open the file
        img =  cv2.imread(path  +'/' + file)
        
        #find old and new image dimensions
        h, w, c = img.shape
        new_height = h / factor
        new_width = w / factor
        
        #resize the image -down
        img = (cv2.resize(img, (int(new_width), int(new_height)), interpolation = cv2.INTER_LINEAR))
        img = (cv2.resize(img, (int(w), int(h)), interpolation = cv2.INTER_LINEAR))
    
        #save the image
        print('Saving {}'.format(file))
        cv2.imwrite('images//{}'.format(file), img)
        
prepare_images('source_images/', 2)

#testing the generated images using image quality matrics

for file in os.listdir('images/'):
     
    #open target and reference images
    target = cv2.imread('images/{}'.format(file))
    ref = cv2.imread('source_images/{}'.format(file))
    
    #calculate the scores
    scores = compare_images(target, ref)
    
    #print all three scores
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    
#define the SRCNN model
    
def model():
    #define the model type
    SRCNN = Sequential()
    
    #add model layers
    SRCNN.add(Conv2D(filters = 128, kernel_size = (9,9), activation ='relu', padding = 'valid', use_bias = True, input_shape = (None, None, 1)))
    SRCNN.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu', padding = 'same', use_bias = True ))
    SRCNN.add(Conv2D(filters = 1, kernel_size = (5,5),  activation ='linear', padding = 'valid', use_bias = True))

    #define optimizer
    adam = Adam(learning_rate = 0.0003)
    #compile model
    SRCNN.compile(loss ='mean_squared_error', optimizer = adam, metrics =['mean_squared_error'])
    
    return SRCNN


#define necessary image processing functions
def modcrop(img, scale):
        
   tmpsz = img.shape
   sz= tmpsz[0:2]
   sz = sz - np.mod(sz, scale)
   img = img[0:sz[0], 1:sz[1]]
   return img


def shave(image, border):
   img = image[border: -border, border: -border]
   return img

#define main prediction function
def predict(image_path):
     
    #load the srcnn model with weights
    srcnn =model()
    srcnn.load_weights('3051crop_weight_200.h5')
    
    #load  the degraded and reference images
    path, file =os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('source_images/{}'.format(file))
    
    #preprocess the image with modcrop
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)
    
    #convert the image to YCrCb -srcnn trained on Y channel
    temp =cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    
    #create image slice and normalize
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype = float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float)/ 255
    
    #perform super resolution with srcnn
    pre = srcnn.predict(Y, batch_size = 1)
    
    #post process the output
    pre*= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    
    #copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    
    #remove border from reference  and degraded image
    ref = shave(ref.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)
    
    #image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref))
    scores.append(compare_images(output, ref))
    
    #return images and scores
    return ref, degraded, output, scores
    
    
     
ref, degraded, output, scores = predict('images/flowers.bmp')

#print all score for all images
print('Degraded Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[0][0], scores[0][1], scores[0][2]))
print('Reconstructed Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[1][0], scores[1][1], scores[1][2]))

#display images as subplots
fig, axs = plt.subplots(1, 3, figsize = (20, 8))
axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
axs[1].set_title('Degraded')
axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[2].set_title('SRCNN')


#remove the x and y tick marks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])