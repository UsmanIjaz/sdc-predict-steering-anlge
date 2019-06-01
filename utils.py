import os
import pandas as pd
import numpy as np
from scipy import signal
import cv2
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from pathlib import Path
import json

path = 'data/'
pr_threshold= 1

#data augmentation techniques are inspired by Vivek's blogpost
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.f6k9iuv50

#changing the brightness
#output: image with changed brightness
def change_brightness(image,brightness_range):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = brightness_range+np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2]*random_bright
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb

# rotating the image along x and y axis and calculating the new angle
# output: rotated image and new angle
def image_translation(image, angle, x_translation_range, y_translation_range, angle_translation_range):
    x_translation = (x_translation_range * np.random.uniform()) - (x_translation_range / 2)
    new_angle = angle + ((x_translation / x_translation_range) * 2) * angle_translation_range
    y_translation = (y_translation_range * np.random.uniform()) - (y_translation_range / 2)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0])),new_angle


#focusing on the road part of the image only
#returns the cropped image
def image_preprocess(image,rows,cols):
    roi = image[60:140, :, :]
    new_image = cv2.resize(roi,(cols,rows),interpolation=cv2.INTER_AREA)
    
    return new_image

#Generating new image and corresponding angle
def augment_train_image(line_data):
    
    shift_angle = {}
    shift_angle['left'] = 0.25
    shift_angle['center'] = 0.0
    shift_angle['right'] = -0.25
    
    #randomly selecting the image to be augmented
    image_selection = ['left','center','right']
    choice = random.choice(image_selection)
    path_file = line_data[choice][0].strip()
    angle = line_data['steering'][0] + shift_angle[choice]
    
    image = plt.imread(path+path_file)
    
    #image translation
    x_translation_range, y_translation_range, angle_translation_range = 150,10,0.2
    image,angle = image_translation(image, angle, x_translation_range, y_translation_range, angle_translation_range)
    
    #changing brightness
    image = change_brightness(image,brightness_range=0.25)
    
    #cropping image
    image = image_preprocess(image,64,64)
    image = np.array(image)
    
    #randomly flipping the image and corresponding angle
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        angle = -angle
    
    return image,angle


def augment_predict_image(line_data):
    path_file = line_data['center'][0].strip()
    image = plt.imread(path+path_file)
    image = image_preprocess(image,64,64)
    image = np.array(image)
    return image

#keras generator
def train_generator(data,batch_size = 32):
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for batch in range(batch_size):
            line_data = data.iloc[[np.random.randint(len(data))]].reset_index()
            
            keep_pr = 0
            while keep_pr == 0:
                image,angle = augment_train_image(line_data)
                if abs(angle)<.15:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            batch_images[batch] = image
            batch_steering[batch] = angle
        yield batch_images, batch_steering

def validation_generator(data):
    while 1:
        for line_index in range(len(data)):
            line_data = data.iloc[[line_index]].reset_index()
            image = augment_predict_image(data)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            angle = line_data['steering'][0]
            angle = np.array([[angle]])
            yield image, angle





def save_model(fileModelJSON,fileWeights,model):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        f.write(json_string)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)    
    model.save_weights(fileWeights)
