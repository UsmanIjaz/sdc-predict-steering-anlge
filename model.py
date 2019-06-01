import os
import pandas as pd
import numpy as np
from scipy import signal
import cv2
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import utils
import gc

tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras import initializations

from pathlib import Path
import json

#Nvidia's Model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
#first layer to normalize data

def get_model():
    input_shape = (64, 64, 3)
    filter_size = 3
    pool_size = (2,2)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1,input_shape=input_shape))
    model.add(Convolution2D(24,5,5,border_mode='valid', subsample=(2,2)))
    model.add(ELU())
    
    model.add(Convolution2D(36,5,5,border_mode='valid', subsample=(2,2)))
    model.add(ELU())
    
    model.add(Convolution2D(48,5,5,border_mode='valid', subsample=(2,2)))
    model.add(ELU())
    
    model.add(Convolution2D(64,3,3,border_mode='valid', subsample=(1,1)))
    model.add(ELU())
    
    model.add(Convolution2D(64,3,3,border_mode='valid', subsample=(1,1)))
    model.add(ELU())
    
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dropout(0.5))
    
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    
    return model
    

#using udacity data    
path = 'data/'
csv_path = 'data/driving_log.csv'
df = pd.read_csv(csv_path,index_col = False)
ind = df['throttle']>.25
df= df[ind].reset_index()
image_c = plt.imread(path+df['center'][0].strip())
rows,cols,channels = image_c.shape


pr_threshold = 1
batch_size = 256
best_model = 0
val_best = 1000
iterations = 30


model = get_model()

#using Adam's Optimizer
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam,loss='mse')

#validation data generation
valid_gen = utils.validation_generator(df)

for iteration in range(iterations):

    train_gen = utils.train_generator(df,batch_size)
    history = model.fit_generator(train_gen,
            samples_per_epoch=256*79, nb_epoch=1,validation_data=valid_gen,
                        nb_val_samples=len(df))
    
    utils.save_model('model_' + str(iteration) + '.json','model_' + str(iteration) + '.h5',model)
    
    val_loss = history.history['val_loss'][0]
    if val_loss < val_best:
        best_model = iteration 
        val_best = val_loss
        utils.save_model('model.json','model.h5',model)
    
    
    pr_threshold = 1/(iteration+1)

print('Best model found at iteration # ' + str(best_model))
print('Best Validation score : ' + str(np.round(val_best,4)))




gc.collect()