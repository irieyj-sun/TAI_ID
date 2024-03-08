# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:56:12 2024

@author: azarf
"""
import tensorflow as tf  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Rescaling
import numpy as np
import random as rn


def My_LSTM(input_shape,scale,myactivation,drop1,drop2,unit1,unit2):
    

    inputs = Input(shape=input_shape) 
    x = tf.keras.layers.Masking(mask_value= 0,input_shape=input_shape)(inputs)
    x = LSTM(unit1, activation='tanh', recurrent_activation='tanh', return_sequences=True)(x)#5,8
    x = Dropout(drop1)(x)
    x = LSTM(unit2, activation='tanh', recurrent_activation='tanh')(x)
    x = Dropout(drop2)(x)
    x = Rescaling(scale, offset=0.0)(x)
    outputs = Dense(1, activation=myactivation)(x)
    model = Model(inputs, outputs)

    return model


myseed = 100
scale = 2
myactivation = 'swish'
drop1 =0.2
drop2 =0.2
unit1= 8
unit2= 8
time_point = 5
feature = 40
input_shape = ( time_point, feature)
np.random.seed(myseed)
tf.random.set_seed(myseed)
rn.seed(myseed)
tf.experimental.numpy.random.seed(myseed)
My_LSTM = My_LSTM(input_shape,scale,myactivation,drop1,drop2,unit1,unit2)