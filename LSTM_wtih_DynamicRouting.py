# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:33:06 2024

@author: azarf
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:33:06 2024

@author: azarf
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Rescaling,Layer
import tensorflow as tf
import numpy as np
import random as rn

class MyLayer(Layer):
  def __init__(self,unit1):
    super(MyLayer, self).__init__()

    #your variable goes here#[1,5,8]
    self.variable = tf.Variable(tf.random_normal_initializer()(shape=[1,5,unit1]), dtype=tf.float32, name="PoseEstimation", trainable=True)


  def call(self, inputs):

    # your mul operation goes here
    x = tf.multiply(inputs,self.variable)
    
    return x
def squash(s, epsilon = 1e-7):
  s_norm = tf.norm(s, axis=-1, keepdims=True)
  return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + epsilon)


def LSTM_with_DynamicRouting(input_shape,scale,myactivation,drop1,drop2,unit1,unit2,dyna):
    
    #model = Sequential()
    #model.add()
    inputs = Input(shape=input_shape) 
    x = tf.keras.layers.Masking(mask_value= 0,input_shape=input_shape)(inputs)
    # x = Embedding(input_dim = 10, output_dim = 5)(inputs)
    x = LSTM(unit1, activation='tanh', recurrent_activation='tanh', return_sequences=True)(x)#5,8
    x = Dropout(drop1)(x)
    u = tf.expand_dims(x,axis = -3) # u.shape: (None, 509, 1, 8)
    u_hat = MyLayer(unit1)(u)
    b = tf.zeros((input_shape[0],5,5))
    for i in range(dyna):
        c = tf.nn.softmax(b,axis = 1)
        s = tf.reduce_sum(tf.matmul(c,u_hat), axis = 1, keepdims = True)#5,8
        v = squash(s)#5,8
        agreement = tf.matmul(u_hat, v, transpose_b=True)
        b += agreement
    x = LSTM(unit2, activation='tanh', recurrent_activation='tanh')(tf.squeeze(v,axis = 1))
    x = Dropout(drop2)(x)
    x = Rescaling(scale, offset=0.0)(x)
    outputs = Dense(1, activation=myactivation)(x)
    model = Model(inputs, outputs)

    return model    

myseed =705
scale =2
myactivation ='tanh'
drop1 =0.1
drop2=0.1
unit1 =128
unit2=32
dyna =3
time_point = 5
feature = 40
input_shape = ( time_point, feature)
np.random.seed(myseed)
tf.random.set_seed(myseed)
rn.seed(myseed)
tf.experimental.numpy.random.seed(myseed)
LSTM_DyanRout_model = LSTM_with_DynamicRouting(input_shape,scale,myactivation,drop1,drop2,unit1,unit2,dyna)
