# -*- coding: utf-8 -*-
"""
Name: D.R.P.R.M. Lustermans
Version: 1
Date: 01-03-2021
Email: d.r.p.r.m.lustermans@student.tue.nl
GitHub: https://github.com/DidierLustermans
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam


def bounding_box_CNN(bb_stride, bb_kernel, shape, alpha_l2, bb_learning_rate, double_conv = False):
    (xdim, ydim) = shape
    # Create the bounding box CNN
    # Make the model
    model = Sequential()
    # Layer 1
    model.add(Conv2D(8,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (xdim,ydim,1)))
    if double_conv:
        model.add(BatchNormalization())
        model.add(Conv2D(8,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (xdim,ydim,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Conv2D(16,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (int(xdim/2),int(ydim/2),1)))
    if double_conv:
        model.add(BatchNormalization())
        model.add(Conv2D(16,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (int(xdim/2),int(ydim/2),1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())

    # Layer 3
    model.add(Conv2D(32,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (int(xdim/4),int(ydim/4),1)))
    if double_conv:
        model.add(BatchNormalization())   
        model.add(Conv2D(32,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (int(xdim/4),int(ydim/4),1))) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())

    # Layer 4
    model.add(Conv2D(64,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (int(xdim/8),int(ydim/4),1)))
    if double_conv:
        model.add(BatchNormalization())   
        model.add(Conv2D(64,bb_kernel, strides = bb_stride, kernel_regularizer = l2(alpha_l2), activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', input_shape = (int(xdim/8),int(ydim/4),1))) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())

    # Put your feature maps into one bog vector
    model.add(Flatten())
    # Add the fully connected layers to your program
    model.add(Dense(512, kernel_regularizer = l2(alpha_l2), activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(4, activation = 'linear'))
    
    # Compile your program
    opt = Adam(bb_learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    # model.summary()

    return model
