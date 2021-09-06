import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Flatten, Layer, ConvLSTM2D, Bidirectional
from keras.layers import Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.layers import LSTM, RNN, Embedding, TimeDistributed, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras import backend

def CNN(shape):
    model = Sequential()
    model.add(Conv2D(kernel_size=(3,3), filters=32, padding='same',data_format = 'channels_last', activation='swish',input_shape=(shape[0], shape[1], shape[2])))
    model.add(Conv2D(kernel_size=(3,3), filters=32, padding='same', activation='swish'))
    model.add(Dropout(0.15))
    #model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(kernel_size=(3,3), filters=64, padding='same',  activation='swish'))
    model.add(Conv2D(kernel_size=(3,3), filters=64, padding='same',  activation='swish'))
    model.add(Dropout(0.15))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(kernel_size=(3,3), filters=128, padding='same',  activation='swish'))
    model.add(Conv2D(kernel_size=(3,3), filters=128, padding='same',  activation='swish'))
    model.add(Dropout(0.15))
    #model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(kernel_size=(3,3), filters=256, padding='same',  activation='swish'))
    model.add(Conv2D(kernel_size=(3,3), filters=256, padding='same',  activation='swish'))
    model.add(Dropout(0.15))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(units = 1000,  activation='swish'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 1000,  activation='swish'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 1000,  activation='swish'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('linear'))

    return(model)
