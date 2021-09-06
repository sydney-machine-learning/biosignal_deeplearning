import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Flatten, Layer, ConvLSTM2D, Bidirectional
from keras.layers import Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.layers import LSTM, RNN, Embedding, TimeDistributed, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras import backend

def Conv_LSTM(shape):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3), activation='swish', input_shape=shape,padding='same', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3), activation='swish', padding='same', return_sequences=True))
    model.add(Dropout(0.5))
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
