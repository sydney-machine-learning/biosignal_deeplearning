import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Flatten, Layer, ConvLSTM2D, Bidirectional
from keras.layers import Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.layers import LSTM, RNN, Embedding, TimeDistributed, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D

def BiLSTM(shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(512,input_shape=(shape[0], shape[1]), return_sequences=True)))
    model.add(Dropout(0.25))   
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.25))
    #model.add(Attention(return_sequences=False))
    
    model.add(Flatten())
    model.add(Dense(units = 1024,  activation='swish'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 1024,  activation='swish'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 1024,  activation='swish'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('linear'))
    return(model)
    
