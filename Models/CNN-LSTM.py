import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Flatten, Layer, ConvLSTM2D, Bidirectional
from keras.layers import Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.layers import LSTM, RNN, Embedding, TimeDistributed, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras import backend

def CNN_LSTM(shape):
    model = Sequential()
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='swish', padding='same'), input_shape=(None,shape[1],shape[2])))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='swish', padding='same')))
    #model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='swish', padding='same')))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='swish', padding='same')))
    #model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='swish', padding='same')))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='swish', padding='same')))
    #model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='swish', padding='same')))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='swish', padding='same')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Flatten()))
    #model.add(Flatten())
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='swish'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='swish'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='swish'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    return(model)
