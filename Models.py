import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Flatten, Layer, ConvLSTM2D, Bidirectional
from keras.layers import Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.layers import LSTM, RNN, Embedding, TimeDistributed, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras import backend

class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),initializer="zeros")        
        super(Attention,self).build(input_shape)
        
    def call(self, x):        
        e = backend.tanh(backend.dot(x,self.W)+self.b)
        a = backend.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
          
        return backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'return_sequences': self.return_sequences})
        return config
      
def Bi_LSTM(shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(512,input_shape=(shape[0], shape[1]), return_sequences=True)))
    model.add(Dropout(0.25))   
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.25))
    #model.add(Attention(return_sequences=False))
    
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

def Bi_LSTM_Attn(shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(512,input_shape=(shape[0], shape[1]), return_sequences=True)))
    model.add(Dropout(0.25))   
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(Attention(return_sequences=False))    
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

def LSTM(shape):
    model = Sequential()
    model.add(LSTM(512,input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(Dropout(0.25))   
    model.add(LSTM(256))
    model.add(Dropout(0.25))
    #model.add(Attention(return_sequences=False))
    
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

def LSTM_Attn(shape):
    model = Sequential()
    model.add(LSTM(512,input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(Dropout(0.25))   
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Attention(return_sequences=False))
    
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
  
def CNN_LSTM(shape):
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
