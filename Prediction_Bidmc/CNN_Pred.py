import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense
from keras.layers import Flatten, Layer
from keras.layers import Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.layers import LSTM, RNN, Embedding
from keras.layers import TimeDistributed, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from glob import glob
from keras import backend
import tensorflow_addons as tfa

PATH = "../input/bidmc-pred/bidmc_csv/"
EXT = "*Signals.csv"
all_csv_files = [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))]

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

time_steps = 30
Xtrain = []
Ytrain = []

for file in all_csv_files:
  try:
      df = pd.read_csv(file)
      X1, X2, X3, X4 = df[' PLETH'].values, df[' V'].values, df[' AVR'].values, df[' II'].values
      X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1),X3.reshape(len(X1),1),X4.reshape(len(X1),1)], axis=1)
      Y = df[' RESP'].values

      for i in range(X.shape[0]-time_steps+1):
          X_ = X[i:i+time_steps,:]
          X_ = X_.reshape(X_.shape[0],X_.shape[1],1)
          Y_ = Y[i+time_steps-1]
          Xtrain.append(X_)
          Ytrain.append(Y_)
  except:
      continue
'''

df = pd.read_csv(all_csv_files[11])
X1, X2, X3, X4 = df[' PLETH'].values, df[' V'].values, df[' AVR'].values, df[' II'].values
X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1),X3.reshape(len(X1),1),X4.reshape(len(X1),1)], axis=1)
Y = df[' RESP'].values

for i in range(X.shape[0]-time_steps+1):
    X_ = X[i:i+time_steps,:]
    X_ = X_.reshape(X_.shape[0],X_.shape[1],1)
    Y_ = Y[i+time_steps-1]
    Xtrain.append(X_)
    Ytrain.append(Y_) 
'''
X = np.array(Xtrain)
Y = np.array(Ytrain)
Xtrain, Xval, Ytrain, Yval = train_test_split(X,Y,test_size=0.25, random_state=42)
'''

Xtrain = pickle.load(open('../input/bidmc-subject2/Xtrain.pkl','rb'))
Xval = pickle.load(open('../input/bidmc-subject2/Xval.pkl','rb'))
Ytrain = pickle.load(open('../input/bidmc-subject2/Ytrain.pkl','rb'))
Yval = pickle.load(open('../input/bidmc-subject2/Yval.pkl','rb'))

Xtrain = Xtrain.reshape((-1,30,4,1))
Xval = Xval.reshape((-1,30,4,1))
'''

model = Sequential()

model.add(Conv2D(kernel_size=(2,1), filters=32, padding='same',data_format = 'channels_last', name='layer_conv1', activation='swish',input_shape=(30, 4, 1)))
model.add(Conv2D(kernel_size=(2,1), filters=32, padding='same', activation='swish', name='layer_conv2'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = (2,1)))

model.add(Conv2D(kernel_size=(2,2), filters=64, padding='same',  activation='swish',name='layer_conv4'))
model.add(Conv2D(kernel_size=(2,2), filters=64, padding='same',  activation='swish',name='layer_conv5'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = (1,2)))

model.add(Conv2D(kernel_size=(2,2), filters=128, padding='same',  activation='swish', name='layer_conv7'))
model.add(Conv2D(kernel_size=(2,2), filters=128, padding='same',  activation='swish',name='layer_conv8'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = (2,1)))

model.add(Conv2D(kernel_size=(2,2), filters=256, padding='same',  activation='swish', name='layer_conv9'))
model.add(Conv2D(kernel_size=(2,2), filters=256, padding='same',  activation='swish',name='layer_conv10'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = (1,2)))

model.add(Flatten())
model.add(Dense(units = 512,  activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(units = 256,  activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(units = 128,  activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(units = 64,  activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('linear'))

model.summary()

ch1 = ModelCheckpoint('model_conv_all_2.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
ch2 = EarlyStopping(monitor="val_loss",patience=150,verbose=0,mode="min")

model.compile(loss='mse', optimizer='adam', metrics=[rmse])
history = model.fit(Xtrain, Ytrain, validation_data = (Xval, Yval), epochs=2000, batch_size=5000, callbacks=[ch1,ch2])

Ypred = model.predict(Xval)
file_name = "Ypred2.pkl"

open_file = open(file_name, "wb")
pickle.dump(Ypred, open_file)
open_file.close()
