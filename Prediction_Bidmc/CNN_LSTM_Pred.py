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
ch1 = ModelCheckpoint('model_cnn_lstm_9.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
ch2 = EarlyStopping(monitor="val_loss",patience=150,verbose=0,mode="min")
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='swish', padding='same'), input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='swish', padding='same')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='swish', padding='same')))
model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='swish', padding='same')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='swish', padding='same')))
model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='swish', padding='same')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
#model.add(TimeDistributed(Conv1D(filters=512, kernel_size=3, activation='swish', padding='same')))
#model.add(TimeDistributed(Conv1D(filters=512, kernel_size=3, activation='swish', padding='same')))
#model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Flatten()))
#model.add(Flatten())
model.add(LSTM(1024))
model.add(Dropout(0.5))
model.add(Dense(512, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=[rmse])
history = model.fit(Xtrain, Ytrain, validation_data = (Xval, Yval), epochs=2000, batch_size=5000, callbacks=[ch1,ch2])

Ypred = model.predict(Xval)
file_name = "Ypred9.pkl"

open_file = open(file_name, "wb")
pickle.dump(Ypred, open_file)
open_file.close()
