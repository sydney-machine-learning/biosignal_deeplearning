import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import pickle
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
from glob import glob
from keras import backend
import tensorflow_addons as tfa

from glob import glob
PATH = "../input/bidmc-pred/bidmc_csv/"
EXT = "*Signals.csv"
all_csv_files = [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))]

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation='swish')(x)
        x = Dropout(dropout_rate)(x)
    return x

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # print(positions)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

learning_rate = 0.001
weight_decay = 0.0001
projection_dim = 256
num_heads = 12

transformer_units = [32, 4]  
transformer_layers = 12
input_shape = (30, 4)

inputs = Input(shape=input_shape, name='1')
#encoded_patches = PatchEncoder(30, projection_dim)(inputs)
#encoded_patches = Dense(projection_dim, activation='swish',name='2')(inputs)

for _ in range(transformer_layers):
    x1 = LayerNormalization(epsilon=1e-6)(inputs)
    x1 = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.2)(x1, x1)
    x1 = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.2)(x1, x1)
    x2 = Add()([x1, inputs])
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=[512,128], dropout_rate=0.2)
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.2)
    encoded_patches = Add()([x3, x2])

x = LayerNormalization(epsilon=1e-6)(encoded_patches)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='swish')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='swish')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='swish')(x)
x = Dropout(0.5)(x)
logits = Dense(1, activation='linear')(x)
model = Model(inputs=inputs, outputs=logits)

Xtrain = []
Ytrain = []
val_loss = []
train_loss = []
time_steps = 30

df = pd.read_csv(all_csv_files[11])
X1, X2, X3, X4 = df[' PLETH'].values, df[' V'].values, df[' AVR'].values, df[' II'].values
X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1),X3.reshape(len(X1),1),X4.reshape(len(X1),1)], axis=1)
Y = df[' RESP'].values

for i in range(X.shape[0]-time_steps+1):
    X_ = X[i:i+time_steps,:]
    X_ = X_.reshape(X_.shape[0],X_.shape[1])
    Y_ = Y[i+time_steps-1]
    Xtrain.append(X_)
    Ytrain.append(Y_)
 
X = np.array(Xtrain)
Y = np.array(Ytrain)

Xtrain, Xval, Ytrain, Yval = train_test_split(X,Y,test_size=0.2, random_state=42)

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
ch1 = ModelCheckpoint('model_transformers_3.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
ch2 = EarlyStopping(monitor="val_loss",patience=150,verbose=0,mode="min")

model.compile(loss='mse', optimizer=optimizer, metrics=[rmse])
history = model.fit(Xtrain, Ytrain, validation_data = (Xval, Yval), epochs=2000, batch_size=1000, callbacks=[ch1,ch2])

Ypred = model.predict(Xval)
file_name = "Ypred3.pkl"

open_file = open(file_name, "wb")
pickle.dump(Ypred, open_file)
open_file.close()
