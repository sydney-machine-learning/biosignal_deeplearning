import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from glob import glob
from keras import backend

from Models import Bi_LSTM, Bi_LSTM_Attn, LSTM, LSTM_Attn, CNN, CNN_LSTM, Conv_LSTM
from Preprocess import sEMG_Preprocessing, Capno_Preprocessing, MIMIC_II_Preprocessing 

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def mae(y_true, y_pred):
    return backend.mean(np.abs(y_pred - y_true), axis=-1)

# Path = Name of Directory for each dataset
Xtrain, Ytrain, Xtest, Ytest = Capno_Preprocessing(Path)
#Xtrain, Ytrain, Xtest, Ytest = MIMIC_II_Preprocessing(Path)
#Xtrain, Ytrain, Xtest, Ytest = sEMG_Preprocessing(Path)

Results = {}
Results['Bi-lstm'] = {}
Results['Bi-lstm']['RMSE'] = []
Results['Bi-lstm']['MAE'] = []
Results['Bi-lstm-attn'] = {}
Results['Bi-lstm-attn']['RMSE'] = []
Results['Bi-lstm-attn']['MAE'] = []
Results['lstm'] = {}
Results['lstm']['RMSE'] = []
Results['lstm']['MAE'] = []
Results['lstm-attn'] = {}
Results['lstm-attn']['RMSE'] = []
Results['lstm-attn']['MAE'] = []
Results['cnn-lstm'] = {}
Results['cnn-lstm']['RMSE'] = []
Results['cnn-lstm']['MAE'] = []
Results['conv-lstm'] = {}
Results['conv-lstm']['RMSE'] = []
Results['conv-lstm']['MAE'] = []
Results['cnn'] = {}
Results['cnn']['RMSE'] = []
Results['cnn']['MAE'] = []

runs = 10
for m in range(runs):
    print("Run {} started".format(m+1))
    print("Bi-LSTM training started")
    Xtrain = Xtrain.reshape(-1,32,4)
    Xtest = Xtest.reshape(-1,32,4)
    model1 = Bi_LSTM(Xtrain.shape[1:])
    model1.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/Bi_lstm/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history1 = model1.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model1.predict(Xtest)
    np.save('Saved/Capno/Bi_lstm/{}.npy'.format(m),Ypred)
    Results['Bi-lstm']['RMSE'].append(rmse(Ytest, Ypred))
    Results['Bi-lstm']['MAE'].append(mae(Ytest, Ypred))
    
    print("Bi-LSTM-Attn training started")
    Xtrain = Xtrain.reshape(-1,32,4)
    Xtest = Xtest.reshape(-1,32,4)
    model2 = Bi_LSTM_Attn(Xtrain.shape[1:])
    model2.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/Bi_lstm_attn/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history2 = model2.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model2.predict(Xtest)
    np.save('Saved/Capno/Bi_lstm_attn/{}.npy'.format(m),Ypred)
    Results['Bi-lstm-attn']['RMSE'].append(rmse(Ytest, Ypred))
    Results['Bi-lstm-attn']['MAE'].append(mae(Ytest, Ypred))
    
    print("LSTM training started")
    Xtrain = Xtrain.reshape(-1,32,4)
    Xtest = Xtest.reshape(-1,32,4)
    model3 = LSTM(Xtrain.shape[1:])
    model3.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/Lstm/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history3 = model3.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model3.predict(Xtest)
    np.save('Saved/Capno/lstm_attn/{}.npy'.format(m),Ypred)
    Results['lstm']['RMSE'].append(rmse(Ytest, Ypred))
    Results['lstm']['MAE'].append(mae(Ytest, Ypred))
    
    print("LSTM-Attn training started")
    Xtrain = Xtrain.reshape(-1,32,4)
    Xtest = Xtest.reshape(-1,32,4)
    model4 = LSTM_Attn(Xtrain.shape[1:])
    model4.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/Lstm_attn/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history4 = model4.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model4.predict(Xtest)
    np.save('Saved/Capno/lstm_attn/{}.npy'.format(m),Ypred)
    Results['lstm-attn']['RMSE'].append(rmse(Ytest, Ypred))
    Results['lstm-attn']['MAE'].append(mae(Ytest, Ypred))
    
    print("CNN training started")
    Xtrain = Xtrain.reshape(-1,32,4,1)
    Xtest = Xtest.reshape(-1,32,4,1)
    model5 = CNN(Xtrain.shape[1:])
    model5.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/CNN/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history5 = model5.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model5.predict(Xtest)
    np.save('Saved/Capno/lstm_attn/{}.npy'.format(m),Ypred)
    Results['cnn']['RMSE'].append(rmse(Ytest, Ypred))
    Results['cnn']['MAE'].append(mae(Ytest, Ypred))
    
    print("CNN-LSTM training started")
    Xtrain = Xtrain.reshape(-1,4,8,4)
    Xtest = Xtest.reshape(-1,4,8,4)
    model = CNN_LSTM(Xtrain.shape[1:])
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/Cnn_lstm/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history6 = model.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model6.predict(Xtest)
    np.save('Saved/Capno/lstm_attn/{}.npy'.format(m),Ypred)
    Results['cnn-lstm']['RMSE'].append(rmse(Ytest, Ypred))
    Results['cnn-lstm']['MAE'].append(mae(Ytest, Ypred))
    
    print("Conv-LSTM training started")
    Xtrain = Xtrain.reshape(-1,4,1,8,4)
    Xtest = Xtest.reshape(-1,4,1,8,4)
    model = Conv_LSTM(Xtrain.shape[1:])
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])    
    ch1 = ModelCheckpoint('Saved/Capno/Conv_lstm/{}.h5'.format(m), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    ch2 = EarlyStopping(monitor="val_loss",patience=100,verbose=0,mode="min")
    history7 = model.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), epochs=2000, batch_size=100000, callbacks=[ch1,ch2], verbose=0)
    Ypred = model3.predict(Xtest)
    np.save('Saved/Capno/lstm_attn/{}.npy'.format(m),Ypred)
    Results['conv-lstm']['RMSE'].append(rmse(Ytest, Ypred))
    Results['conv-lstm']['MAE'].append(mae(Ytest, Ypred))

with open("Results.json","w") as file:
    json.dump(file, Results)
    file.close()
