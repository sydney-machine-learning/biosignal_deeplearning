# !pip install xlrd==1.2.0 Use version less than 1.9.0

import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from expt_LSTM import evaluate_model, summarize_results, run_experiment

df = pd.read_excel('../input/biosignals/Database2.xlsx', sheet_name=[0,1,2], header=None)

Values = []
labels = []
for i in range(3):
    for j in range(len(df[i])):
        A = df[i].loc[j][:2500].values
        B = df[i].loc[j][2501:].values
        Values.append(np.concatenate([A,B]).reshape((2,2500)).T)
        
        if j<100:
            labels.append(0)
        elif j>=105 and j<205:
            labels.append(1)
        elif j>=210 and j<310:
            labels.append(2)
        elif j>=315 and j<415:
            labels.append(3)
        elif j>=420 and j<520:
            labels.append(4)
        elif j>=525 and j<625:
            labels.append(5)
        else:
            Values.pop()

Values = np.dstack(Values).transpose([2,0,1])
labels = np.dstack(labels).transpose([2,0,1]).flatten()

X_train, X_test,Y_train,Y_test = train_test_split(Values, labels, test_size=0.2, random_state=42, stratify = labels)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

run_experiment(X_train, Y_train, X_test, Y_test)