import numpy as np
import pandas as pd
from tqdm import tqdm

def sEMG_Preprocessing(Base_Dir):

    df = pd.read_csv(Base_Dir+'sub1.csv')
    df.replace(np.nan,45,inplace=True)

    df['Air_breathing'] = ((df['Air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['V2'][:7113].values)+list(df['V2'][7113:21769].values)+list(df['V2'][21769:767871].values)
    Y = list(df['V1'][:7113].values)+list(df['V1'][7113+64:21769+64].values)+list(df['V1'][21769+128:767871+128].values)
    
    Xtrain, Xtest = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    Ytrain, Ytest = Y[:int(len(X)*0.8)], Y[int(len(X)*0.8):]

    df = pd.read_csv(Base_Dir+'sub2.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['V2'][:42565].values)+list(df['V2'][42565:64197].values)+list(df['V2'][64197:488005].values)+list(df['V2'][488005:767807])
    Y = list(df['V1'][:42565].values)+list(df['V1'][42565+64:64197+64].values)+list(df['V1'][64197+128:488005+128].values)+list(df['V1'][488005+192:767807+192].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]
    
    df = pd.read_csv(Base_Dir+'sub3.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['Val2'][:16814].values)+list(df['Val2'][16814:322798].values)+list(df['Val2'][322798:535086].values)+list(df['Val2'][535086:579807])
    Y = list(df['Val1'][:16814].values)+list(df['Val1'][16814+64:322798+64].values)+list(df['Val1'][322798+128:535086+128].values)+list(df['Val1'][535086+192:579807+192].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]

    df = pd.read_csv(Base_Dir+'sub4.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['Val2'][:774280].values)
    Y = list(df['Val1'][:690696].values)+list(df['Val1'][690696+64:774088+64].values)+list(df['Val1'][774088+128:774280+128].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]

    df = pd.read_csv(Base_Dir+'sub5.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['Val2'][:700000].values)
    Y = list(df['Val1'][:280124].values)+list(df['Val1'][280124+64:446716+64].values)+list(df['Val1'][446716+128:656892+128].values)+list(df['Val1'][656892+192:700000+192].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]

    df = pd.read_csv(Base_Dir+'sub6.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['Val2'][:82000].values)+list(df['Val2'][83000:101000].values)+list(df['Val2'][101500:131500].values)+list(df['Val2'][161500:271500].values)+list(df['Val2'][281500:311500].values)+list(df['Val2'][320000:432000].values)+list(df['Val2'][469000:530000].values) + list(df['Val2'][550000:782335].values)
    Y = list(df['Val1'][:82000].values)+list(df['Val1'][83000+128:101000+128].values)+list(df['Val1'][101500+192:131500+192].values)+list(df['Val1'][161500+320:271500+320].values)+list(df['Val1'][281500+448:311500+448].values)+list(df['Val1'][320000+512:432000+512].values)+list(df['Val1'][469000+640:530000+640].values)+list(df['Val1'][550000+1600:715349+1600].values)+list(df['Val1'][715349+1664:782335+1664].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]
    
    df = pd.read_csv(Base_Dir+'sub7.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['Val2'][:767807].values)
    Y = list(df['Val1'][:204477].values)+list(df['Val1'][204477+64:420413+64].values)+list(df['Val1'][420413+128:645885+128].values)+list(df['Val1'][645885+192:767807+192].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]
    
    df = pd.read_csv(Base_Dir+'sub8.csv')
    df.replace(np.nan,45,inplace=True)

    df['air_breathing'] = ((df['air_breathing']*10000).astype(int).astype(float))/10000
    df['emg_breathing'] = ((df['emg_breathing']*10000).astype(int).astype(float))/10000
    df['emg_resting'] = ((df['emg_resting']*10000).astype(int).astype(float))/10000

    X = list(df['Val2'][:330000].values)+ list(df['Val2'][350000:550000].values) + list(df['Val2'][560000:767551].values)
    Y = list(df['Val1'][:1138].values)+list(df['Val1'][1138+64:330000+64].values)+list(df['Val1'][350000+320:550000+320].values)+list(df['Val1'][560000+448:767551+448].values)
    
    Xtrain += X[:int(len(X)*0.8)]
    Xtest += X[int(len(X)*0.8):]
    Ytrain += Y[:int(len(X)*0.8)]
    Ytest += Y[int(len(X)*0.8):]
    
    time_steps = 128
    Xtrain_ = []
    Ytrain_ = []
    Xtest_ = []
    Ytest_ = []

    for i in tqdm(range(len(Xtrain)-time_steps+1)):
        X_ = np.array(Xtrain[i:i+time_steps])
        X_ = X_.reshape(X_.shape[0],1)
        Y_ = Ytrain[i+time_steps-1]
        Xtrain_.append(X_)
        Ytrain_.append(Y_)

    for i in tqdm(range(len(Xtest)-time_steps+1)):
        X_ = np.array(Xtest[i:i+time_steps])
        X_ = X_.reshape(X_.shape[0],1)
        Y_ = Ytest[i+time_steps-1]
        Xtest_.append(X_)
        Ytest_.append(Y_)
    
    Xtrain = np.array(Xtrain_)
    Xtest = np.array(Xtest_)
    Ytrain = np.array(Ytrain_)
    Ytest = np.array(Ytest_)
    
    return (Xtrain, Ytrain, Xtest, Ytest)

def Capno_Preprocessing(Path):
  
    EXT = "*signal.csv"
    all_csv_files = [file for path, subdir, files in os.walk(Path) for file in glob(os.path.join(path, EXT))]

    time_steps = 128
    Xtrain = []
    Ytrain = []

    for file in all_csv_files[:-5]:
        try:
            df = pd.read_csv(file)
           #X1,X2 = df['pleth_y'].values, df['ecg_y'].values
           #X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1)], axis=1)
            X1 = df['pleth_y'].values
            X = np.concatenate([X1.reshape(len(X1),1)], axis=1)
            Y = df['co2_y'].values

            for i in range(X.shape[0]-time_steps+1):
                X_ = X[i:i+time_steps,:]
                X_ = X_.reshape(X_.shape[0],X_.shape[1])
                Y_ = Y[i+time_steps-1]
                Xtrain.append(X_)
                Ytrain.append(Y_)
        except:
            continue

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    
    Xtest = []
    Ytest = []

    for file in all_csv_files[-5:]:
        try:
            df = pd.read_csv(file)
           #X1,X2 = df['pleth_y'].values, df['ecg_y'].values
           #X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1)], axis=1)
            X1 = df['pleth_y'].values
            X = np.concatenate([X1.reshape(len(X1),1)], axis=1)
            Y = df['co2_y'].values

            for i in range(X.shape[0]-time_steps+1):
                X_ = X[i:i+time_steps,:]
                X_ = X_.reshape(X_.shape[0],X_.shape[1])
                Y_ = Y[i+time_steps-1]
                Xtest.append(X_)
                Ytest.append(Y_)
        except:
            continue

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    
    return (Xtrain,Ytrain, Xtest,Ytest)
  
def MIMIC_II_Preprocessing(Path):
  
    EXT = "*Signals.csv"
    all_csv_files = [file for path, subdir, files in os.walk(Path) for file in glob(os.path.join(path, EXT))]

    time_steps = 32
    Xtrain = []
    Ytrain = []

    for file in all_csv_files[:-5]:
      try:
          df = pd.read_csv(file)
          X1, X2, X3, X4 = df[' PLETH'].values, df[' V'].values, df[' AVR'].values, df[' II'].values
          X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1),X3.reshape(len(X1),1),X4.reshape(len(X1),1)], axis=1)
          Y = df[' RESP'].values

          for i in range(X.shape[0]-time_steps+1):
              X_ = X[i:i+time_steps,:]
              X_ = X_.reshape(X_.shape[0],X_.shape[1])
              Y_ = Y[i+time_steps-1]
              Xtrain.append(X_)
              Ytrain.append(Y_)
      except:
        continue
        
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    
    Xtest = []
    Ytest = []

    for file in all_csv_files[-5:]:
      try:
          df = pd.read_csv(file)
          X1, X2, X3, X4 = df[' PLETH'].values, df[' V'].values, df[' AVR'].values, df[' II'].values
          X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1),X3.reshape(len(X1),1),X4.reshape(len(X1),1)], axis=1)
          Y = df[' RESP'].values

          for i in range(X.shape[0]-time_steps+1):
              X_ = X[i:i+time_steps,:]
              X_ = X_.reshape(X_.shape[0],X_.shape[1])
              Y_ = Y[i+time_steps-1]
              Xtest.append(X_)
              Ytest.append(Y_)
      except:
        continue
        
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    
    return (Xtrain,Ytrain, Xtest,Ytest)
