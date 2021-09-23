#fingerfrint name to use
ffpp = "pattern"

#read csv
import pandas as pd
train = pd.read_csv('./_data/samsung01/train.csv')
test = pd.read_csv('./_data/samsung01/test.csv')

ss = pd.read_csv('./_data/samsung01/sample_submission.csv')
dev = pd.read_csv('./_data/samsung01/dev.csv')

train = pd.concat([train,dev])

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

# !pip install kora -q
import kora.install.rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

import math
train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

print(np_train_fps_array.shape)
print(len(train_y))

pd.Series(np_train_fps_array[:,0]).value_counts()

import math
test_fps = []#test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

# print(np_test_fps_array.shape)
# print(len(test_y))

pd.Series(np_test_fps_array[:,0]).value_counts()

# np_test_fps_array = np.delete(np_test_fps_array,0,1)

# print(np_test_fps_array.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV, KFold

# 하이퍼 파라미터 모델

def create_deep_model_hyper(drop=0.5, optimizer=Adam(), units_1=128, units_2=128, units_3=128):
    model = Sequential()
    model.add(Dense(units=units_1, input_dim=2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=units_2, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(units=units_3, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer=optimizer(lr=0.01))
    return model

def create_hyperparameter():
    optimizers = [Adam]
    dropout = [0.3, 0.6, 0.8]
    nodes_1 = [128, 256, 512, 1024, 2048]
    nodes_2 = [128, 256, 512, 1024, 2048]
    nodes_3 = [128, 256]  
    epochs = [30, 40]
    return {"optimizer" : optimizers,
            "drop" : dropout, "epochs" : epochs, "units_1" : nodes_1, "units_2" : nodes_2, "units_3" : nodes_3, 
            }

hyperparameters = create_hyperparameter()

'''
# 기존 모델

def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
'''

X, Y = np_train_fps_array , np.array(train_y)

#validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

model = KerasRegressor(build_fn=create_deep_model_hyper, epochs=10)
model = RandomizedSearchCV(model, hyperparameters, cv=5)

estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', model))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

model.fit(X, Y, epochs = 20, callbacks=[reduce_lr])
test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y

ss.to_csv("./dacon/_data/_save/csv_samsung01/pattern_mlp.csv",index=False)