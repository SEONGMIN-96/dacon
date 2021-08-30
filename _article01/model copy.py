from itertools import groupby
from operator import mod
from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np
from pandas.tseries.offsets import Second
import time
import re


# 1. 데이터

filepath = './dacon/_data/'
fname = 'train_data.csv'
fname_test = 'test_data.csv'

np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

x = np.load('./dacon/_data/_npy/x_data.npy')
y = np.load('./dacon/_data/_npy/y_data.npy')
x_predict = np.load('./dacon/_data/_npy/x_predict.npy')

x = x.reshape(91294)
x_predict = x_predict.reshape(9131)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# AttributeError: 'numpy.ndarray' object has no attribute 'lower'
# reshape 2dim to 1dim 

token = Tokenizer()

token.fit_on_texts(x)
token.fit_on_texts(x_predict)
# print(token.word_index) # ~1197

x = token.texts_to_sequences(x)
x_predict = token.texts_to_sequences(x_predict)

x = pad_sequences(x, maxlen=20, padding='pre')
x_predict = pad_sequences(x_predict, maxlen=20, padding='pre')

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True, random_state=23)

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer

# scaler = MinMaxScaler()
# scaler = PowerTransformer()
# scaler = QuantileTransformer()
# scaler.fit_transform(x_train)
# scaler.transform(x_test)
# scaler.transform(x_predict)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, Bidirectional, MaxPool1D, BatchNormalization

# model = Sequential()
# model.add(Embedding(input_dim=150000, output_dim=128, input_length=20))
# # model.add(LSTM(64, return_sequences=True))
# # model.add(Bidirectional(LSTM(64, return_sequences=True)))
# model.add(Conv1D(128, 2, padding='same'))
# model.add(MaxPool1D())
# model.add(Dropout(0.6))
# model.add(Conv1D(128, 2, padding='same'))
# model.add(MaxPool1D())
# model.add(Dropout(0.6))
# model.add(BatchNormalization())
# model.add(Conv1D(128, 2, padding='same'))
# model.add(MaxPool1D())
# model.add(Dropout(0.6))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(7, activation='softmax'))

######################################################################

start_time = time.time()
filepath = './dacon/_data/_save/'
fname = 'dacon0809_1544_.0123-0.1930.hdf5'
model = load_model(filepath + fname)
end_time = time.time() - start_time

######################################################################

model.summary()

# 3. 컴파일, 훈련

######################################################################

import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath_save = './dacon/_data/_save/'
filename = '.{epoch:04d}-{loss:.4f}.hdf5'
modelpath = "".join([filepath_save, "dacon", date_time, "_", filename])

######################################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.001)

es = EarlyStopping(monitor='val_loss', mode='min', patience=200)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                filepath=modelpath)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, factor=0.2)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='acc')

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=2000, batch_size=512, verbose=1, validation_split=0.2, callbacks=[es, mcp])
# end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
pre = model.predict(x_predict)

# 5. plt 시각화

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# # 1)
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.legend(loc='upper right')

# # 2)
# plt.subplot(2,1,2)
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()

# # to_csv

pre = np.argmax(pre, axis=1)
ans = pd.DataFrame(pre)
ans.index = ans.index + 45654
ans.reset_index(inplace=True)
ans.rename(columns={0:'topic_idx'}, inplace=True)
ans.to_csv(filepath_save + date_time + '_predict.csv', index=False)

print('val_loss :', loss[0])
print('val_acc :', loss[1])
print('소요 시간 :', end_time/60)

'''

val_loss : 0.3408834934234619
val_acc : 0.9319787621498108
소요 시간 : 5.321057093143463

'''