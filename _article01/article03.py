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

# #############################################################

# dataset = pd.read_csv(filepath + fname)
# x = dataset.loc[:,['title']]
# y = dataset.loc[:,['topic_idx']]

# print(x.shape)  # (45654, 1)
# print(y.shape)  # (45654, 1)
# print(np.unique(y))
# print(type(x))  # <class 'pandas.core.frame.DataFrame'>

# # 1_1. 데이터 전처리

# def clean_text(document):
#     document = re.sub("[.,!?'\":;~()]","",document)
#     document = re.sub("[^ㄱ-ㅣ가-힣A-Za-z]"," ",document)
#     return document

# x["title"] = x["title"].apply(lambda x:clean_text(x))

# x = np.array(x)
# y = np.array(y)
# print(type(x))  # <class 'numpy.ndarray'>

# x = x.reshape(45654)
# print(x.shape)  # (45654,)

# #############################################################

# dataset_test = pd.read_csv(filepath + fname_test)
# x_predict = dataset_test.loc[:,['title']]

# x_predict["title"] = x_predict["title"].apply(lambda x:clean_text(x))

# x_predict = np.array(x_predict)
# x_predict = x_predict.reshape(9131)
# print(x_predict.shape) # (9131)

# # #############################################################

# a1 = np.where(y == 0)
# a2 = np.where(y == 1)
# a3 = np.where(y == 2)
# a4 = np.where(y == 3)
# a5 = np.where(y == 4)
# a6 = np.where(y == 5)
# a7 = np.where(y == 6)

# x0 = x[a1[0]]
# x1 = x[a2[0]]
# x2 = x[a3[0]]
# x3 = x[a4[0]]
# x4 = x[a5[0]]
# x5 = x[a6[0]]
# x6 = x[a7[0]]

# # split_함수를 이용해 같은 카테고리의 행끼리 concatenate 한후 데이터셋 추가

# array1 = np.array(range(len(x0)))
# size = 3

# def split_x(a, size):
#     aaa = []
#     for i in range(len(a) - size + 1): 
#         subset = a[i : (i + size)] 
#         aaa.append(subset)
#     return np.array(aaa)

# x0 = split_x(x0, size)
# x1 = split_x(x1, size)
# x2 = split_x(x2, size)
# x3 = split_x(x3, size)
# x4 = split_x(x4, size)
# x5 = split_x(x5, size)
# x6 = split_x(x6, size)

# def cons(a):
#     aaa = []
#     for i in range(len(a)):
#         subset = a[i][0]+a[i][1]+a[i][2]
#         aaa.append(subset)
#     return np.array(aaa)

# x0 = cons(x0)
# x1 = cons(x1)
# x2 = cons(x2)
# x3 = cons(x3)
# x4 = cons(x4)
# x5 = cons(x5)
# x6 = cons(x6)

# def cons_y(a, type):
#     aaa = []
#     for i in range(len(a)):
#         subset = type
#         aaa.append(subset)
#     return np.array(aaa)

# y0 = cons_y(x0, 0)
# y1 = cons_y(x1, 1)
# y2 = cons_y(x2, 2)
# y3 = cons_y(x3, 3)
# y4 = cons_y(x4, 4)
# y5 = cons_y(x5, 5)
# y6 = cons_y(x6, 6)

# x_cons = np.concatenate((x0,x1,x2,x3,x4,x5,x6))
# y_cons = np.concatenate((y0,y1,y2,y3,y4,y5,y6))

# print(x_cons[0])

# # np.save

# np.save('./dacon/_data/_npy/x.npy', arr=x)
# np.save('./dacon/_data/_npy/y.npy', arr=y)
# np.save('./dacon/_data/_npy/x_predict.npy', arr=x_predict)
# np.save('./dacon/_data/_npy/x_cons.npy', arr=x_cons)
# np.save('./dacon/_data/_npy/y_cons.npy', arr=y_cons)

# ValueError: Object arrays cannot be loaded when allow_pickle=False 오류 해결

np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

x = np.load('./dacon/_data/_npy/x.npy')
y = np.load('./dacon/_data/_npy/y.npy')
x_predict = np.load('./dacon/_data/_npy/x_predict.npy')
x_cons = np.load('./dacon/_data/_npy/x_cons.npy')
y_cons = np.load('./dacon/_data/_npy/y_cons.npy')

print(y.shape, y_cons.shape)
y_cons = y_cons.reshape(45640, 1)

# 기존의 데이터와 augment 한 데이터를 병합한다.
x = np.concatenate((x,x_cons))
y = np.concatenate((y,y_cons))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

token = Tokenizer()

token.fit_on_texts(x)
token.fit_on_texts(x_predict)
# print(token.word_index) # ~1197

x = token.texts_to_sequences(x)
x_predict = token.texts_to_sequences(x_predict)
# print(x)

x = pad_sequences(x, maxlen=20, padding='pre')
x_predict = pad_sequences(x_predict, maxlen=20, padding='pre')
print(x.shape)  # (45654, 10)
print(np.unique(x))
# [     0      1      2 ... 101079 101080 101081]
print(x_predict.shape)
print(np.unique(x_predict))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True, random_state=66)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer

# scaler = MinMaxScaler()
# scaler = PowerTransformer()
scaler = QuantileTransformer()
scaler.fit_transform(x_train)
scaler.transform(x_test)
scaler.transform(x_predict)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, Bidirectional, MaxPool1D

model = Sequential()
model.add(Embedding(input_dim=150000, output_dim=128, input_length=20))
# model.add(LSTM(64, return_sequences=True))
# model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Conv1D(128, 2, padding='same'))
model.add(MaxPool1D())
model.add(Dropout(0.6))
model.add(Conv1D(128, 2, padding='same'))
model.add(MaxPool1D())
model.add(Dropout(0.6))
model.add(Conv1D(128, 2, padding='same'))
model.add(MaxPool1D())
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(1024, activation='relu'))
model.add(Dense(7, activation='softmax'))

######################################################################

# start_time = time.time()
# filepath = './dacon/_data/_save/'
# fname = 'dacon0804_0035_.0167-0.2142.hdf5'
# model = load_model(filepath + fname)
# end_time = time.time() - start_time

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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                filepath=modelpath)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=30, batch_size=256, verbose=1, validation_split=0.1, callbacks=[es, mcp])
end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
pre = model.predict(x_predict)

# 5. plt 시각화

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1)
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.legend(loc='upper right')

# 2)
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# to_csv

print(pre)
print(pre.shape)

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
전처리 이후 acc, loss 안정됨
베이스라인에 Dense 모델이 성능이 좋아보이니 테스트
Bidiractional 테스트
시퀀셜 모델 acc 0.7 중반 아쉽...
데이터가 적어 loss값을 잡지못하는것이라면
opimizer=sgd 테스트 -> 망

동일한 카테고리의 값끼리 병합하여 train데이터에 더해보자
4.5만->9.0만 데이터크기 증가
val_loss : 0.3168591856956482
val_acc : 0.9315406084060669
소요 시간 : 3.714602478345235
loss와 acc가 눈에띄게 좋아진것을 볼수있다

하지만 데이컨 평가 결과 69점으로 기존 점수보다 낮음
-> 3가지 이유 1) 데이터의 신뢰도 낮음 2) 내 모델이 별로
3) 데이콘 평가기준이 실제 데이터의 30~50%이기 때문에, 퍼블릭 결과가
나쁠수 있다. -> 피드백으론... 내가 조작한 데이터가 신뢰할수 있는가에 무게가실림.
기존의 데이터로 append된 데이터인데 신뢰할수 없다라는게 이해되진 않음...


'''