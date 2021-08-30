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

#############################################################

dataset = pd.read_csv(filepath + fname)
x = dataset.loc[:,['title']]
y = dataset.loc[:,['topic_idx']]

def clean_text(document):
    document = re.sub("[.,!?'\":;~()]","",document)
    document = re.sub("[^ㄱ-ㅣ가-힣A-Za-z]"," ",document)
    return document

x["title"] = x["title"].apply(lambda x:clean_text(x))

print(x.shape)  # (45654, 1)
print(y.shape)  # (45654, 1)
print(np.unique(y))
print(type(x))  # <class 'pandas.core.frame.DataFrame'>

x = np.array(x)
y = np.array(y)

a1 = np.where(y == 0)
a2 = np.where(y == 1)
a3 = np.where(y == 2)
a4 = np.where(y == 3)
a5 = np.where(y == 4)
a6 = np.where(y == 5)
a7 = np.where(y == 6)

x0 = x[a1[0]]
x1 = x[a2[0]]
x2 = x[a3[0]]
x3 = x[a4[0]]
x4 = x[a5[0]]
x5 = x[a6[0]]
x6 = x[a7[0]]

# split_함수를 이용해 같은 카테고리의 행끼리 concatenate 한후 데이터셋 추가

array1 = np.array(range(len(x0)))
size = 3

def split_x(a, size):
    aaa = []
    for i in range(len(a) - size + 1): 
        subset = a[i : (i + size)] 
        aaa.append(subset)
    return np.array(aaa)

x0 = split_x(x0, size)
x1 = split_x(x1, size)
x2 = split_x(x2, size)
x3 = split_x(x3, size)
x4 = split_x(x4, size)
x5 = split_x(x5, size)
x6 = split_x(x6, size)

def cons(a):
    aaa = []
    for i in range(len(a)):
        subset = a[i][0]+a[i][1]+a[i][2]
        aaa.append(subset)
    return np.array(aaa)

x0 = cons(x0)
x1 = cons(x1)
x2 = cons(x2)
x3 = cons(x3)
x4 = cons(x4)
x5 = cons(x5)
x6 = cons(x6)

def cons_y(a, type):
    aaa = []
    for i in range(len(a)):
        subset = type
        aaa.append(subset)
    return np.array(aaa)

y0 = cons_y(x0, 0)
y1 = cons_y(x1, 1)
y2 = cons_y(x2, 2)
y3 = cons_y(x3, 3)
y4 = cons_y(x4, 4)
y5 = cons_y(x5, 5)
y6 = cons_y(x6, 6)

plus_x = np.concatenate((x0,x1,x2,x3,x4,x5,x6))
plus_y = np.concatenate((y0,y1,y2,y3,y4,y5,y6))
plus_y = plus_y.reshape(45640, 1)

# print(plus_x[0])
# print(plus_y[0])
# print(plus_x.shape)
# print(plus_y.shape)
# print(x.shape)
# print(y.shape)


x = np.concatenate((x, plus_x))
y = np.concatenate((y, plus_y))

np.save('./dacon/_data/_npy/x_data.npy', arr=x)
np.save('./dacon/_data/_npy/y_data.npy', arr=y)

##############################################################

dataset_test = pd.read_csv(filepath + fname_test)
x_predict = dataset_test.loc[:,['title']]

def clean_text(document):
    document = re.sub("[.,!?'\":;~()]","",document)
    document = re.sub("[^ㄱ-ㅣ가-힣A-Za-z]"," ",document)
    return document

x_predict["title"] = x_predict["title"].apply(lambda x:clean_text(x))

x_predict = np.array(x_predict)

np.save('./dacon/_data/_npy/x_predict.npy', arr=x_predict)