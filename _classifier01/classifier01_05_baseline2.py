import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from xgboost.training import train
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt
import re
import tqdm
import json
import os


fname_sample_submission = './dacon/_data/classifier01/sample_submission.csv'

train = pd.read_csv('./dacon/_data/classifier01/train.csv')
test = pd.read_csv('./dacon/_data/classifier01/test.csv')
sample_submission = pd.read_csv(fname_sample_submission)

train=train[['과제명','label']]
test=test[['과제명']]

'''
# picle

np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

# load_data

train_data = np.load(file_path+fname_train_data)
train_target = np.load(file_path+fname_target)
sample_train = np.load(file_path+fname_sample_train)
sample_test = np.load(file_path+fname_sample_test)


train = pd.DataFrame(sample_train)
test = pd.DataFrame(sample_test)
'''

# 1. re.sub 한글 및 공백을 제외한 문자 제거
# 2. okt 객체를 활용해 형태소 단위로 나눔
# 3. remove_stopwords로 불용어 제거

def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","",text)
    word_text = okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
    return word_review

stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한']
okt = Okt()
clean_train_text = []
clean_test_text = []

# 시간 오래걸림

for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer=lambda x:x, lowercase=False)
train_feature = vectorizer.fit_transform(clean_train_text)
test_feature = vectorizer.transform(clean_test_text)

# train_feature = np.save('./dacon/_data/_npy/_classifier01/train_feature1.npy', arr=train_feature)
# test_feature = np.save('./dacon/_data/_npy/_classifier01/test_feature1 .npy', arr=test_feature)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_feature, train['label'],
            train_size=0.8, shuffle=True, random_state=66        
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train, y_train) 

print(model.score(x_test, y_test))

results = model.predict(test_feature)

sample_submission['label'] = results

sample_submission.to_csv('./dacon/_data/classifier01/baseline.csv', index=False)
