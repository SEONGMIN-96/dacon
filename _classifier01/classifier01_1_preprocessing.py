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

'''
print(train_data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 174304 entries, 0 to 174303
Data columns (total 13 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   index      174304 non-null  int64
 1   제출년도       174304 non-null  int64
 2   사업명        174304 non-null  object
 3   사업_부처명     174304 non-null  object
 4   계속과제여부     174304 non-null  object
 5   내역사업명      174304 non-null  object
 6   과제명        174304 non-null  object
 7   요약문_연구목표   171302 non-null  object
 8   요약문_연구내용   171303 non-null  object
 9   요약문_기대효과   171253 non-null  object
 10  요약문_한글키워드  171276 non-null  object
 11  요약문_영문키워드  171217 non-null  object
 12  label      174304 non-null  int64
dtypes: int64(3), object(10)
memory usage: 17.3+ MB
'''

train=train[['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과', '요약문_한글키워드', 'label']]
test=test[['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과', '요약문_한글키워드']]

def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","",text)
    word_text = okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
    return word_review

stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한']
okt = Okt()

clean_train_text0 = []
clean_test_text0 = []

clean_train_text1 = []
clean_test_text1 = []

clean_train_text2 = []
clean_test_text2 = []

clean_train_text3 = []
clean_test_text3 = []

clean_train_text4 = []
clean_test_text4 = []


for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text0.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text0.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text0.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text0.append([])

for text in tqdm.tqdm(train['요약문_연구목표']):
    try:
        clean_train_text1.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text1.append([])

for text in tqdm.tqdm(test['요약문_연구목표']):
    if type(text) == str:
        clean_test_text1.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text1.append([])

for text in tqdm.tqdm(train['요약문_연구내용']):
    try:
        clean_train_text2.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text2.append([])

for text in tqdm.tqdm(test['요약문_연구내용']):
    if type(text) == str:
        clean_test_text2.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text2.append([])

for text in tqdm.tqdm(train['요약문_기대효과']):
    try:
        clean_train_text3.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text3.append([])

for text in tqdm.tqdm(test['요약문_기대효과']):
    if type(text) == str:
        clean_test_text3.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text3.append([])

for text in tqdm.tqdm(train['요약문_한글키워드']):
    try:
        clean_train_text4.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text4.append([])

for text in tqdm.tqdm(test['요약문_한글키워드']):
    if type(text) == str:
        clean_test_text4.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text4.append([])

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer=lambda x:x, lowercase=False)
train_feature0 = vectorizer.fit_transform(clean_train_text0)
test_feature0 = vectorizer.transform(clean_test_text0)

train_feature1 = vectorizer.fit_transform(clean_train_text1)
test_feature1 = vectorizer.transform(clean_test_text1)

train_feature2 = vectorizer.fit_transform(clean_train_text2)
test_feature2 = vectorizer.transform(clean_test_text2)

train_feature3 = vectorizer.fit_transform(clean_train_text3)
test_feature3 = vectorizer.transform(clean_test_text3)

train_feature4 = vectorizer.fit_transform(clean_train_text4)
test_feature4 = vectorizer.transform(clean_test_text4)