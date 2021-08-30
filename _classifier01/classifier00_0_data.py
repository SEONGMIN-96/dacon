import numpy as np
import pandas as pd

import re

file_path = './dacon/_data/classifier01/'
fname_train = 'train.csv'
fname_test = 'test.csv'

train_data = pd.read_csv(file_path+fname_train)
test_data = pd.read_csv(file_path+fname_test)

# 현재 데이터의 불균형존재

# print(data.label.value_counts(sort=False)/len(data))

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

# data

sample_train = np.array(train_data[['과제명','label']])
sample_test = np.array(test_data[['과제명']])

train = np.array(train_data[['과제명','요약문_연구목표','요약문_연구내용','요약문_기대효과','label']])
test = np.array(test_data[['과제명','요약문_연구목표','요약문_연구내용','요약문_기대효과']])

np.save('./dacon/_data/_npy/_classifier01/train.npy', arr=train)
np.save('./dacon/_data/_npy/_classifier01/test.npy', arr=test)
np.save('./dacon/_data/_npy/_classifier01/sample_train.npy', arr=sample_train)
np.save('./dacon/_data/_npy/_classifier01/sample_test.npy', arr=sample_test)