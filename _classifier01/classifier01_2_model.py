from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

file_path = './dacon/_data/_npy/_classifier01/'
fname_sample_submission = './dacon/_data/classifier01/sample_submission.csv'

sample_submission = pd.read_csv(fname_sample_submission)
train_feature = np.load(file_path+'train_feature1.npy')
test_feature = np.load(file_path+'test_feature1.npy')

print(len(train_feature))
print(len(test_feature))


x_train, x_test, y_train, y_test = train_test_split(train_feature, train_feature[1],
            train_size=0.8, shuffle=True, random_state=66        
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

results = model.predict(test_feature)

sample_submission['label'] = results

sample_submission.to_csv('baseline.csv', index=False)
