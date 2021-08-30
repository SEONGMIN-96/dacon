# 데이터부터 가져옵시다.

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.python.keras.layers import embeddings
from sklearn.model_selection import train_test_split

file_path = './dacon/_data/samsung01/'

# 분자구조

def model_s1(x_train, s1_train, x_test, s1_test, x_predict):
        
        model = Sequential()
        model.add(Embedding(input_dim=12449, output_dim=64, input_length=15))
        model.add(LSTM(32))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        # model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        model.fit(x_train, s1_train, epochs=10, batch_size=2, validation_split=0.2, verbose=1)

        loss, mae = model.evaluate(x_test, s1_test)
        output = model.predict(x_predict)

        print("loss :", loss)
        print("mae :", mae)
        return output


def model_t1(x_train, t1_train, x_test, t1_test, x_predict):
        
        model = Sequential()
        model.add(Embedding(input_dim=12449, output_dim=64, input_length=15))
        model.add(LSTM(32))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        # model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        model.fit(x_train, t1_train, epochs=10, batch_size=2, validation_split=0.2, verbose=1)

        loss, mae = model.evaluate(x_test, t1_test)
        output = model.predict(x_predict)

        print("loss :", loss)
        print("mae :", mae)
        return output


def train():
        train = pd.read_csv(file_path+'train.csv')

        x = train.loc[:,['SMILES']]
        y_s1 = train.loc[:,['S1_energy(eV)']].values             
        y_t1 = train.loc[:,['T1_energy(eV)']].values         
        
        # print(x.shape)          # (30274, 1)
        token = Tokenizer()
        token.fit_on_texts(x['SMILES'])
        # print(token.word_index)

        x = token.texts_to_sequences(x['SMILES'])

        # word_size = len(token.word_index)
        # print(word_size)        # 12449

        x = pad_sequences(x, padding='pre', maxlen=15) # or post
        # print(x.shape)            # (30274, 15)

        return {"y_s1": y_s1, "y_t1": y_t1, "x": x}

def test():
        test = pd.read_csv(file_path+'test.csv')
        test_data = test.loc[:,'SMILES'].values

        return {"x_test": test_data}        

def train_test_splits(x_data, s1_data, t1_data):
        x_train, x_test, s1_train, s1_test, t1_train, t1_test = train_test_split(x_data, s1_data, t1_data,
                                train_size=0.8, shuffle=True, random_state=66)
        return {"x_train": x_train, "s1_train": s1_train, "t1_train": t1_train,
                "x_test": x_test, "s1_test": s1_test, "t1_test": t1_test}


train_data = train()
s1_data = train_data['y_s1']
t1_data = train_data['y_t1']
x_data = train_data['x']

test = test()
x_predict = test['x_test']

tts = train_test_splits(x_data, s1_data, t1_data)
x_train = tts['x_train']
s1_train = tts['s1_train']
t1_train = tts['t1_train']
x_test = tts['x_test']
s1_test = tts['s1_test']
t1_test  = tts['t1_test']

s1_output = model_s1(x_train, s1_train, x_test, s1_test, x_predict)
t1_output = model_s1(x_train, t1_train, x_test, t1_test, x_predict)

st1_gap = s1_output - t1_output