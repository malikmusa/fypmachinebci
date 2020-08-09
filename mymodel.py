import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
#from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
import pickle

import os
for dirname, _, filenames in os.walk('./emotions.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df=pd.read_csv('./emotions.csv')
df.isnull().sum()
encode = ({'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2} )
df_encoded = df.replace(encode)

df_encoded['label'].unique()
#df_encoded.head()
x=df_encoded.drop(["label"]  ,axis=1)

y = df_encoded.loc[:,'label'].values

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
x_train = np.reshape(x_train, (x_train.shape[0],1,x.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0],1,x.shape[1]))

tf.keras.backend.clear_session()

model = Sequential()
model.add(LSTM(64, input_shape=(1,2548),activation="relu",return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs = 100, validation_data= (x_test, y_test))
score, acc = model.evaluate(x_test, y_test)


pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

correct = accuracy_score(expected_classes,predict_classes)

pickle.dump(scaler, open('./model.pkl','wb'))
model = pickle.load(open('./model.pkl','rb'))

