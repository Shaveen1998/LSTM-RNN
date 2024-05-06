import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

df = pd.read_csv('dailymilk_production.csv', index_col='Date',parse_dates=True)
df.index.freq = 'D'

train = df.iloc[:156]
test = df.iloc[156:]

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

#define generator
n_input = 10
n_features = 1
generator = TimeseriesGenerator(scaled_train,scaled_train,length =n_input,batch_size=1)

model = Sequential()
model.add(LSTM(100,activation='relu', input_shape=(n_input,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(generator,epochs=120)

model.save('model.h5')