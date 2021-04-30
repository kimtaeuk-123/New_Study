import numpy as np
import tensorflow as tf

x = np.array([1,2,3])
y = np.array([1,2,3])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=100, batch_size=1)
#(사이즈가 클수록 빠르고 작을 수록 정확하다)
loss = model.evaluate(x,y,batch_size=1)

x_pred = model.predict([4]) #내가 예측하고 싶은 값을 넣는다

print('result : ', x_pred)
