import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,5,8,10,12,14,16,18,20])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113]) #예측할 때 쓸 것도 주자

model = Sequential()
model.add(Dense(50, input_dim = 1, activation='relu')) #linear 보다 relu가 더 성능이 좋다. 여기까지만 알아둬라
model.add(Dense(30,activation='relu')) #linear 보다 relu가 더 성능이 좋다. 여기까지만 알아둬라
model.add(Dense(25,activation='relu')) #linear 보다 relu가 더 성능이 좋다. 여기까지만 알아둬라
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1200, batch_size=10)


loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

result = model.predict([x_predict])
print('result: ', result)