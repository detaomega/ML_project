import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


train = pd.read_csv("./dataset/train.csv")
train_x = train.drop(['id'], axis=1).drop(['product_code'], axis=1).drop(['failure'], axis=1)
train_x = train_x.to_numpy()
train_y = train[['failure']]
train_y = train_y.to_numpy()
print(train_x[0])
for i in range(train_x.shape[0]):
    train_x[i][1] = train_x[i][1].split('_')[-1]
    train_x[i][2] = train_x[i][2].split('_')[-1]
train_x = train_x.astype(np.float32)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
train_imp = imputer.fit(train_x)
train_x = train_imp.transform(train_x)
train_y = train_y.astype(np.float32)


model = Sequential()
model.add(Dense(92, input_dim=23))
# 使用 'relu' 當作 activation function
model.add(Activation('relu'))
# 使用 Dropout 隨機捨棄 30% 的連線
model.add(Dropout(0.3))
# 加入 hidden layer-2 of 256 neurons
model.add(Dense(92))
# 使用 'relu' 當作 activation function
model.add(Activation('relu'))
# 使用 Dropout 隨機捨棄 30% 的連線
model.add(Dropout(0.3))
# 加入 hidden layer-3 of 128 neurons
model.add(Dense(1))
# 使用 'relu' 當作 activation function
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=train_x, y=train_y,  epochs=200, batch_size=200, verbose=2)
model.save('model.h5')

