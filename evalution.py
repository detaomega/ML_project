import pandas as pd
import numpy as np
import csv

from keras.models import load_model
from sklearn.impute import SimpleImputer


test = pd.read_csv("./data/test.csv")
test_x = test.drop(['id'], axis=1).drop(['product_code'], axis=1)
test_x = test_x.to_numpy()
for i in range(test_x.shape[0]):
    test_x[i][1] = test_x[i][1].split('_')[-1]
    test_x[i][2] = test_x[i][2].split('_')[-1]

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
test_x = test_x.astype(np.float32)
test_imp = imputer.fit(test_x)
test_x = test_imp.transform(test_x)

model = load_model('model.h5')
y_pred = model.predict(x=test_x, batch_size=200)
answer = np.array([])
for pred in y_pred:
    pred = pred.ravel()
    answer = np.append(answer, pred)
submission = pd.read_csv("./dataset/sample_submission.csv")
submission["failure"] = answer
submission.reset_index(drop=True).to_csv("result.csv", index=False)