import pandas as pd
import quandl
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

#print(df.head())

#print(df.iloc[:, 4:9])

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#print(df.head())

df['HL'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']
df['OC'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

df = df[['Adj. Close', 'HL', 'OC', 'Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))

#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
#print(df)
#df.dropna(inplace=True)
#print(df)

X = np.array(df.drop(['label'], 1))
#print(X)
#print(y)
X = preprocessing.scale(X)
# print(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# print(X)

df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
#clf = svm.SVR()
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
# print(df['label'])
# print(forecast_set)

#df['Forecast'] = np.nan



