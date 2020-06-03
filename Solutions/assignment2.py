# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:00:09 2019

@author: Ritodeep
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
d = pd.read_csv('andrew.csv').as_matrix()
#boston_dataset = load_boston()



bins=40
a=(int)(len(Y)/bins)
def sortSecond(val): 
    return val[1] 
  
# list1 to demonstrate the use of sorting  
# using using second key
  
# sorts the array in ascending according to  
# second element 
data=d[:,0]
data2=[]
data.sort(axis=0)




X = []
for i in range(len(d)):
    X.append([])
for i in range(len(d)):
    X[i]=(d[i,0])
Y = []
for i in range(len(d)):
    Y.append([]) 
for i in range(len(d)):
    Y[i]=(d[i,1])
X=np.array(X)
Y=np.array(Y)
X=pd.DataFrame(X);
#features = ['B']
#target = boston['MEDV']
#
#for i, col in enumerate(features):
#    plt.subplot(1, len(features) , i+1)
#    x = boston[col]
#    y = target
#    plt.scatter(x, y, marker='o')
#    plt.title(col)
#    plt.xlabel(col)
#    plt.ylabel('MEDV')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
#X_test = np.array(X_test)
#Y_test = np.array(Y_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

Y_test_predict = lin_model.predict(X_test)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,lin_model.predict(X_train),color='blue')
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,lin_model.predict(X_test),color='blue')
plt.show()

error=[]
for i in range(0,len(Y_test)):
    error.append(abs((Y_test[i]-Y_test_predict[i])/Y_test[i]))
error=np.array(error)
error=np.mean(error)