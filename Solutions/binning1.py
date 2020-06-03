# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:58:49 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:14:04 2019

@author: user
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
X = []
X=d[:,0]
Y = []
Y=d[:,1]
X=np.array(X)
Y=np.array(Y)
bins=5
a=(int)(len(Y)/bins)
def sortSecond(val): 
    return val[1] 
  
# list1 to demonstrate the use of sorting  
# using using second key
print(X)
# sorts the array in ascending according to  
# second element 
data=d[:,0]
data2=[]
data.sort(axis=0)
Y.sort()
binn=[]
for i in range(bins):
    mean=0
    for j in range(i*a,a*i+a):
        mean+=d[j,0]
    mean/=a;
    binn.append(mean)
for i in range(len(binn)):
    for j in range(i*a,a*i+a):
        data2.append((binn[i],d[j,1]))
        
binn=[]
for i in range(bins):
    med=[]
    for j in range(i*a,a*i+a):
        med.append(d[j,0])
    np.array(med)
    binn.append(np.median(med))
data3=[]
for i in range(len(binn)):
    for j in range(i*a,a*i+a):
        data3.append((binn[i],d[j,1]))
        
        
        
data4=[]
for i in range(bins):
    for j in range(i*a,a*i+a):
        if((d[j,0]-d[i*a,0])<(d[a*i+a-1,0]-d[j,0])):
            data4.append((d[i*a,0],d[j,1]))
        else:
            data4.append((d[i*a+a-1,0],d[j,1]))

for i in range(len(binn)):
    for j in range(i*a,a*i+a):
        data3.append((binn[i],d[j,1]))