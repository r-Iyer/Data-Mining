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
interval=300;
a=0
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
i=d[0,0]
while(i<d[len(d)-1,0]):
    i=i+interval
    a=a+1
binn=[]
for i in range(a):
    binn.append([])
j=0
i=0
while(i<len(d)):
    if(d[i,0]<=(d[0,0]+((j+1)*interval))):
        binn[j].append((d[i,0],d[i,1]))
        i=i+1
    else:
        j=j+1

       
    
