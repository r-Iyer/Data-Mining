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
a=(int)(math.ceil(d[len(d)-1,0]/1000)-math.floor(d[0,0]/1000))

if a==3 or a==6 or a==7 or a==9:
    it=3
if a==2 or a==4 or a==8:
    it=4
if a==1 or a==5 or a==7 or a==10:
    it=5
binn=[]
for i in range(it+1):
    binn.append([])
x=(int)(len(d)/it)
for i in range(it):
    for j in range(x):
        binn[i].append((d[i,0],d[i,1]))
for y in range((x)*it,len(d)):
    binn[i+1].append((d[y,0],d[y,1]))
    