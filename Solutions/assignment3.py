# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:26:09 2019

@author: user
"""

import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.datasets import load_boston
import seaborn as sns
data = pd.read_csv('wine.csv')
#bd=load_boston()
#data=pd.DataFrame(bd.data,columns=bd.feature_names)

cm=data.corr().round(2)
cm = (cm.values)
summ=0
#d = pd.read_csv('Iris2.csv').as_matrix()
x=0
y=0
th=0.2
feat=[]
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if(cm[i][j]!=1):
            x=x+cm[i][j]
            y=y+1
th=x/y
#th=-0.01
k=0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if(cm[i][j]<=th):
            cm[i][j]=0
sns.heatmap(data=cm,annot=True)
print(cm)
while(1):
    weight=[]
    for i in range(len(cm)):
        summ=0        
        for j in range(len(cm[i])):
            summ+=cm[i][j]
        weight.append(summ)
    if(max(weight)==1):
        break;
    for i in range(len(weight)):
        if(weight[i]==max(weight)):
            for f in range(len(cm)):
                for g in range(len(cm)):
                    cm[i][g]=0
                    cm[g][i]=0
            feat.append(i+1)
            feat.sort()
            
        