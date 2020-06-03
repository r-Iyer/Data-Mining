# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:51:51 2019

@author: user
"""

import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
import seaborn as sns

def en(a):
    x0 = list(a[:,1]).count(0)
    x1 = list(a[:,1]).count(1)
    x2 = list(a[:,1]).count(2)
    tot = x0 + x1 + x2
    p0 = x0 / tot
    p1 = x1 / tot
    p2 = x2 / tot
    if p0 == 0:
        k0 = 0
    else:
        k0 = math.log(p0,2)
    if p1 == 0:
        k1 = 0
    else:
        k1 = math.log(p1,2)
    if p2 == 0:
        k2 = 0
    else:
        k2 = math.log(p2,2)
    ex = -((p0 * k0) + (p1 * k1) + (p2 * k2))
    return ex

def gain(k,a,x):
    a1 = []
    a2 = []
    for i in range(0,len(a)):
        if a[i,0] <= k:
              a1.append(a[i])
        else:
            a2.append(a[i])
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
#    print(a1)
#    print(a2)
    if a1.size:
        e1 = en(a1)
    else:
        e1 = 0
    if a2.size:
        e2 = en(a2)
    else: 
        e2 = 0
    t = len(a1) + len(a2)
    info = ((len(a1)/t)*e1) + ((len(a2)/t)*e2)
#    print(info)
    g = x - info
    return g
#    print(a)
    
def gg(k):
    a1 = []
    a2 = []
    for i in range(0,len(data)):
        if n[i,0] <= k:
            a1.append(n[i])
        else:
            a2.append(n[i])
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
#    print(a1)
#    print(a2)
    e1 = en(a1)
    e2 = en(a2)
    return a1,a2,e1,e2

data = pd.read_csv('Iris2.csv').as_matrix()
#c_m=data.corr().round(2)
#sns.heatmap(data=c_m,annot=True)
data = data[np.argsort(data[:,0])]
label_encoder = preprocessing.LabelEncoder() 
data[:,4]= label_encoder.fit_transform(data[:,4])
n = data[:,[0,4]]
#print(n)

e = en(n)
print(e)

a = []
for i in range(0,len(n)-1):
    k = (n[i,0] + n[i+1,0]) / 2
    g = gain(k,n,e)
    a.append([k,g])
a = np.asarray(a)
for i in range(0,len(a)):
    if a[i,1] == max(a[:,1]):
        x = a[i,0]
        break
bin3 = []
a1 = []
a2 = []
for i in range(0,len(data)):
    if n[i,0] <= x: 
        a1.append(n[i,0])
    else:
        a2.append(n[i,0])
#a1 = np.asarray(a1)
#a2 = np.asarray(a2)
bin3.append(a1)
bin3.append(a2)
#print(bin3)

a1,a2,ee1,ee2 = gg(x)
#print(a1)
print(ee1)
aa1 = []
for i in range(0,len(a1)-1):
    k = (a1[i,0] + a1[i+1,0]) / 2
    g = gain(k,a1,ee1)
    aa1.append([k,g])
aa1 = np.asarray(aa1)
#print(aa1)
aa2 = []
for i in range(0,len(a2)-1):
    k = (a2[i,0] + a2[i+1,0]) / 2
    g = gain(k,a2,ee2)
    aa2.append([k,g])
aa2 = np.asarray(aa2)
#print(aa2)
no = 4
if no == 3:
    b3 = []
    if max(aa1[:,1])>=max(aa2[:,1]):
        for i in range(0,len(aa1)):
            if aa1[i,1] == max(aa1[:,1]):
                x = aa1[i,0]
                break
        m1 = []
        m2 = []
        for i in range(0,len(a1)):
            if a1[i,0] <= x: 
                m1.append(a1[i,0])
            else:
                m2.append(a1[i,0])
        b3.append(m1)
        b3.append(m2)
        b3.append(list(a2[:,0]))
    else:
        for i in range(0,len(aa2)):
            if aa2[i,1] == max(aa2[:,1]):
                x = aa2[i,0]
                break
        m1 = []
        m2 = []
        for i in range(0,len(a2)):
            if a2[i,0] <= x: 
                m1.append(a2[i,0])
            else:
                m2.append(a2[i,0])
        b3.append(list(a1[:,0]))
        b3.append(m1)
        b3.append(m2)   
        print(b3)
elif no == 4:
    b3 = []
    for i in range(0,len(aa1)):
        if aa1[i,1] == max(aa1[:,1]):
            x = aa1[i,0]
            break
    m1 = []
    m2 = []
    for i in range(0,len(a1)):
        if a1[i,0] <= x: 
            m1.append(a1[i,0])
        else:
            m2.append(a1[i,0])
    b3.append(m1)
    b3.append(m2)
    for i in range(0,len(aa2)):
        if aa2[i,1] == max(aa2[:,1]):
            x = aa2[i,0]
            break
    m1 = []
    m2 = []
    for i in range(0,len(a2)):
        if a2[i,0] <= x: 
            m1.append(a2[i,0])
        else:
            m2.append(a2[i,0])
    b3.append(m1)
    b3.append(m2)   
    print(b3)