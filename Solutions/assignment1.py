# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:38:36 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:19:07 2019

@author: user
"""
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
def euclid(a,b):
    dist=math.sqrt(((a[0]-b[0])**2)+((a[1]-b[1])**2)+((a[2]-b[2])**2)+((a[3]-b[3])**2))
    return dist
def avg(a):
    sum=0
    for i in range(len(a)):
        sum+=a[i]
    return (sum/len(a))  
def jaccard(a,b):
   # print("XXXXXXXXXXXXXXXXXXXXXX")
    return ((len(set(a)&set(b)))/len((set(a)|set(b))))
    
    
#d = np.genfromtxt('iris.csv',delimiter=',', missing_values=0,skip_header=1, dtype=float)
d = pd.read_csv('iris.csv').as_matrix()

#only first 3 columns are considered
#fig = plt.figure(figsize=(10,8))
#ax = Axes3D(fig)
#ax.scatter(d[:, 0], d[:, 1], d[:, 2])
#fig.savefig('kmeans_plot/data.jpg', bbox_inches='tight')
for i in range(len(d[0])):
    ma=np.max(d[:,i])
    mi=np.min(d[:,i])
    for j in range(len(d)):
        d[j][i]=(d[j][i]-mi)/(ma-mi)
#print(d)
d=np.array(d)
eucl=[]
for i in range(len(d)):
    eucl.append([])
for i in range(len(d)):
    for j in range(len(d)):
        eucl[i].append(euclid(d[i],d[j]))
#print(eucl)
eucl=np.array(eucl)
clusters=[]
for i in range(len(eucl)):
    a=[]
    a.append(i)
    for j in range(len(eucl[i])):
        if(eucl[i][j]<avg(eucl[i]) and i!=j):
            a.append(j)
    clusters.append(a)

    
p=set()
for i in range(len(clusters)):
    for j in range(len(clusters)):
        if(set(clusters[i]).issubset(set(clusters[j])) and i!=j):
            p.add(i)
            
p=list(p)
p.sort(reverse=True)            
for i in p:
    clusters.pop(i)
while(True):
    simil=[]
    for i in range(len(clusters)):
        simil.append([])
    #jaccard(clusters[0],clusters[1])
                
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            simil[i].append(jaccard(clusters[i],clusters[j]))
    for i in range(len(simil)):
        simil[i][i]=0
            
    simil=np.array(simil)    
    maxx=np.max(simil)
    r=0
    c=0
    for i in range(len(simil)):
        for j in range(len(simil)): 
            if(simil[i][j]==maxx):
                r=i
                c=j
    clusters[r]=list(set(clusters[c])|set(clusters[r]))
    clusters.remove(clusters[c])
    if(len(clusters)<=3  ):
        break


med1=[]
med2=[]
med3=[]
med4=[]
centroid=[]
for i in range(len(clusters)):
    centroid.append([])
for i in range(len(clusters)):
    for j in range(len(clusters[i])):
        med1.append(d[clusters[i][j]][0])
        med2.append(d[clusters[i][j]][1])
        med3.append(d[clusters[i][j]][2])
        med4.append(d[clusters[i][j]][3])        
    centroid[i].append(np.median(med1))
    centroid[i].append(np.median(med2))
    centroid[i].append(np.median(med3))
    centroid[i].append(np.median(med4))
table=[]
u=0
for i in range(len(d)):
    table.append([])
for i in range(len(d)):
    if((i in clusters[0])   and(i in clusters[1]) and(i in clusters[2])):
        a=euclid(d[i],centroid[0])
        b=euclid(d[i],centroid[1])
        c=euclid(d[i],centroid[2])
        table[u].append(a/(a+b+c))
        table[u].append(b/(a+b+c))
        table[u].append(c/(a+b+c))
        u+=1
    elif((i in clusters[0])   and(i in clusters[1])):
        a=euclid(d[i],centroid[0])
        b=euclid(d[i],centroid[1])
        table[u].append(a/(a+b))
        table[u].append(b/(a+b))
        table[u].append(1)
        u+=1
    elif((i in clusters[1]) and(i in clusters[2])):
        b=euclid(d[i],centroid[1])
        c=euclid(d[i],centroid[2])
        table[u].append(b/(b+c))
        table[u].append(c/(b+c))
        table[u].append(1)
        u+=1
    elif((i in clusters[0])and(i in clusters[2])):
        a=euclid(d[i],centroid[0])
        c=euclid(d[i],centroid[2])
        table[u].append(a/(a+c))
        table[u].append(c/(a+c))
        table[u].append(1)
        u+=1
    elif((i in clusters[0])):
        table[u].append(0)
        table[u].append(1)
        table[u].append(1)
        u+=1
    elif((i in clusters[1])):
        table[u].append(1)
        table[u].append(0)
        table[u].append(1)
        u+=1
    elif((i in clusters[2])):
        table[u].append(1)
        table[u].append(1)
        table[u].append(0)
        u+=1
    else:
        table[u].append(1)
        table[u].append(1)
        table[u].append(1)
        u+=1
table=np.array(table)

lm={0:'r',1:'g',2:'b',3:'y',4:'c'}
for i in range(0,len(clusters)):
    diag=[d[x] for x in clusters[i]]
    diag=np.array(diag)
    plt.scatter(diag[:,0],diag[:,1],c=lm[i])
centroid=np.array(centroid)
plt.scatter(centroid[:,0],centroid[:,1],marker='^',c='y',s=500)

#for a in range(2,6):
#    
#    kmeans = KMeans(n_clusters=a, n_init=20, max_iter=1000)
#
#    kmeans = kmeans.fit(d)
#
#    labels = kmeans.predict(d)
#
#    print(labels)
#
#    label_map = {0 : 'm',
#                 1 : 'b',
#                 2 : 'g',
#                 3 : 'c',
#                 4 : 'y',
#                }
#
#    label_color = [label_map[l] for l in labels]
#
#
#    C = kmeans.cluster_centers_
#    fig = plt.figure(figsize=(10,8))
#    ax = Axes3D(fig)
#    ax.scatter(d[:, 0], d[:, 1], d[:, 2],c=label_color)
#    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='^', c='r', s=300)
#    fig.savefig('kmeans_plot/kmeans(n_c='+str(a)+').jpg', bbox_inches='tight')
    
#kmeans.labels_.astype(float)
