# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:19:07 2019

@author: user
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
#d = np.genfromtxt('iris.csv',delimiter=',', missing_values=0,skip_header=1, dtype=float)
d = pd.read_csv('iris.csv').as_matrix()
print(d)

#only first 3 columns are considered
fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)
ax.scatter(d[:, 0], d[:, 1], d[:, 2])
#fig.savefig('kmeans_plot/data.jpg', bbox_inches='tight')

for a in range(2,6):
    
    kmeans = KMeans(n_clusters=a, n_init=20, max_iter=1000)

    kmeans = kmeans.fit(d)

    labels = kmeans.predict(d)

    print(labels)

    label_map = {0 : 'm',
                 1 : 'b',
                 2 : 'g',
                 3 : 'c',
                 4 : 'y',
                }

    label_color = [label_map[l] for l in labels]


    C = kmeans.cluster_centers_
    fig = plt.figure(figsize=(10,8))
    ax = Axes3D(fig)
    ax.scatter(d[:, 0], d[:, 1], d[:, 2],c=label_color)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='^', c='r', s=300)
#    fig.savefig('kmeans_plot/kmeans(n_c='+str(a)+').jpg', bbox_inches='tight')
    
#kmeans.labels_.astype(float)
