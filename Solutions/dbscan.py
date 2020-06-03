# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:22:33 2019

@author: user
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances

def dunn(c, distances):
    unique_cluster_distances = np.unique(min_cluster_distances(c, distances))
    max_diameter = max(diameter(c, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter

def min_cluster_distances(c, distances):
    min_distances = np.zeros((max(c) + 1, max(c) + 1))
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            if c[i] != c[ii] and distances[i, ii] > min_distances[c[i], c[ii]]:
                min_distances[c[i], c[ii]] = min_distances[c[ii], c[i]] = distances[i, ii]
    return min_distances

def diameter(c, distances):
    diameters = np.zeros(max(c) + 1)
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            if c[i] != -1 or c[ii] != -1 and c[i] == c[ii] and distances[i, ii] > diameters[c[i]]:
                diameters[c[i]] = distances[i, ii]
    return diameters

data = np.genfromtxt('iris.csv',delimiter=',', missing_values=0,skip_header=1, dtype=float)
scaler = MinMaxScaler()
n_data = scaler.fit_transform(data)
#print(n_data)

db = DBSCAN(eps=0.3, min_samples=5).fit(n_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(n_clusters)
print(len(labels))

s = metrics.silhouette_score(n_data, labels, metric='euclidean')
d = metrics.davies_bouldin_score(n_data, labels)
print(s)
print(d)

dund = dunn(labels, euclidean_distances(n_data))
print(dund)
#dunk = dunn(k, euclidean_distances(x))

label_map = {-1 : 'r',
             0 : 'm',
             1 : 'b',
             2 : 'g',
             3 : 'c',
             4 : 'y',
             }

label_color = [label_map[l] for l in labels]

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)
ax.scatter(n_data[:, 0], n_data[:, 1], n_data[:, 2],c=label_color)