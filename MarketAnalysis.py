
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:43:48 2017

@author: paritosh.yadav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Data/Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#using elbow method to find no of clusetr
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmens=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmens.fit(X)
    wcss.append(kmens.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Cluster')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='b',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='g',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='c',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='m',label='sensible')
#plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=100,c='black',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='y',label='Centeroid')
plt.title('Cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score')
plt.legend()
plt.show()

