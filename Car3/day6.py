# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 07:18:06 2019

@author: cttc
"""


import pandas
data=pandas.read_csv(r"C:\Users\cttc\Downloads\iris.data")
data.shape
data.columns=["sepal_length","sepal_width","petal_length","petal_width","class1"]
data.isnull()
data.isnull().sum().sum()

for value in ['sepal_length']:
    print(value,":",sum(data[value]=='?'))       #check '?' is present or not
    
for value in ['sepal_width']:
    print(value,":",sum(data[value]=='?'))
    
for value in ['petal_length']:
    print(value,":",sum(data[value]=='?'))
    
for value in ['petal_width']:
    print(value,":",sum(data[value]=='?'))

ip=data.drop(["class"],axis=1)
op=data["class"]

from sklearn.preprocessing import LabelEncoder         #labelencoder change output (string to int)
le1=LabelEncoder()
data.class1=le1.fit_transform(data.class1)


from sklearn import cluster
km=cluster.KMeans(n_clusters=3)
km.fit(ip)                                              #fit is used for training purpose
k=km.predict(ip)
print(k)  

from sklearn import metrics
print(metrics.confusion_matrix(data['class1'],k))               #this is confusion matrix(here no. of classes is 3 so it comes 3*3 matrix)                       


data['predict']=k
centroids=km.cluster_centers_
print(centroids)                                              #it is the position of centroid (4 input) and 3 class (output)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.scatter(data.sepal_length,data.petal_length)
plt.show()


x=data.sepal_length
y=data.petal_length
plt.figure(figsize=(12,5))
plt.scatter(x,y,c=k,s=50,cmap='copper')
#plt.show
plt.scatter(x=centroids[:,0],y=centroids[:,2],c='red',s=300,alpha=0.8,marker="*")
plt.show()
print(km)



#SBM(Support Vector Machine) most powerfull algo
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

iris=datasets.load_iris()  #given dataset
x=iris.data[:, :2]         #input extracted from the table
y=iris.target            #already label encoded,output

x_min,x_max= x[:,0].min()-1,x[:, 0].max() + 1
y_min,y_max= x[:,1].min()-1,x[:, 1].max() + 1
h= (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]


#C is a parameter of SVM which tune the svm so that we can find segmentize easily
#gamma vary 0 to 1 more gamma vary more curve nature of graph and C vary from 0 to 1000
#fit is for training
#ravel flat the matrix into a array(change 2*2 matrix into 1d array)


C=100.0  #svm regulation parameter
svc = svm.SVC(kernel='linear',C=C,gamma=0.01).fit(x,y)          
Z=svc.predict(X_plot)
Z=Z.reshape(xx.shape)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.8)         #contour function fill the 3 region and display
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1)          #xx yy it store the position of the0 coordinates of Z
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.xlim(xx.min(),xx.max())
plt.title('SVC with linear kernel')




C=1000.0  #svm regulation parameter
svc = svm.SVC(kernel='rbf',C=C,gamma=0.01).fit(x,y)           #rbf is used for linear and polinomial 
Z=svc.predict(X_plot)                                         #rbf is better than anything it is used for all thingh
Z=Z.reshape(xx.shape)


#calculation of the accuracy
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.8)         #contour function fill the 3 region and display
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1)          #xx yy it store the position of the0 coordinates of Z
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.xlim(xx.min(),xx.max())
plt.title('SVC with linear kernel')