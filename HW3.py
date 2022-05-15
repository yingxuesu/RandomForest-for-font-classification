#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:44:51 2021

@author: yingxue
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###load the dataset
data1=pd.read_csv('/home/yingxue/Desktop/AGENCY.csv')
Font1=data1.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

data2=pd.read_csv('/home/yingxue/Desktop/BAITI.csv')
Font2=data2.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

data3=pd.read_csv('/home/yingxue/Desktop/CAARD.csv')
Font3=data3.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

data4=pd.read_csv('/home/yingxue/Desktop/BANKGOTHIC.csv')
Font4=data4.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

CL1=Font1.loc[(Font1['strength'] == 0.4) & (Font1['italic'] == 0)]
CL2=Font2.loc[(Font2['strength'] == 0.4) & (Font2['italic'] == 0)]
CL3=Font3.loc[(Font3['strength'] == 0.4) & (Font3['italic'] == 0)]
CL4=Font4.loc[(Font4['strength'] == 0.4) & (Font4['italic'] == 0)]

DATA=pd.concat([CL1,CL2,CL3,CL4])
DATA=DATA.drop(columns=['font','strength','italic'])
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(DATA)
SDATA = pd.DataFrame(scaler.transform(DATA))

true=[1]*251+[2]*412+[3]*424+[4]*560

from sklearn.model_selection import train_test_split
###split training and test set on CLi
trainCL1,testCL1,trainY1,testY1=train_test_split(CL1.drop(columns=['font','strength','italic']),[1]*251,test_size=0.2)
trainCL2,testCL2,trainY2,testY2=train_test_split(CL2.drop(columns=['font','strength','italic']),[2]*412,test_size=0.2)
trainCL3,testCL3,trainY3,testY3=train_test_split(CL3.drop(columns=['font','strength','italic']),[3]*424,test_size=0.2)
trainCL4,testCL4,trainY4,testY4=train_test_split(CL4.drop(columns=['font','strength','italic']),[4]*560,test_size=0.2)

### Generate the whole training set Train and the whole test set Test
### and standarize the dataset
Train = pd.concat([trainCL1,trainCL2,trainCL3,trainCL4])
scaler = preprocessing.StandardScaler().fit(Train)
Train = pd.DataFrame(scaler.transform(Train))
TrainY=pd.DataFrame(trainY1+trainY2+trainY3+trainY4)
Test = pd.concat([testCL1,testCL2,testCL3,testCL4])
scaler = preprocessing.StandardScaler().fit(Test)
Test = pd.DataFrame(scaler.transform(Test))
TestY=pd.DataFrame(testY1+testY2+testY3+testY4)

### SMOTE function
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, Y_train = sm.fit_resample(Train,TrainY)
X_test, Y_test = sm.fit_resample(Test,TestY)

### compute the correlation matrix      
CORR=X_train.corr()
###Eigenvalues Eigen vector of CORR
from numpy import linalg as LA     
l, w = LA.eig(CORR) #l is eigenvalues, w is the eigen vector

L=np.sort(l)[::-1] #sort the eigenvalues in decreasing order
idx = np.argsort(-l) # index of sorted eigenvalues
plt.plot(L)
plt.title('eigenvalues')

###find r such that PEV(r)>90%
for r in range(400):
    if sum(L[0:r])>0.9*400:
        print(r)

W=w[:,idx[0:46]] #find eigenvectors that corresponding to the largest r eigenvalues

### Compute the training set and test set after PCA analysis
NTrain=pd.DataFrame(np.matmul(X_train.to_numpy(),W))
NTest=pd.DataFrame(np.matmul(X_test.to_numpy(),W))

###Random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model = RandomForestClassifier(n_estimators=100, random_state=7)
model=model.fit(NTrain, Y_train)
y_pred = model.predict(NTrain)
y_pred1=model.predict(NTest)
print("Train Accuracy:",metrics.accuracy_score(Y_train, y_pred))
print("Test Accuracy:",metrics.accuracy_score(Y_test, y_pred1))
model.feature_importances_
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train, y_pred)
confusion_matrix(Y_test, y_pred1)

model = RandomForestClassifier(n_estimators=200, random_state=7)
model=model.fit(NTrain, Y_train)
y_pred = model.predict(NTrain)
y_pred1=model.predict(NTest)
print("Train Accuracy:",metrics.accuracy_score(Y_train, y_pred))
print("Test Accuracy:",metrics.accuracy_score(Y_test, y_pred1))
im=model.feature_importances_
### plot the importance of features
plt.plot(model.feature_importances_)
### plot the eigenvalues vs. the importance of features
plt.scatter(L[0:10],im[0:10])
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train, y_pred)
confusion_matrix(Y_test, y_pred1)

model = RandomForestClassifier(n_estimators=300, random_state=7)
model=model.fit(NTrain, Y_train)
y_pred = model.predict(NTrain)
y_pred1=model.predict(NTest)
print("Train Accuracy:",metrics.accuracy_score(Y_train, y_pred))
print("Test Accuracy:",metrics.accuracy_score(Y_test, y_pred1))
model.feature_importances_
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train, y_pred)
confusion_matrix(Y_test, y_pred1)


###Classify class 2 and class 4 separately
Train1 = pd.concat([trainCL2,trainCL4])
scaler = preprocessing.StandardScaler().fit(Train1)
Train1 = pd.DataFrame(scaler.transform(Train1))
TrainY1=pd.DataFrame(trainY2+trainY4)
Test1 = pd.concat([testCL2,testCL4])
scaler = preprocessing.StandardScaler().fit(Test1)
Test1 = pd.DataFrame(scaler.transform(Test1))
TestY1=pd.DataFrame(testY2+testY4)

X_train1, Y_train1 = sm.fit_resample(Train1,TrainY1)
X_test1, Y_test1 = sm.fit_resample(Test1,TestY1)
NTrain1=pd.DataFrame(np.matmul(X_train1.to_numpy(),W))
NTest1=pd.DataFrame(np.matmul(X_test1.to_numpy(),W))


model = RandomForestClassifier(n_estimators=200, random_state=7)
model=model.fit(NTrain1, Y_train1)
y_pred = model.predict(NTrain1)
y_pred1=model.predict(NTest1)
print("Train Accuracy:",metrics.accuracy_score(Y_train1, y_pred))
print("Test Accuracy:",metrics.accuracy_score(Y_test1, y_pred1))

### bag of random forests classifier
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train1, y_pred)
confusion_matrix(Y_test1, y_pred1)

from sklearn.ensemble import BaggingClassifier
clf = RandomForestClassifier(n_estimators=200, random_state=7)
bag = BaggingClassifier(clf, n_estimators=10, max_samples=0.7,
                        random_state=1)

bag=bag.fit(NTrain, Y_train)
y_pred = bag.predict(NTest)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
























