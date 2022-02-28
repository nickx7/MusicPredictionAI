#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction = model.predict( X_test)

score = accuracy_score(y_test,prediction)
score
# accuracy drops on changing the training set

