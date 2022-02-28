#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score


# X = music_data.drop(columns=['genre'])
# y = music_data['genre']
# # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)

# model = DecisionTreeClassifier()
# model.fit(X, y)
# prediction = model.predict( X_test)

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21,1]])
predictions
# score = accuracy_score(y_test,prediction)

# accuracy drops on changing the training set


# In[ ]:




