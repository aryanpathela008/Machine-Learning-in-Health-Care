#!/usr/bin/env python
# coding: utf-8

# # Dataset Link :-
# https://github.com/aryanpathela008/Dataset-Breast-Cancer-.git

# # Machine Learning in Health Care : - Breast Cancer Detection
# 
In this project, I am going to predict whether a tumor is benign(not a breast cancer) or malignant(breast cancer). So in this 
project I have used two classification type Machine Learning Algorithms.
1.Logistic Regression
2.K-Nearest Neighbors(K-NN)
# # Importing the libraries

# In[12]:


import pandas as pd


# # Importing the dataset

# In[13]:


dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# # Splitting the dataset into the Training set and Test set

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Logistic Regression : -

# # Training the Logistic Regression model on the Training set

# In[15]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# # Predicting the Test set results

# In[16]:


y_pred = classifier.predict(X_test)


# # Making the Confusion Matrix

# In[17]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[18]:


(84 + 47) / (84+47+3+3)                # For Accuracy = Number of Correct Predicton / Total number of observations


# # Computing the accuracy with k-Fold Cross Validation

# In[19]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

We get an Accuracy of 96.70% which is very good. Now I am determining with K-Nearest Neighbors will it create a difference in Accuracy or not.
# # K-Nearest Neighbors(K-NN) : -

# # Training the K-NN model on the Training Set

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# # Computing the accuracy with k-Fold Cross Validation

# In[21]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

Alright we get a better accuracy with K-Nearest Neighbors(K-NN) is 97.44%. Thank You.
# In[ ]:




