#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[56]:


os.chdir('C:\\Users\\HP\\Downloads')


# **#STEPS:**
# **INSURANCE COST DATA -----> DATA ANALYSIS------> DATA PREPROCESSING-------->TRAIN TEST SPLIT------>LINEAR REGRESSION MODEL
# --------------->TRAINED LINEAR REGREESION MODEL**

# **DATA COLLECTION AND ANALYSIS**

# In[57]:


#loading data from csv file to pandas dataframe:
insurance_data=pd.read_csv('insurance.csv')
insurance_data


# In[58]:


#number of rows and columns:
insurance_data.shape


# In[59]:


#getting some information about the dataset
insurance_data.info()


# **Categorical Features:**
# **sex,**
# **smoker,**
# **region**

# In[60]:


#checking for missing values:
insurance_data.isnull().sum()


# **Data has No Missing Value**

# In[61]:


#statistical Measures of data:
insurance_data.describe()


# In[62]:


#distibution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_data[['age']])
plt.title('age distribution')
plt.show()


# In[63]:


#Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance_data)
plt.title('sex distribution')
plt.show()


# In[64]:


insurance_data['sex'].value_counts()


# In[65]:


#distibution of bmi value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_data[['bmi']])
plt.title('bmi distribution')
plt.show()


# Normal BMI range is 18.5 to 24.9
# But above we observe that there are more number of people in 25 to 40 so lot of people sre overweight.so it will affect the insurance cost.

# In[66]:


#children column
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=insurance_data)
plt.title('children countplot')
plt.show()


# In[67]:


insurance_data['children'].value_counts()


# In[68]:


#smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=insurance_data)
plt.title('smoker countplot')
plt.show()


# In[69]:


#region column
plt.figure(figsize=(6,6))
sns.countplot(x='region',data=insurance_data)
plt.title('region countplot')
plt.show()


# In[70]:


#distibution of charges value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_data[['charges']])
plt.title('charges distribution')
plt.show()


# **DATA PREPROCESSING**

# In[71]:


#encoding for categorical features:
insurance_data.replace({'sex':{'male':0,'female':1}},inplace=True)
insurance_data.replace({'smoker':{'yes':0,'no':1}},inplace=True)
insurance_data.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# In[72]:


insurance_data


# In[73]:


#splitting the features and Target:
X=insurance_data.drop(columns='charges',axis=1)
Y=insurance_data[['charges']]


# In[76]:


X


# In[77]:


Y


# In[78]:


#splitting the data into training and testing data:
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[79]:


print(X.shape,X_train.shape,X_test.shape)


# In[80]:


#Linear model
regressor=LinearRegression()


# In[81]:


regressor.fit(X_train,Y_train)


# **MODEL EVALUATION**

# In[82]:


#predicion of training data
training_data_predict=regressor.predict(X_train)


# In[83]:


# R square
r2_train=metrics.r2_score(Y_train,training_data_predict)
print('R square for training data:',r2_train)


# In[84]:


#R square for test data:
test_data_predict=regressor.predict(X_test)
r2_test=metrics.r2_score(Y_test,test_data_predict)
print('R square for testing data:',r2_test)


# In[ ]:


#for good model R square for both training and testing data should be equal approximately.


# In[ ]:





# In[ ]:




