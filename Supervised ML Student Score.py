#!/usr/bin/env python
# coding: utf-8

# ## The Sparks Foundation
# 
# ### Graduate Rotational Internship Programme (GRIP)
# 
# #### `MAY 2023 Batch`
# #### `Candidate Name : Abhishek K Zine`
# 
# ### Task_1 : Supervised ML

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


df = pd.read_csv(r'G:\Data Analyst\TSF\Supervised ML\student_scores - student_scores.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.cov()


# In[8]:


df.corr()


# In[9]:


df.hist(bins=20,figsize=(15,5))
plt.show()


# In[10]:


df.boxplot(figsize=(15,5))
plt.show()


# In[11]:


df.plot(kind='scatter', x='Hours',y='Scores',figsize=(10,5))
plt.show()


# In[12]:


X = df.drop('Scores',axis=1)
y = df['Scores']


# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[14]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[15]:


line = lr.coef_*X+lr.intercept_

plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[16]:


predict_value = lr.predict(X_test)
predict_value


# In[17]:


df_value = pd.DataFrame({'Actual': y_test, 'Predicted': predict_value})  
df_value


# In[18]:


print('Mean Squared Error:', 
      metrics.mean_squared_error(y_test, predict_value))


# In[19]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, predict_value))


# In[20]:


print('R2 Score:', 
      metrics.r2_score(y_test, predict_value))


# In[ ]:


result = float(input('How Many Hours You re Spending to Improve Your Skills:'))
lr.predict([[result]])


# #### `Thank You`
