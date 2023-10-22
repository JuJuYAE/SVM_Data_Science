#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer["DESCR"])


# In[6]:


df_feat = pd.DataFrame(cancer["data"], columns = cancer["feature_names"])


# In[7]:


df_feat.head()


# In[9]:


cancer["target_names"]


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = df_feat
y = cancer["target"]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[14]:


from sklearn.svm import SVC


# In[15]:


model = SVC()


# In[16]:


model.fit(X_train, y_train)


# In[17]:


predictions = model.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix, classification_report


# In[19]:


print (confusion_matrix(y_test, predictions))


# In[20]:


print(classification_report(y_test, predictions))


# In[21]:


from sklearn.model_selection import GridSearchCV


# In[29]:


param_grid = {"C" : [0.1,1,10,100,1000]}


# In[30]:


grid = GridSearchCV(SVC(gamma ="auto" ), param_grid, verbose = 3)


# In[31]:


grid.fit(X_train, y_train)


# In[32]:


grid.best_params_


# In[33]:


grid_prediction = grid.predict(X_test)


# In[34]:


print(confusion_matrix(y_test, grid_prediction))


# In[35]:


print(classification_report(y_test, grid_prediction))


# In[ ]:




