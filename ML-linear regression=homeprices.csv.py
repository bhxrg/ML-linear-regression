#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model 


# In[2]:


df=pd.read_csv("C:\\Users\\prasanna\\OneDrive\\Documents\\homeprices.csv")
df


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.area,df.price,color='red',marker="+")
plt.xlabel("area")
plt.ylabel("price($)")


# In[26]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[11]:


reg.intercept_


# In[12]:


reg.coef_


# In[25]:


reg.predict([[3300]])


# In[ ]:




