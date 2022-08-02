#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.datasets import mnist
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(-1,784)
X_train = X_train/255


# In[2]:


from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42, verbose=True)
rbm.fit(X_train)


# In[3]:


rbm.n_components


# In[4]:


rbm.components_.shape


# In[5]:


rbm.intercept_hidden_.shape


# In[6]:


rbm.intercept_visible_.shape


# In[7]:


X_train[:1].shape


# In[8]:


rbm.transform(X_train[:1])


# In[ ]:




