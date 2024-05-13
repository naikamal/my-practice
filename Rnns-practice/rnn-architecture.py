#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# this is the simple architecture of RNN having for four inputs having 31 trainable parameters
# and it consists on 4 nodes


# In[5]:


from keras import Sequential
from keras.layers import Dense,SimpleRNN


# In[6]:


model=Sequential()
model.add(SimpleRNN(3,input_shape=(4,5)))
model.add(Dense(1,activation="sigmoid"))
model.summary()


# In[3]:


print(model.get_weights()[0].shape,
model.get_weights()[1].shape,
model.get_weights()[2].shape,
model.get_weights()[3].shape,
model.get_weights()[4].shape)


# In[4]:


print(model.get_weights()[0])


# In[ ]:




