#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this is vectorization technique using for how to create vectors from texts
# this technique  called Integer Encoding,and there are many other techniques as well

# first i tokenize 
# second i align integers into sequence
# third i apply padding


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras 


# In[3]:


doc=["I remember, I remember",
    "he house where I was born",
    "The little window where the sun",
    "Came peeping in at morn",
    "He never came a wink too soon",
    "Nor brought too long a day",
    "But now, I often wish the night",
    "Had borne my breath away!"
    ]


# In[4]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(oov_token="<nothing>")


# In[5]:


tokenizer.fit_on_texts(doc)


# In[6]:


print(tokenizer.word_index)


# In[7]:


print(tokenizer.word_counts)


# In[8]:


print(tokenizer.document_count)


# In[9]:


sequences=tokenizer.texts_to_sequences(doc)


# In[10]:


sequences


# In[11]:


from keras.utils import pad_sequences


# In[12]:


sequences=pad_sequences(sequences,padding="post")


# In[14]:


print(sequences)


# In[ ]:




