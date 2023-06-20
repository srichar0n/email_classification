#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#jaiii sri ram


# In[1]:


#email-classification


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import nltk


# In[9]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\SMSSpamCollection",sep='\t',names=['label','message'])


# In[10]:


df.head()


# In[11]:


#tokenization


# In[12]:


import nltk
import re


# In[17]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[18]:


nltk.download('stopwords')


# In[19]:


ps = PorterStemmer()


# In[20]:


df.info()


# In[21]:


corpus = []


# In[23]:


for i in range(len(df)):
    rp = re.sub('[^a-zA-Z]'," ",df['message'][i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if word not in set(stopwords.words("english"))]
    rp = ' '.join(rp)
    corpus.append(rp)


# In[25]:


#vectorization 


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer


# In[27]:


cv = CountVectorizer()


# In[28]:


x = cv.fit_transform(corpus).toarray()


# In[30]:


y = pd.get_dummies(df['label'],drop_first=True)


# In[92]:


y.head()


# In[31]:


from sklearn.model_selection import train_test_split


# In[76]:


x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=9)


# In[77]:


from sklearn.naive_bayes import MultinomialNB


# In[78]:


model = MultinomialNB()


# In[79]:


model.fit(x_train,y_train)


# In[80]:


y_pred_train = model.predict(x_train)


# In[81]:


y_pred_test = model.predict(x_test)


# In[82]:


model.score(x_train,y_train)


# In[83]:


model.score(x_test,y_test)


# In[84]:


#predicting on a new data 


# In[103]:


st = 'hey you won a lottery with cash please click this link to withdraw'


# In[104]:


corpus = []


# In[105]:


rp = re.sub('[^a-zA-Z]'," ",st)
rp = rp.lower()
rp = rp.split()
rp = [ps.stem(word) for word in rp if word not in set(stopwords.words("english"))]
rp = ' '.join(rp)
corpus.append(rp)


# In[106]:


x = cv.transform(corpus).toarray()


# In[107]:


ans = model.predict(x)


# In[108]:


if(ans==0):
    print("NORMAL MAIL")
else:
    print("SPAM MAIL")


# In[110]:


example_2 = 'Hey i am charan how are you buddy'


# In[111]:


corpus = []
rp = re.sub('[^a-zA-Z]'," ",example_2)
rp = rp.lower()
rp = rp.split()
rp = [ps.stem(word) for word in rp if word not in set(stopwords.words("english"))]
rp = ' '.join(rp)
corpus.append(rp)
x = cv.transform(corpus).toarray()
ans = model.predict(x)
if(ans==0):
    print("NORMAL MAIL")
else:
    print("SPAM MAIL")


# In[112]:


#this is how email classification works using naive bayes classification 


# In[ ]:




