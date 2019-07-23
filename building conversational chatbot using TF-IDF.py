#!/usr/bin/env python
# coding: utf-8

# In[1]:


#building a conversational chatbot for store's answer to questions with TF=IDF
# Reference: "Python Deep Learning Projects", M. Lamons, R. Kumar, A. Nagaraja


# In[2]:


#step 1: prepare the dataset and preprocessing


# In[3]:


import pandas as pd
import numpy as np
import operator ,os
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


filepath='sample_data.csv'
csv_reader =pd.read_csv(filepath)
print(csv_reader)


# In[5]:


question_list = csv_reader[csv_reader.columns[0]].values.tolist()
answers_list  = csv_reader[csv_reader.columns[1]].values.tolist()


# In[6]:


print(question_list)


# In[7]:


query ='can I get an Americano, btw how much it will cost ?'


# In[8]:


# creating the vector
vectorizer = TfidfVectorizer(min_df=0, ngram_range=(2, 4), strip_accents='unicode',norm='l2' , encoding='ISO-8859-1')
print(vectorizer)


# In[9]:


#step 2: train the model on the questions


# In[10]:


X_train = vectorizer.fit_transform(np.array([''.join(que) for que in question_list]))
print(X_train)


# In[11]:


# step 3: transform the query to chatbot


# In[12]:


X_query=vectorizer.transform([query])
print(X_query)


# In[13]:


#step 4: computing similarity score for the query


# In[14]:


XX_similarity=np.dot(X_train.todense(), X_query.transpose().todense())
XX_sim_scores= np.array(XX_similarity).flatten().tolist()
print(XX_sim_scores)


# In[15]:


#step 5; ranking results


# In[16]:


dict_sim= dict(enumerate(XX_sim_scores))
sorted_dict_sim = sorted(dict_sim.items(), key=operator.itemgetter(1), reverse =True)


# In[17]:


# step 6: retrieve the answer result


# In[18]:


# checking the index with the most similar question and the response with the index
if sorted_dict_sim[0][1]==0:
    print("Sorry I have no answer, please try asking again in a nicer way :)")
    resp = "Sorry I have no answer, please try asking again in a nicer way :)"
elif sorted_dict_sim[0][1]>0:
    print (answers_list [sorted_dict_sim[0][0]])        
    resp = answers_list [sorted_dict_sim[0][0]]


# In[19]:


query ='do you have fruits ?'
X_query=vectorizer.transform([query])
XX_similarity=np.dot(X_train.todense(), X_query.transpose().todense())
XX_sim_scores= np.array(XX_similarity).flatten().tolist()
dict_sim= dict(enumerate(XX_sim_scores))
sorted_dict_sim = sorted(dict_sim.items(), key=operator.itemgetter(1), reverse =True)
if sorted_dict_sim[0][1]==0:
    print("Sorry I have no answer, please try asking again in a nicer way :)")
    resp = "Sorry I have no answer, please try asking again in a nicer way :)"
elif sorted_dict_sim[0][1]>0:
    print (answers_list [sorted_dict_sim[0][0]])        
    resp = answers_list [sorted_dict_sim[0][0]]


# In[ ]:




