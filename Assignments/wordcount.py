#!/usr/bin/env python
# coding: utf-8

# In[6]:


from pyspark import SparkConf, SparkContext    #import spark libraries
import string as str

data = open("C:\\Users\\Swayanshu\\pg100.txt", "r")   #open the file in directory

word_file = data.read()
data.close()
wordList = word_file.split()

words_count = []

for i in list(str.ascii_lowercase):             #considered the lowecase only
    count = 0
    dict_char_count = {}
    for word in wordList:
        if word[0]== i:
            count += 1
        #endIf
    #endFor
    dict_char_count[i] = count
    words_count.append(dict_char_count)
#endFor
print(words_count)


# In[8]:


words_count


# In[ ]:




