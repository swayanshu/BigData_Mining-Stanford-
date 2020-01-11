#!/usr/bin/env python
# coding: utf-8

# The notebook contains-
# 
# 
# 1.factors for recommendations given in assignment
# 2. m and n calculation to create a matrix
# 3. SGD based LR_recommendation
# 4. Graph after 40 iteration, time taken 10hours  <b>Learning_rate = 0.01 # 𝜂 
# 5. For other learning rate error calculation I have use GoogleCoLab 
# 6. Execution time for 40 iteration is -  07:65:02 (GoogleCoLab)
# 7. Conclusion:
#     Error value decreased with increasing number of iterations

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import random
import math


# In[2]:


s = pd.read_csv('ratings.csv')


# In[3]:


s.info()


# # Adding all the factors for recommendations

# In[3]:


iterations = 40
k = 20
regularization_factor = 0.1 #𝜆
learning_rate = 0.01
data_path = r'C:\Users\Swayanshu\ratings.csv'


# # m and n for matrix

# In[5]:


import pandas as pd
chunksize = 1
max_movie = 0
max_user = 0
i = 0
for chunk in pd.read_csv('ratings.csv', chunksize=chunksize, header=None):
    if chunk.loc[i,0] > max_movie :
        max_movie = chunk.loc[i,0]
    if chunk.loc[i,1] > max_user:
        max_user = chunk.loc[i,1]
    i = i + 1


# In[6]:


max_user           #n user


# In[7]:


max_movie          #m movie


# In[8]:


#Choose k = 20, λ = 0.1, 
#P and Q to random values in [0,√5/k].


# # SGD recommendation 
# 
# #learning_rate = 0.1 # 𝜂  shows "infinity" for derivative of p so I changed to 0.01

# In[9]:


def latent_factor_recommnder(data_path, regularization_factor, learning_rate, iterations, k):
    P_matrix = np.array([[random.uniform(0,math.sqrt(5/k) ) for x in range(k)] for y in range(max_user+1)])
    Q_matrix = np.array([[random.uniform(0,math.sqrt(5/k) ) for x in range(k)] for y in range(max_movie+1)])
    error = []
    for k in range(0,iterations):
        i=0
        for chunk in pd.read_csv(data_path, chunksize=chunksize, header=None):
            derivative_E = 2*(chunk.loc[i,2] -( Q_matrix[chunk.loc[i,0],:].dot(P_matrix[chunk.loc[i,1],:])))
            #print(i)
            #print(derivative_E)
            temporary_q_vector = Q_matrix[chunk.loc[i,0],:] +                            learning_rate*(                            (derivative_E*P_matrix[chunk.loc[i,1],:]) -                            ((2*regularization_factor) * Q_matrix[chunk.loc[i,0],:])                                         )
           
            
            temporary_p_vector = P_matrix[chunk.loc[i,1],:] +                    learning_rate*(                    (derivative_E*Q_matrix[chunk.loc[i,0],:]) -                    ((2*regularization_factor)*P_matrix[chunk.loc[i,1],:])                                         )
            Q_matrix[chunk.loc[i,0],:] = temporary_q_vector
            P_matrix[chunk.loc[i,1],:] = temporary_p_vector
            i += 1
        
        #Error  calculation
        
        error_val = 0
        i = 0
        for chunk in pd.read_csv(data_path, chunksize=chunksize, header=None):
            error_val +=  (chunk.loc[i,2] - (Q_matrix[chunk.loc[i,0],:].dot(P_matrix[chunk.loc[i,1],:]) )) **2
            i += 1
        for q_rows in Q_matrix:
            error_val += np.sum(q_rows * q_rows)
        for p_rows in P_matrix:
            error_val += np.sum(p_rows * p_rows)

        error.append(error_val)
        print("Iteration " +str(k+1) + ": " +str(error_val))
    return error
                   


# In[10]:


error_list = []


# In[11]:


error_list.append(latent_factor_recommnder(data_path, regularization_factor, learning_rate, iterations, k))


# # For learning_rate = 0.01 #𝜂, 40th error =80622     
# 1. For learning_rate = 0.02 #𝜂, 40th error =84102
# 2. For learning_rate = 0.03 #𝜂, 40th error =89872

# # Graph for learning_rate = 0.01 # 𝜂, executed in GoogleColab

# # X-axis = number of Interation and Y = Errors after execution

# In[1]:


import plotly.graph_objects as go
import numpy as np


x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]                  #All the interations values are here
y = [197283,117034,112147,110101,108758,107602,106428,105152,103759,102278,100757,99235,97743,96301,94926,93630,92420,91299,90266,89317,88449,87655,86615,86329,86075,85847,85642,84121,83690,83293,82926,82588,82274,81984,81715,81464,81231,81014,80812,80622]  #All the errors from calculation 

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y,
    error_y=dict(
        type='constant',
        value=0.1,
        color='black',
        thickness=1.5,
        width=3,
    ),
    error_x=dict(
        type='constant',
        value=0.4,
        color='blue',
        thickness=1.8,
        width=4,
    ),
    marker=dict(color='maroon', size=10)
))
fig.show()


# # Conclusion:
#     
# Inference:
#         
#   1. In 1st iteration the error = 198238.9
#   2. 40th iteration Error = 84102.4
#   3. Linear changes in decrementing values after 4th iteration which is good
#   
# <b>4. With increasing iteration the error value is decreasing

# In[ ]:




