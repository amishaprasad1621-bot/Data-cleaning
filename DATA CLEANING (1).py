#!/usr/bin/env python
# coding: utf-8

# In[61]:


#QUESTION: 1
import pandas as pd
df = pd.read_csv('healthcare_data_cleaning_dataset.csv')
print(df.head())


# In[59]:


missing_count = df.isnull().sum()
print(missing_count)


# In[9]:


missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)


# In[10]:


missing_info = pd.DataFrame({'Missing Values': missing_count, 'Percentage (%)': missing_percentage})
print(missing_info)


# QUESTION: 2

# In[14]:


df['Age'] = df['Age'].fillna(df['Age'].mean())
print(df['Age'])


# REASON: In general population value for age mean provides a good central representative.
# 

# QUESTION:3

# In[15]:


df['Treatment_Cost'] = df['Treatment_Cost'].fillna(df['Treatment_Cost'].median())
print(df['Treatment_Cost'] )


# REASON: Median is strong for outliers and represents the typical cost better when the data is skewed.

# QUESTION:4

# In[17]:


before = df.shape[0]
print(before)


# In[20]:


df = df.drop_duplicates()
print(df)


# In[22]:


after = df.shape[0]

print(f"Rows before: {before}")
print(f"Rows after: {after}")
print(f"Total duplicates removed: {before - after}")


# QUESTION:5

# In[24]:


invalid_age = df[(df['Age'] < 0) | (df['Age'] > 100)]
print("Invalid Age Records:")
print(invalid_age)


# In[26]:


df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]

print(f"Remaining records after cleaning age: {len(df)}")


# QUESTION: 6

# In[34]:


Q1 = df['Treatment_Cost'].quantile(0.25)
Q3 = df['Treatment_Cost'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[39]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Treatment_Cost'] < lower_bound) | (df['Treatment_Cost'] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")


# QUESTION: 7

# In[43]:


lower_limit = df['Treatment_Cost'].quantile(0.05)
upper_limit = df['Treatment_Cost'].quantile(0.95)
import numpy as np
df['Treatment_Cost'] = np.where(df['Treatment_Cost'] > upper_limit, upper_limit,
                        np.where(df['Treatment_Cost'] < lower_limit, lower_limit, 
                        df['Treatment_Cost']))

print(f"Values capped between: {lower_limit} and {upper_limit}")


# QUESTION: 8

# In[46]:


import numpy as np
import pandas as pd
print("Minimum cost before log:", df['Treatment_Cost'].min())


# In[49]:


df['Log_Treatment_Cost'] = np.log1p(df['Treatment_Cost'])
print(df['Log_Treatment_Cost'])


# In[60]:


print("New column created successfully!")
print(df[['Treatment_Cost', 'Log_Treatment_Cost']].head())


# QUESTION: 9

# In[53]:


df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
print(df['Admission_Date'])


# In[56]:


df = df.sort_values(by='Admission_Date')
print(df)


# In[57]:


df['Admission_Date'] = df['Admission_Date'].ffill().bfill()
print("Time-based missing values handled and data sorted.")

