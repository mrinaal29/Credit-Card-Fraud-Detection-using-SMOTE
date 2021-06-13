#!/usr/bin/env python
# coding: utf-8

# In[1]:


# to install imbalanced learn 
# pip install imbalanced-learn


# In[2]:


#pip install scikit-learn==0.24.2
# to use the latest version of scikit learn as SMOTE
#works on 23.1 version or above in  scikit learn 


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[4]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[5]:


df=pd.read_csv('creditcard.csv')
df.head(10)


# In[6]:


df.shape


# In[7]:


df.info() # no null values 


# In[8]:


df.describe()


# In[9]:



# EDA
df['Class'].value_counts().plot.barh()
plt.title('No. of fraud and non-fraud transactions')
df['Class'].value_counts()


# In[10]:


# this function tells about the probability function of the class varible 
from scipy.stats import norm
import statistics

x_axis = df['Class']
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)
# To calculate normal probability density of the data norm.pdf is used, it refers to the normal probability density 
# function which is a module in scipy library that uses the above probability density function to calculate the value.
plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
plt.show()

Totally one sided dataset  ie imbalanced dataset  . As it has large number of columns and is a big datset so we will use oversampling method
# # In oversampling condition , we will use SMOTE(Synthetic Minority Over-Sampling Technique) which says 
# SMOTE first selects a minority class instance a at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b.

# In[11]:


x=df.drop("Class",axis=1)
y=df['Class']


# In[12]:


y.value_counts().plot.barh() # before using smote


# In[13]:


#split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=.25)


# In[14]:


#using smote
smote = SMOTE(random_state=42)
X,Y  = smote.fit_resample(x_train, y_train)

print("AFTER SMOTE")
Y.value_counts().plot.bar()


# In[15]:


# checking the shape
print(X.shape,Y.shape)
# balanced dataset 


# In[16]:


# Cheking the correlation and removing the least correlated
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), cmap='tab20_r',linewidths=.5,annot=False)


# In[18]:


# using the ensemble method as it gives better accuracy regarding others 
# random forest classifier algorithm
random_clf = RandomForestClassifier()
random_clf.fit(X, Y)

y_pred = random_clf.predict(x_test)

acc_scr = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)

print(f"Accuracy Score of Random Forest is : {acc_scr}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# In[ ]:


# AS our Random forest classifier is gving a accuracy of 99.95 % and the TP and TN are also in a very great number .
#so this model seems to work well on this 


# In[ ]:





# In[ ]:




