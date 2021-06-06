#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[2]:


df = pd.read_csv('heloc_dataset_v1.csv')


# In[3]:


X = df.iloc[:,1:24]
y = []
for i in range(df.shape[0]):
    if df.iloc[i,0] == 'Bad':
        y.append(1)
    else:
        y.append(0)
Y = pd.Series(y)


# In[ ]:





# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


# In[ ]:





# In[5]:


Y_train = pd.DataFrame(data = Y_train)
Y_train.columns = ['RiskPerformance']
df_train = pd.merge(X_train, Y_train,left_index=True, right_index=True)


# In[ ]:





# In[6]:


Y_train = Y_train[X_train.ExternalRiskEstimate!=-9]
Y_test  = Y_test[X_test.ExternalRiskEstimate!=-9]
X_train = X_train[X_train.ExternalRiskEstimate!=-9]
X_test  = X_test[X_test.ExternalRiskEstimate!=-9]
Y_train = Y_train.squeeze()


# In[ ]:





# In[7]:


df_count_missing = pd.concat([(X_train==-7).sum(), (X_train==-8).sum(), (X_train==-9).sum()], axis=1)
df_count_missing.columns = [-7,-8,-9]


# In[ ]:





# In[8]:


from sklearn.impute import MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion


# In[9]:


from sklearn.pipeline import Pipeline
do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
pipeline = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])


# In[10]:


arr_X_train_t = pipeline.fit_transform(X_train)


# In[11]:


arr_X_train_t.shape


# # column names of the transformed dataset

# In[12]:


minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_train)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_train)
col_names_minus_7 = X_train.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = X_train.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_train.columns.values.tolist() + col_names_minus_7 + col_names_minus_8


# In[13]:


X_train_t = pd.DataFrame(arr_X_train_t, columns=column_names)
X_train_t.info()


# In[14]:


X_test_t = pipeline.transform(X_test)
X_test_t = pd.DataFrame(X_test_t, columns=column_names)


# # standardizing 

# In[15]:


X_train_ts=pd.DataFrame(X_train_t.iloc[:,:23],columns = column_names[:23])


# In[16]:


mmean= np.array(X_train_ts.mean(axis = 0))
sstd = np.array(X_train_ts.std(axis = 0))
sstd


# In[17]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values1 = scaler.fit_transform(X_train_t.iloc[:,:23]) 
X_train_t1=pd.DataFrame(scaled_values1,columns = column_names[:23])
X_train_t2 = X_train_t.iloc[:,23:]
X_train_t3 = pd.merge(X_train_t1,X_train_t2,left_index = True, right_index = True)


# In[18]:


scaled_values2 = scaler.fit_transform(X_test_t.iloc[:,:23]) 
X_test_t1=pd.DataFrame(scaled_values2,columns = column_names[:23])
X_test_t2 = X_test_t.iloc[:,23:]
X_test_t3 = pd.merge(X_test_t1,X_test_t2,left_index = True, right_index = True)


# In[ ]:





# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from os import path
from sklearn.linear_model import SGDClassifier


# In[20]:


svc = SGDClassifier(max_iter=1000, random_state=0).fit(X_train_t3, Y_train) 


# log_reg = LogisticRegression(max_iter=10000, random_state=0).fit(X_train_t_tr, Y_train_t_tr) # Logistic regression
# svc     = SGDClassifier(max_iter=1000, random_state=0).fit(X_train_t_tr, Y_train_t_tr)       # Linear SVM
# nb      = BernoulliNB().fit(X_train_t_tr, Y_train_t_tr)                                      # Naive Bayes
# lda     = LinearDiscriminantAnalysis().fit(X_train_t_tr, Y_train_t_tr)                       # LDA

# In[21]:


#svc_coefficients = svc.coef_.reshape((svc.coef_.shape[1],))


# In[ ]:





# In[ ]:





# In[22]:


#linear_coefficients = pd.DataFrame({'SVC':pd.Series(svc_coefficients)})
#linear_coefficients.index = column_names


# In[23]:


#linear_coefficients


# In[ ]:





# In[24]:


#from sklearn.model_selection import cross_val_score
#cross_validate(SGDClassifier(max_iter=1000, random_state=0), X_train_t, Y_train, cv=5, return_estimator=True)


# In[ ]:





# In[ ]:


from sklearn import svm
from sklearn.model_selection import cross_val_score # cross validation
X, Y, results =X_train_t3, Y_train, []
for i in np.linspace(0.01,1,20): 
    clf = svm.SVC(kernel='linear',C=i)  
    scores = cross_val_score(clf, X, Y, cv=5)
    results.append([i,scores.mean(), scores.std()])    
df_accuracy = pd.DataFrame(data=results,columns=['C','mean','std']).round(2)
df_accuracy.plot.bar(x='C',y='mean',yerr='std', title='CV Accuracy vs. C', figsize=(10,2));


# In[25]:


from sklearn.model_selection import cross_validate
cv_results_svc_train = cross_validate(SGDClassifier(max_iter=1000, random_state=0), X_train_t3, Y_train, cv=5, return_estimator=True)
cv_results_svc_train


# In[26]:


cv_results_svc_test = cross_validate(SGDClassifier(max_iter=1000, random_state=0), X_test_t, Y_test, cv=5, return_estimator=True)
cv_results_svc_test


# In[27]:


print('svc - CV accuracy score %.3f'%cv_results_svc_train['test_score'].mean())


# In[28]:


print('svc - CV accuracy score %.3f'%cv_results_svc_test['test_score'].mean())


# In[29]:


filename = 'SVC.sav'
with open(filename,'wb')as fp:
    pickle.dump(svc,fp)


# In[30]:


train_accuracy_best_knn = svc.score(X_train_t3, Y_train)
train_accuracy_best_knn


# In[ ]:





# In[ ]:




