#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from os import path
from sklearn.metrics import accuracy_score

       


# In[2]:


df = pd.read_csv('heloc_dataset_v1.csv')


# In[3]:


df = df[df['ExternalRiskEstimate']!=-9]


# In[4]:


X = df.iloc[:,1:]
Y = (df.iloc[:,0]=='Bad').astype(int)


# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 1234)


# In[6]:


X_train = X_train[X_train['ExternalRiskEstimate']!=-9]
X_test  = X_test[X_test['ExternalRiskEstimate']!=-9]
Y_train = Y_train.loc[X_train.index.tolist()]
Y_test  = Y_test.loc[X_test.index.tolist()]


# In[7]:


do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
 
pipeline = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])

arr_X_train_t = pipeline.fit_transform(X_train)


# In[8]:


minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_train)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_train) 
col_names_minus_7 = X_train.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = X_train.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_train.columns.values.tolist() + col_names_minus_7 + col_names_minus_8


# In[9]:


X_train_t = pd.DataFrame(arr_X_train_t, columns = column_names)


# In[10]:


arr_X_test_t = pipeline.transform(X_test)

minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_test)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_test) 
col_names_minus_7 = X_test.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = X_test.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_test.columns.values.tolist() + col_names_minus_7 + col_names_minus_8

X_test_t = pd.DataFrame(arr_X_test_t , columns=column_names)


# # Standardized Data

# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values1 = scaler.fit_transform(X_train_t.iloc[:,:23]) 
X_train_t1=pd.DataFrame(scaled_values1,columns = column_names[:23])
X_train_t2 = X_train_t.iloc[:,23:]
X_train_t3 = pd.merge(X_train_t1,X_train_t2,left_index = True, right_index = True)
X_train_t = X_train_t3 

scaled_values2 = scaler.fit_transform(X_test_t.iloc[:,:23]) 
X_test_t1=pd.DataFrame(scaled_values2,columns = column_names[:23])
X_test_t2 = X_test_t.iloc[:,23:]
X_test_t3 = pd.merge(X_test_t1,X_test_t2,left_index = True, right_index = True)
X_test_t = X_test_t3


# In[12]:


X_train_t_tr, X_train_t_val, Y_train_t_tr, Y_train_t_val = train_test_split(X_train_t, Y_train, test_size=0.25, random_state=1234)


# # Linear_models

# In[13]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=10000, random_state=0).fit(X_train_t_tr, Y_train_t_tr) # Logistic regression


# # Accuracy_score

# In[14]:


print('Train accuracy')
print('Log. Reg. accuracy: %.3f'%log_reg.score(X_train_t_tr, Y_train_t_tr))


print('\nValidation accuracy')
print('Log. Reg. accuracy: %.3f'%log_reg.score(X_train_t_val, Y_train_t_val))


# # Check Coefficients

# In[15]:


log_reg_coefficients =np.append(log_reg.coef_[0],log_reg.intercept_)
log_reg_coefficients


# # Model without smallest coefficients

# In[16]:


value=pd.Series(log_reg_coefficients)
value = value[value.abs() < 0.018]
name = value.index.tolist()
name


# In[17]:


X_train_t_new=X_train_t.copy().drop("NumInstallTradesWBalance", axis=1)
X_train_t_tr_new=X_train_t_tr.copy().drop("NumInstallTradesWBalance", axis=1)
X_train_t_val_new=X_train_t_val.copy().drop("NumInstallTradesWBalance", axis=1)
Y_train_new=Y_train
Y_train_t_tr_new=Y_train_t_tr
Y_train_t_val_new=Y_train_t_val


# In[18]:


log_reg_new = LogisticRegression(max_iter=10000, random_state=0).fit(X_train_t_tr_new, Y_train_t_tr_new)


# In[19]:


print('Train accuracy')
print('Log. Reg. accuracy: %.3f'%log_reg_new.score(X_train_t_tr_new, Y_train_t_tr_new))

print('\nValidation accuracy')
print('Log. Reg. accuracy: %.3f'%log_reg_new.score(X_train_t_val_new, Y_train_t_val_new))
 


# # Cross validation 

# In[20]:


from sklearn.model_selection import cross_validate
log_reg = cross_validate(LogisticRegression(max_iter=10000), X_train_t, Y_train, cv=5, return_estimator=True)

print('Logistic regresion - CV accuracy score %.3f'%log_reg['test_score'].mean()) # this is their average value


# In[21]:


from sklearn.model_selection import cross_validate
log_reg_new = cross_validate(LogisticRegression(max_iter=10000), X_train_t_new, Y_train_new, cv=5, return_estimator=True)

print('Logistic regresion - CV accuracy score new %.3f'%log_reg_new['test_score'].mean()) # this is their average value


# # Model is not improved, we will regularize the data

# # Regularized models

# In[22]:


# Train regularized models
from sklearn.linear_model import LogisticRegressionCV

exp_lb1, exp_ub1, num_values1 = -2, 2, 30
Cs = np.geomspace(10**exp_lb1, 10**exp_ub1, num_values1)
logistic1 = LogisticRegressionCV(Cs=Cs, cv=5, random_state=0, penalty='l1', solver='liblinear').fit(X_train_t_tr, Y_train_t_tr)
means1 = logistic1.scores_[1].mean(axis=0)
stds1 = logistic1.scores_[1].std(axis=0)
best1 = accuracy_score(Y_train_t_val, logistic1.predict(X_train_t_val))

exp_lb2, exp_ub2, num_values2 = -2, 2, 30
Cs = np.geomspace(10**exp_lb2, 10**exp_ub2, num_values2)
logistic2 = LogisticRegressionCV(Cs=Cs, cv=5, random_state=0, penalty='l2', solver='liblinear').fit(X_train_t_tr, Y_train_t_tr)
means2 = logistic2.scores_[1].mean(axis=0)
stds2 = logistic2.scores_[1].std(axis=0)
best2 = accuracy_score(Y_train_t_val, logistic2.predict(X_train_t_val))

print('Validation accuracy best 1')
print('Log. Reg. accuracy: %.3f'%best1)

print('\nValidation accuracy best 2')
print('Log. Reg. accuracy: %.3f'%best2)


# # We get the best model "logistic1"

# # Test Model

# In[23]:


best1_test = logistic1.score(X_test_t, Y_test)
print('Test accuracy')
print('best1: %.3f'%best1_test)


# # More probability threshold with Confusion Matrix, Recall and Precision

# In[24]:





# In[25]:


from sklearn.metrics import confusion_matrix
pred_proba_df = pd.DataFrame(logistic1.predict_proba(X_test_t))
threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
for i in threshold_list:
    print ('\n******** For i = {} ******'.format(i))
    Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
    test_accuracy = accuracy_score(Y_test.to_numpy().reshape(Y_test.to_numpy().size,1),
                                    Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1))
    print('Our testing accuracy is {}'.format(test_accuracy))

    print(confusion_matrix(Y_test.to_numpy() .reshape(Y_test.to_numpy() .size,1),
                           Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1)))
    tn, fp, fn, tp = confusion_matrix(Y_test.to_numpy().reshape(Y_test.to_numpy().size,1),
                            Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1)).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    tnr = tn/(tn+fp)
    fnr = fn/(tp+fn)
    recall = tpr
    precision = tp/(tp+fp)
    print('recall: %.3f'%recall)
    print('precision: %.3f'%precision)
   


# # Best Model 

# In[26]:



log_reg_best = LogisticRegressionCV(Cs=Cs, cv=5, random_state=0, penalty='l1', solver='liblinear').fit(X_train_t, Y_train)


# In[27]:


import pickle
filename = 'Best_model.sav'
with open (filename, 'wb') as fp:
    pickle.dump(log_reg_best, fp)

