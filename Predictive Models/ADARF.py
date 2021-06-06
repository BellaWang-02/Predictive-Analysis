#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from IPython.display import Image 
from IPython.display import IFrame
import pydot_ng as pydot 
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv('heloc_dataset_v1.csv')
df


# In[3]:


X = df.iloc[:,1:24]
y = []
for i in range(df.shape[0]):
    if df.iloc[i,0] == 'Bad':
        y.append(1)
    else:
        y.append(0)
Y = pd.Series(y)


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


# In[5]:


Y_train = pd.DataFrame(data = Y_train)
Y_train.columns = ['RiskPerformance']
df_train = pd.merge(X_train, Y_train,left_index=True, right_index=True)


# In[6]:


Y_train = Y_train[X_train.ExternalRiskEstimate!=-9]
Y_test  = Y_test[X_test.ExternalRiskEstimate!=-9]
X_train = X_train[X_train.ExternalRiskEstimate!=-9]
X_test  = X_test[X_test.ExternalRiskEstimate!=-9]
Y_train = Y_train.squeeze()


# In[7]:


df_count_missing = pd.concat([(X_train==-7).sum(), (X_train==-8).sum(), (X_train==-9).sum()], axis=1)
df_count_missing.columns = [-7,-8,-9]


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


minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_train)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_train)
col_names_minus_7 = X_train.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = X_train.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_train.columns.values.tolist() + col_names_minus_7 + col_names_minus_8


# In[12]:


X_train_t = pd.DataFrame(arr_X_train_t, columns=column_names)
X_train_t.head()


# In[13]:


X_test_t = pipeline.transform(X_test)
X_test_t = pd.DataFrame(X_test_t, columns=column_names)


# In[14]:


X_train_t_tr, X_train_t_val = train_test_split(X_train_t, test_size=0.25, random_state=1234)
Y_train_t_tr, Y_train_t_val = train_test_split(Y_train, test_size=0.25, random_state=1234)


# In[15]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10, random_state=0, max_depth=1).fit(X_train_t_tr, Y_train_t_tr)

rf.score(X_train_t_tr, Y_train_t_tr)


# In[16]:


test = {}
test[1] = []
test[2]=[]
for i in range(19):
    rf = RandomForestClassifier(n_estimators = (i+1)*10, random_state = 0, max_depth=1).fit(X_train_t_tr, Y_train_t_tr)
    test[1] = np.append(test[1], accuracy_score(Y_train_t_tr, rf.predict(X_train_t_tr)))
    test[2] = np.append(test[2], accuracy_score(Y_train_t_val, rf.predict(X_train_t_val)))

test = pd.DataFrame(data = test, index = (10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190))
test.columns = ['Train accuracy','Validation accuracy']
test.index.name = 'n_estimators'
rf_accuracy = test
rf_accuracy


# In[ ]:





# In[17]:


param_grid = [{'n_estimators':[10,20],
               'max_depth':range(1,10),
               'min_samples_leaf':[10,20],
               'max_leaf_nodes':[2,4,8,16,32]}]

grid_search = GridSearchCV(RandomForestClassifier(random_state=0), 
                           param_grid, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0), 
                           scoring='accuracy')

grid_search.fit(X_train_t_tr, Y_train_t_tr)


# In[18]:


# the best configuration
grid_search.best_params_


# In[19]:


# a model trained using the best configuration on all folds
old_good = RandomForestClassifier(max_depth=8, max_leaf_nodes=32, min_samples_leaf=10,
                       n_estimators=10, random_state=0)

best_RF_model = grid_search.best_estimator_
best_RF_model


# In[20]:


train_accuracy_RF_opt = best_RF_model.score(X_train_t_tr, Y_train_t_tr)
val_accuracy_RF_opt = best_RF_model.score(X_train_t_val, Y_train_t_val)

print('Train accuracy (RF optimized): %.3f'%train_accuracy_RF_opt)
print('Validation accuracy (RF optimized): %.3f'%val_accuracy_RF_opt)


# In[21]:


test_accuracy_RF_opt = best_RF_model.score(X_test_t, Y_test)

print('Train accuracy (RF optimized): %.3f'%test_accuracy_RF_opt)


# In[22]:


# plot feature importance
pd.Series(data=grid_search.best_estimator_.feature_importances_, 
          index=column_names).sort_values().plot.bar(figsize=(15,5), 
                                                     title='Feature importance of best RF model');


# In[ ]:





# In[23]:


test = {}
test[1] = []
test[2]=[]
for i in range(19):
    ab = AdaBoostClassifier(n_estimators = (i+1)*10, random_state = 0, learning_rate=1).fit(X_train_t_tr, Y_train_t_tr)
    test[1] = np.append(test[1], accuracy_score(Y_train_t_tr, ab.predict(X_train_t_tr)))
    test[2] = np.append(test[2], accuracy_score(Y_train_t_val, ab.predict(X_train_t_val)))

test = pd.DataFrame(data = test, index = (10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190))
test.columns = ['Train accuracy','Validation accuracy']
test.index.name = 'n_estimators'
test


# In[24]:


best_ab = AdaBoostClassifier(n_estimators = 40, random_state = 0, learning_rate=1).fit(X_train_t_tr, Y_train_t_tr)
train_accuracy_best_ab = best_ab.score(X_train_t_tr, Y_train_t_tr)
val_accuracy_best_ab = best_ab.score(X_train_t_val, Y_train_t_val)

print('Train accuracy (AB optimized): %.3f'%train_accuracy_best_ab)
print('Validation accuracy (AB optimized): %.3f'%val_accuracy_best_ab)


# In[25]:


test_accuracy_best_ab = best_ab.score(X_test_t, Y_test)
print('Test accuracy (AB optimized): %.3f'%test_accuracy_best_ab)


# In[28]:


filename = 'finalized_model.sav'
with open (filename, 'wb') as fp:
    pickle.dump(best_RF_model, fp)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




