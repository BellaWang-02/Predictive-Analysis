#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator

import pickle


# In[14]:


df = pd.read_csv('heloc_dataset_v1.csv')


# In[15]:


X = df.iloc[:,1:24]
y = []
for i in range(df.shape[0]):
    if df.iloc[i,0] == 'Bad':
        y.append(1)
    else:
        y.append(0)
Y = pd.Series(y)


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


# In[17]:


Y_train = pd.DataFrame(data = Y_train)
Y_train.columns = ['RiskPerformance']
df_train = pd.merge(X_train, Y_train,left_index=True, right_index=True)


# In[18]:


Y_train = Y_train[X_train.ExternalRiskEstimate!=-9]
Y_test  = Y_test[X_test.ExternalRiskEstimate!=-9]
X_train = X_train[X_train.ExternalRiskEstimate!=-9]
X_test  = X_test[X_test.ExternalRiskEstimate!=-9]
Y_train = Y_train.squeeze()


# In[19]:


df_count_missing = pd.concat([(X_train==-7).sum(), (X_train==-8).sum(), (X_train==-9).sum()], axis=1)
df_count_missing.columns = [-7,-8,-9]


# In[20]:


from sklearn.impute import MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion


# In[21]:


from sklearn.pipeline import Pipeline
do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
pipeline = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])


# In[22]:


arr_X_train_t = pipeline.fit_transform(X_train)


# In[23]:


minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_train)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_train)
col_names_minus_7 = X_train.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = X_train.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_train.columns.values.tolist() + col_names_minus_7 + col_names_minus_8


# In[24]:


X_train_t = pd.DataFrame(arr_X_train_t, columns=column_names)
X_train_t.head()


# In[25]:


X_test_t = pipeline.transform(X_test)
X_test_t = pd.DataFrame(X_test_t, columns=column_names)


# In[26]:


X_train_t_tr, X_train_t_val = train_test_split(X_train_t, test_size=0.25, random_state=1234)
Y_train_t_tr, Y_train_t_val = train_test_split(Y_train, test_size=0.25, random_state=1234)


# In[27]:


from sklearn import neighbors
clf_knn = neighbors.KNeighborsClassifier().fit(X_train_t_tr, Y_train_t_tr)


# In[28]:


from sklearn.metrics import accuracy_score
print('KNN accuracy: %.3f'%accuracy_score(Y_train_t_val, clf_knn.predict(X_train_t_val)))


# In[29]:


clf_knn = neighbors.KNeighborsClassifier().fit(X_train_t_tr, Y_train_t_tr)


# In[30]:


from sklearn.model_selection import cross_validate
cv_results_knn = cross_validate(neighbors.KNeighborsClassifier(), X_train_t, Y_train, cv=5, return_estimator=True)


# In[31]:


cv_results_knn['test_score']


# In[32]:


print('KNN - CV accuracy score %.3f'%cv_results_knn['test_score'].mean())


# In[33]:


print('KNN - validation accuracy score: %.3f'%accuracy_score(Y_train_t_val, clf_knn.predict(X_train_t_val)))


# In[30]:


from sklearn.model_selection import GridSearchCV
param_grid = [{'n_neighbors':[80, 85, 90, 95, 100],
              'weights':["uniform", "distance"],
              'algorithm':["auto", "ball_tree", "kd_tree", "brute"],
              'leaf_size':[50, 60, 70],
              'p':[1, 2]}]
clf_knn = neighbors.KNeighborsClassifier()
grid_search = GridSearchCV(clf_knn, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_t, Y_train)


# In[31]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[32]:


grid_search.best_params_


# In[33]:


grid_search.best_estimator_


# In[79]:


clf_knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', leaf_size=50, n_neighbors=90, p=1, weights='distance').fit(X_train_t_tr, Y_train_t_tr)


# In[35]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_train_t_val, clf_knn.predict(X_train_t_val))
conf_matrix


# In[36]:


accuracy = ((710+711)/(710+215+341+711))
tn = 712
fp = 213
fn = 342
tp = 710
tpr = ((tp)/(tp+fn))
fpr = ((fp)/(fp+tn))
tnr = ((tn)/(fp+tn))
fnr = ((fn)/(tp+fn))
recall = ((tp)/(tp+fn))
precision = ((tp)/(tp+fp))


# In[37]:


from sklearn import metrics
import matplotlib.pyplot as plt
scores = clf_knn.predict_proba(X_train_t_val)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_train_t_val, scores)
auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(5,5))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right");
#plt.plot([ANSWER_KEY['fpr']], [ANSWER_KEY['recall']], marker="x", markeredgewidth=5, markersize=12);


# In[38]:


from sklearn.model_selection import cross_validate
cv_results_knn = cross_validate(neighbors.KNeighborsClassifier(algorithm='kd_tree', leaf_size=50, n_neighbors=90, p=1, weights='distance'), X_train_t, Y_train, cv=5, return_estimator=True)


# In[39]:


cv_results_knn['test_score']


# In[40]:


print('KNN - CV accuracy score %.3f'%cv_results_knn['test_score'].mean())


# In[41]:


print('KNN - validation accuracy score: %.3f'%accuracy_score(Y_train_t_val, clf_knn.predict(X_train_t_val)))


# In[80]:


best_knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', leaf_size=50, n_neighbors=90, p=1, weights='uniform').fit(X_train_t_tr, Y_train_t_tr)
train_accuracy_best_knn = best_knn.score(X_train_t_tr, Y_train_t_tr)
val_accuracy_best_knn = best_knn.score(X_train_t_val, Y_train_t_val)

print('Train accuracy (KNN optimized): %.3f'%train_accuracy_best_knn)
print('Validation accuracy (KNN optimized): %.3f'%val_accuracy_best_knn)


# In[81]:


test_accuracy_best_knn = best_knn.score(X_test_t, Y_test)
print('Test accuracy (KNN): %.3f'%test_accuracy_best_knn)


# In[86]:


filename = "knn_model.sav"
with open(filename, 'wb') as fp:
    pickle.dump(best_knn,fp)


# In[ ]:




