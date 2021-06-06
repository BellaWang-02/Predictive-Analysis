#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


df = pd.read_csv('heloc_dataset_v1.csv')
df.head()
df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()
#df.iloc[:,11:16]


# In[ ]:


df.hist(bins=50, figsize=(20,10));


# In[ ]:


df = df[df['ExternalRiskEstimate']!=-9]
df.shape


# In[7]:


X = df.iloc[:,1:]
Y = (df.iloc[:,0]=='Bad').astype(int)
Y


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 1234)


# In[9]:


X_train = X_train[X_train['ExternalRiskEstimate']!=-9]
X_test  = X_test[X_test['ExternalRiskEstimate']!=-9]
Y_train = Y_train.loc[X_train.index.tolist()]
Y_test  = Y_test.loc[X_test.index.tolist()]


# In[10]:


do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
 
pipeline = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])

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


X_train_t = pd.DataFrame(arr_X_train_t, columns = column_names)


# In[13]:


arr_X_test_t = pipeline.transform(X_test)

minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_test)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_test) 
col_names_minus_7 = X_test.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = X_test.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_test.columns.values.tolist() + col_names_minus_7 + col_names_minus_8

X_test_t = pd.DataFrame(arr_X_test_t , columns=column_names)


# In[14]:


X_train_t_tr, X_train_t_val, Y_train_t_tr, Y_train_t_val = train_test_split(X_train_t, Y_train, test_size=0.25, random_state=1234)


# # Tree models:

# In[15]:


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


# In[16]:


clf_tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train_t_tr, Y_train_t_tr)


# In[17]:


from sklearn.metrics import accuracy_score
train_accuracy_DT = accuracy_score(Y_train_t_tr, clf_tree.predict(X_train_t_tr))
val_accuracy_DT = accuracy_score(Y_train_t_val, clf_tree.predict(X_train_t_val))


# In[18]:


print('Train accuracy (DT): %.3f'%train_accuracy_DT)
print('Validation accuracy (DT): %.3f'%val_accuracy_DT)


# In[ ]:





# In[19]:


with open("tree_stump.dot", 'w') as f:                   
    # export visualization of model to a .dot file
    tree.export_graphviz(clf_tree, out_file=f, feature_names=column_names, filled=True, label='all')     
    
pydot.graph_from_dot_file('tree_stump.dot').write_pdf('tree_stump.pdf') # convert .dot to .pdf
IFrame('tree_stump.pdf', width=400, height=300)                         # display pdf in jupyter


# In[20]:


d = {}
d[1] = []
d[2] = []
for i in range(1,13):
    clf_tree_new = DecisionTreeClassifier(max_depth=i, random_state=0).fit(X_train_t_tr, Y_train_t_tr)
    d[1] = np.append(d[1], accuracy_score(Y_train_t_tr, clf_tree_new.predict(X_train_t_tr)))
    d[2] = np.append(d[2], accuracy_score(Y_train_t_val, clf_tree_new.predict(X_train_t_val)))

tree_accuracy = pd.DataFrame(data = d,
                            index = np.arange(1, 13, 1).tolist())
tree_accuracy.columns = ['Train accuracy','Validation accuracy']
tree_accuracy.index.name = 'Depth'
tree_accuracy


# In[22]:


# you may ignore this cell
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

if os.name == 'nt':
    ANSWER_KEY_FILE_NAME = "answer_key(win).p"
elif os.name == 'posix':
    ANSWER_KEY_FILE_NAME = "answer_key(unix).p"
else:
    raise Exception('The code was not tested on',os.name)

GENERATE_ANSWER_KEY=False

if GENERATE_ANSWER_KEY: 
    ANSWER_KEY = {} 
else:        
    with open(ANSWER_KEY_FILE_NAME, "rb") as f:
        ANSWER_KEY = pickle.load( f )           


# In[21]:


tree_accuracy.plot()


# In[22]:


param_grid = [{'max_depth':range(1,16),
               'min_samples_leaf':[10,15,20,100],
               'max_leaf_nodes':[2,4,6,15,20,100,10000]}]

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=1), 
                           param_grid, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1), 
                           scoring='accuracy')

grid_search.fit(X_train_t_tr, Y_train_t_tr)


# In[25]:


d = {}
cvres = grid_search.cv_results_

length = len(cvres["params"])
d[1] = []
d[2] = []
d[3] = []
for i in range(length):
    d[1] = np.append(d[1],cvres["params"][i]['max_depth'])
    d[2] = np.append(d[2],cvres["params"][i]['max_leaf_nodes'])
    d[3] = np.append(d[3],cvres["params"][i]['min_samples_leaf'])

d[4] = cvres["mean_test_score"]
tree_accuracy_grid = pd.DataFrame(data = d)
tree_accuracy_grid.columns = ["max_depth","max_leaf_nodes","min_samples_leaf","Accuracy"]
tree_accuracy_grid


# In[26]:


fig, axes = plt.subplots(1,2,figsize=(20,5))
plt.suptitle('Decision trees: CV Accuracy vs. depth \n(Distribution over hyperparameters that share the same tree depth)')
sns.boxplot(x="max_depth", y='Accuracy', data=tree_accuracy_grid, ax=axes[0]);
tree_accuracy.plot(ax=axes[1]);


# In[27]:


grid_search.best_params_


# In[28]:


grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)


# In[29]:


clf_tree = tree.DecisionTreeClassifier(max_depth=6, max_leaf_nodes=15, min_samples_leaf=15,
                       random_state=1).fit(X_train_t_tr, Y_train_t_tr)


# In[30]:


from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[31]:


train_accuracy = accuracy_score(Y_train_t_tr, clf_tree.predict(X_train_t_tr))


# In[32]:


print("Accuracy: %.2f"%(train_accuracy),'\n')


# In[34]:


val_accuracy = accuracy_score(Y_train_t_val, clf_tree.predict(X_train_t_val))


# In[35]:


print("Accuracy: %.2f"%(val_accuracy),'\n')


# In[36]:


test_accuracy_best_tree = clf_tree.score(X_test_t, Y_test)
print('Test accuracy (tree optimized): %.3f'%test_accuracy_best_tree)


# In[ ]:




