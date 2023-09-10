#!/usr/bin/env python
# coding: utf-8

# In[39]:



import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
G =nx.read_gml('karate.gml',label=None) #lable="label"
print(nx.info(G))


# In[40]:


nx.info(G)


# In[57]:


import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import metrics as skm
from sklearn.tree import DecisionTreeClassifier as dtree 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeClassifier as dtree3
from sklearn import svm as svm_clf_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


X, y = make_classification(
   n_samples=100,
   n_features=1,
   n_classes=2,
   n_clusters_per_class=1,
   flip_y=0.03,
   n_informative=1,
   n_redundant=0,
   n_repeated=0,
)
print(X)
print(y)


# In[59]:


# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  
# printing the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)
  
# printing the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)


# In[60]:


from sklearn import svm
svm_clf_model = svm.SVC(kernel='linear')
svm_clf_model.fit(X_train, y_train)
svm_y_pred = svm_clf_model.predict(X_test)
accuracy= metrics.accuracy_score(y_test, svm_y_pred)
print('Accuracy Score: \n', accuracy_score(y_test, svm_y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, svm_y_pred))
print('Classification Report: \n', classification_report(y_test, svm_y_pred))


# In[61]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy')
dtree.fit(X_train,y_train)
from sklearn import tree
tree.plot_tree(dtree,filled = True)
from sklearn import tree
tree.plot_tree(dtree,filled = True)
pred_train = dtree.predict(X_train)
pred_test = dtree.predict(X_test)
accuracy= metrics.accuracy_score(y_test, pred_test)
accuracy= metrics.accuracy_score(y_train,pred_train)
print(accuracy_score(y_test,pred_test))
print(accuracy_score(y_train,pred_train))


# In[76]:


from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)
dtree = DecisionTreeClassifier(criterion = 'entropy')
dtree.fit(X_train,y_train)
pred_train = dtree.predict(X_train)
pred_test = dtree.predict(X_test)
predictions = dtree.predict(X_test)
accuracy= metrics.accuracy_score(y_test, pred_test)
accuracy= metrics.accuracy_score(y_train,pred_train)
print(accuracy_score(y_train,pred_train))
print(accuracy_score(y_test, pred_test))
print(confusion_matrix(y_test,predictions))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[68]:


from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=10)
dtree3 = DecisionTreeClassifier(max_depth = 3)
dtree3.fit(X_train,y_train)
tree.plot_tree(dtree3,filled = True)
y_pred_training1 = dtree3.predict(X_train)
y_pred_test1 = dtree3.predict(X_test)
predictions = dtree3.predict(X_test)
accuracy= metrics.accuracy_score(y_train,y_pred_training1)
accuracy= metrics.accuracy_score(y_test, y_pred_test1)
print(accuracy_score(y_train,y_pred_training1))
print(accuracy_score(y_test, y_pred_test1))
print(confusion_matrix(y_test,predictions))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[90]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
accuracy= metrics.accuracy_score(y_test, rfc_pred)
print(accuracy_score(y_test, rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# In[91]:


from sklearn.tree import DecisionTreeClassifier as dtree 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeClassifier as dtree3
from sklearn import svm as svm_clf_model
from sklearn.svm import SVC 
pred =[]
import seaborn as sns


# In[92]:


models={'Decision Tree Classifier1':dtree(),'Random Forest Classifier':rfc(),'Decision Tree Classifier':dtree3(),'Support Vector Classifier':SVC()}
pred =[]
print(models.keys())


# In[93]:


from sklearn.metrics import r2_score
for name,algo in models.items():
    model=algo
    model.fit(X_test,y_test)
    predictions = model.predict(X_test)
    accuracy=r2_score(y_test, predictions)
    pred.append(accuracy)
    print(name,accuracy)


# In[94]:


plt.xlabel("Accuracy%", fontdict={'fontweight':'bold','fontsize':15})
plt.ylabel("Algorithms",fontdict={'fontweight':'bold','fontsize':15})
plt.title('Algorithm Accuracy',fontdict={'fontweight':'bold','fontsize':15} )
sns.barplot(y=list(models.keys()),x=pred,linewidth=1.5,orient ='h',edgecolor="0.2")
plt.show()


# In[95]:


from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
dtree3 = DecisionTreeClassifier(max_depth= 3)
from sklearn.model_selection import cross_val_score
y_pred_train = cross_val_score(estimator = dtree3,X = X_train,y=y_train,cv =5)
y_pred_test = cross_val_score(estimator = dtree3,X = X_test,y=y_test,cv =5)


# In[96]:


print(y_pred_train)
print("&&&&&&&&&&&&&&&&&&&&")
print(y_pred_test)


# In[97]:


print(y_pred_train[0])
print("&&&&&&&&&&&&&&&&&&&&")
print(y_pred_test[0])


# In[88]:


print(y_pred_train[1])
print("&&&&&&&&&&&&&&&&&&&&")
print(y_pred_test[1])


# In[ ]:





# In[ ]:




