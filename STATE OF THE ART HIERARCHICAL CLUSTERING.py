#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as skm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[17]:


data =nx.read_gml('celegansneural.gml')
print(nx.info(data))


# In[20]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=2, random_state=20)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Celegans dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[21]:


import networkx as nx
data= nx.davis_southern_women_graph()
print(nx.info(data))


# In[25]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=10, n_features=2, centers=4, cluster_std=2, random_state=20)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Davis dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[26]:



data = nx.read_edgelist('facebook_combined.txt')
print(nx.info(data))


# In[27]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=2, random_state=20)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Ego Facebook,combined dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[28]:


data =nx.read_gml('power.gml',label=None) #lable="label"
print(nx.info(data))


# In[29]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=2, random_state=20)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Electric Power Grid dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[30]:


import networkx as nx
data= nx.florentine_families_graph()
print(nx.info(data))


# In[31]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=10, n_features=2, centers=4, cluster_std=2, random_state=20)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Padget Forentine Families dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[32]:


data= nx.read_pajek('USAir97.net')
print(nx.info(data))


# In[33]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=2, random_state=20)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title(' USAIR97 dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[34]:



data=nx.read_gml('karate.gml',label=None) #lable="label"
print(nx.info(data))


# In[35]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
data,y = make_blobs(n_samples=10, n_features=2, centers=4, cluster_std=2, random_state=15)
plt.figure(2)
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Zakary Karate Club dendrogram')

# create clusters linkage="average", affinity=metric , linkage = 'ward' affinity = 'euclidean'
hc = AgglomerativeClustering(n_clusters=4, linkage="average", affinity='euclidean')

# save clusters for chart
y_hc = hc.fit_predict(data,y)

plt.figure(3)

# create scatter plot
plt.scatter(data[y==0,0], data[y==0,1], c='red', s=50)
plt.scatter(data[y==1, 0], data[y==1, 1], c='black', s=50)
plt.scatter(data[y==2, 0], data[y==2, 1], c='blue', s=50)
plt.scatter(data[y==3, 0], data[y==3, 1], c='cyan', s=50)

plt.xlim(-15,15)
plt.ylim(-15,15)


plt.scatter(data[y_hc ==0,0], data[y_hc == 0,1], s=10, c='red')
plt.scatter(data[y_hc==1,0], data[y_hc == 1,1], s=10, c='black')
plt.scatter(data[y_hc ==2,0], data[y_hc == 2,1], s=10, c='blue')
plt.scatter(data[y_hc ==3,0], data[y_hc == 3,1], s=10, c='cyan')
for ii in range(4):
        print(ii)
        i0=y_hc==ii
        counts = np.bincount(y[i0])
        valCountAtorgLbl = (np.argmax(counts))
        accuracy0Tp=100*np.max(counts)/y[y==valCountAtorgLbl].shape[0]
        accuracy0Fp = 100 * np.min(counts) / y[y ==valCountAtorgLbl].shape[0]

print([accuracy0Tp,accuracy0Fp])
plt.show()


# In[ ]:




