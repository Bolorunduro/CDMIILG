#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import cluster
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


# In[2]:


from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])


# In[3]:


from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])


# In[4]:


normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])


# In[5]:


adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])


# In[6]:


from sklearn.metrics.cluster import adjusted_rand_score
   
labels_true = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels_pred = [0, 0, 2, 2, 3, 3, 2, 2, 3, 3, 2]

adjusted_rand_score(labels_true, labels_pred)


# In[7]:


from sklearn.metrics.cluster import normalized_mutual_info_score
   
labels_true = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels_pred = [0, 0, 2, 2, 3, 3, 2, 2, 3, 3, 2]

normalized_mutual_info_score(labels_true, labels_pred)


# In[8]:


from sklearn.metrics.cluster import normalized_mutual_info_score 
   
labels_true = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]
labels_pred = [0, 0, 2, 2, 3, 3, 2, 3, 3, 4, 3]

normalized_mutual_info_score (labels_true, labels_pred)


# In[9]:


from sklearn.metrics.cluster import adjusted_rand_score
  
labels_true = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]
labels_pred = [0, 0, 2, 2, 3, 3, 2, 3, 3, 4, 3]

adjusted_rand_score(labels_true, labels_pred)


# In[10]:


from sklearn import metrics

score_funcs = [
    ("Rand index", metrics.rand_score),
    ("ARI", metrics.adjusted_rand_score),
    ("MI", metrics.mutual_info_score),
    ("NMI", metrics.normalized_mutual_info_score),
    ("AMI", metrics.adjusted_mutual_info_score),
]


# In[11]:


import numpy as np

rng = np.random.RandomState(0)


def random_labels(n_samples, n_classes):
    return rng.randint(low=0, high=n_classes, size=n_samples)


# In[12]:


def fixed_classes_uniform_labelings_scores(
    score_func, n_samples, n_clusters_range, n_classes, n_runs=5
):
    scores = np.zeros((len(n_clusters_range), n_runs))
    labels_a = random_labels(n_samples=n_samples, n_classes=n_classes)

    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

n_samples = 10000
n_classes = 100
n_clusters_range = np.linspace(2, 1000, 20).astype(int)
plots = []
names = []

sns.color_palette("colorblind")
plt.figure(1)

for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    scores = fixed_classes_uniform_labelings_scores(
        score_func, n_samples, n_clusters_range, n_classes=n_classes
    )
    plots.append(
        plt.errorbar(
            n_clusters_range,
            scores.mean(axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=1,
            marker=marker,
        )[0]
    )
    names.append(score_name)

plt.title(
    "Community samplying for random uniform labelings\n"f" against reference assignment with{n_classes} classes"
)
plt.xlabel(f"Number of communities (Number of samples is fixed to {n_samples})")
plt.ylabel("Score value")
plt.ylim(bottom=-0.05, top=1.05)
plt.legend(plots, names, bbox_to_anchor=(0.5, 0.5))
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

n_samples = 1000
n_classes = 10
n_clusters_range = np.linspace(2, 100, 10).astype(int)
plots = []
names = []

sns.color_palette("colorblind")
plt.figure(1)

for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    scores = fixed_classes_uniform_labelings_scores(
        score_func, n_samples, n_clusters_range, n_classes=n_classes
    )
    plots.append(
        plt.errorbar(
            n_clusters_range,
            scores.mean(axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=1,
            marker=marker,
        )[0]
    )
    names.append(score_name)

plt.title(
    "Commuity measures for random uniform labeling\n"
    f"against reference assignment with {n_classes} classes"
)
plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
plt.ylabel("Score value")
plt.ylim(bottom=-0.05, top=1.05)
plt.legend(plots, names, bbox_to_anchor=(0.5, 0.5))
plt.show()


# In[15]:


def uniform_labelings_scores(score_func, n_samples, n_clusters_range, n_runs=5):
    scores = np.zeros((len(n_clusters_range), n_runs))

    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_a = random_labels(n_samples=n_samples, n_classes=n_clusters)
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


# In[16]:


n_samples = 100
n_clusters_range = np.linspace(2, n_samples, 10).astype(int)

plt.figure(2)

plots = []
names = []

for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    plots.append(
        plt.errorbar(
            n_clusters_range,
            np.median(scores, axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=2,
            marker=marker,
        )[0]
    )
    names.append(score_name)

plt.title(
    "Community evaluations for 2 random uniform labelings\nwith equal number of clusters"
)
plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
plt.ylabel("Score value")
plt.legend(plots, names)
plt.ylim(bottom=-0.05, top=1.05)
plt.show()


# In[17]:


n_samples = 1000
n_clusters_range = np.linspace(5, n_samples, 50).astype(int)

plt.figure(2)

plots = []
names = []

for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    plots.append(
        plt.errorbar(
            n_clusters_range,
            np.median(scores, axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=2,
            marker=marker,
        )[0]
    )
    names.append(score_name)

plt.title(
    "Community measures for 5 random uniform labelings\nwith equal number of clusters"
)
plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
plt.ylabel("Score value")
plt.legend(plots, names)
plt.ylim(bottom=-0.05, top=1.05)
plt.show()


# In[ ]:





# In[ ]:




