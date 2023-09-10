#!/usr/bin/env python
# coding: utf-8

# In[37]:



import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def read_edgelist(filename):
    G = nx.Graph()
    array = np.loadtxt(filename, dtype=int)
    G.add_edges_from(array)
    return G


# In[38]:



G = nx.read_edgelist('facebook_combined.txt')
n = len(G)
m = len(G.edges())
n, m
print(nx.info(G))
print (nx.number_of_nodes(G))
print (nx.number_of_edges(G))

print (nx.is_directed(G))
density = nx.density(G)

print('The edge density is: ' + str(density))


# In[39]:


degree = G.degree()

degree_list = []

for (n,d) in degree:
    degree_list.append(d)

av_degree = sum(degree_list) / len(degree_list)

print('The average degree is ' + str(av_degree))


# In[40]:



import matplotlib.pyplot as plt
from operator import itemgetter
bins=[0, 2, 4, 6, 8, 10 ]
plt.hist(degree_list,label='Degree Distribution',color='#fc6b03')
plt.axvline(av_degree,color='r',linestyle='dashed',label='Average Degree')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes'        ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK COMBINED: Average Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[41]:



plt.hist([v for k,v in nx.degree(G)], label ='Node Degree',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes' ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK COMBINED: Node Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[42]:



degree_centrality = nx.centrality.degree_centrality(G)
plt.hist(nx.centrality.degree_centrality(G).values(),label='Degree Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Degree',fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK COMBINED: Degree Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[43]:



closeness_centrality = nx.centrality.closeness_centrality(G)
plt.hist(nx.centrality.closeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Closeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK  COMBINED: Closeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()





# In[35]:



centrality = nx.eigenvector_centrality(G)
plt.hist(nx.centrality.eigenvector_centrality(G).values(),label='Eigenvector Centrality',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Eigenvector',fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK COMBINED: Eigenvector Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[36]:



local_clustering_coefficient = nx.algorithms.cluster.clustering(G)

#lets find the average clustering coefficient
av_local_clustering_coefficient = sum(local_clustering_coefficient.values())/len(local_clustering_coefficient)

#similarly to the degree lets plot the local clustering coefficient distribution
plt.hist(local_clustering_coefficient.values(),label='Local Clustering Coefficient Distribution', color='#fc6b03')
plt.axvline(av_local_clustering_coefficient,color='r',linestyle='dashed',label='Average Local Clustering Coefficient')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes',fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK COMBINED: Local Clustering Coefficient',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()



# In[10]:



betwenness_centrality = nx.centrality.betweenness_centrality(G)
plt.hist(nx.centrality.betweeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Betweeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('EGO FACEBOOK CONBINED: Betweeness Centrality',fontdict={'fontweight':'bold','fontsize':15})
 
plt.show()


# In[11]:


from networkx.algorithms.approximation import average_clustering
average_clustering = nx.algorithms.cluster.clustering(G)
print(average_clustering)


# In[12]:


print(nx.transitivity(G))


# In[13]:


print(nx.average_shortest_path_length(G))


# In[14]:


print(nx.diameter(G))


# In[14]:


print(nx.radius(G))


# In[15]:


print(nx.periphery(G))


# In[16]:


print(nx.center(G))


# In[17]:


nx.density(G )


# In[18]:


print(nx.number_connected_components)


# In[19]:


G= nx.complete_graph(100)
preds = nx.jaccard_coefficient(G, [(1, 2), (2, 5)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[20]:


G = nx.complete_graph(100)
preds = nx.adamic_adar_index(G, [(1,3), (2,99)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[21]:


G = nx.complete_graph(100)
preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p}")


# In[ ]:




