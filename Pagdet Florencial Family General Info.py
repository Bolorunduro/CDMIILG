#!/usr/bin/env python
# coding: utf-8

# In[37]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
G = nx.florentine_families_graph()
print(nx.info(G))


# In[38]:


print (nx.number_of_nodes(G))
print (nx.number_of_edges(G))

print (nx.is_directed(G))


# In[39]:



G = nx.florentine_families_graph()
nx.info(G)


# In[40]:


density = nx.density(G)

print('The edge density is: ' + str(density))


# In[41]:


#the degree function in networkx returns a DegreeView object capable of iterating through (node, degree) pairs
degree = G.degree()

degree_list = []

for (n,d) in degree:
    degree_list.append(d)

av_degree = sum(degree_list) / len(degree_list)

print('The average degree is ' + str(av_degree))


# In[42]:



import matplotlib.pyplot as plt
from operator import itemgetter
bins=[0, 2, 4, 6, 8, 10 ]
plt.hist(degree_list,label='Degree Distribution',color='#fc6b03')
plt.axvline(av_degree,color='r',linestyle='dashed',label='Average Degree')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes'        ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Padget Florentine Families: Average Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[43]:


plt.hist([v for k,v in nx.degree(G)], label ='Node Degree',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes' ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Pagdet Florentine Families: Node Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[44]:



degree_centrality = nx.centrality.degree_centrality(G)
plt.hist(nx.centrality.degree_centrality(G).values(),label='Degree Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Degree',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Paget Florentine Families: Degree Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[45]:



closeness_centrality = nx.centrality.closeness_centrality(G)
plt.hist(nx.centrality.closeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Closeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Paget Florentine Families: Closeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[46]:



centrality = nx.eigenvector_centrality(G)
plt.hist(nx.centrality.eigenvector_centrality(G).values(),label='Eigenvector Centrality',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Eigenvector',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Padget Florentine Families: Eigenvector Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[47]:



betwenness_centrality = nx.centrality.betweenness_centrality(G)
plt.hist(nx.centrality.betweeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Betweeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Padget Florentine Families: Betweeness Centrality',fontdict={'fontweight':'bold','fontsize':15})
 
plt.show()


# In[48]:



local_clustering_coefficient = nx.algorithms.cluster.clustering(G)

#lets find the average clustering coefficient
av_local_clustering_coefficient = sum(local_clustering_coefficient.values())/len(local_clustering_coefficient)

#similarly to the degree lets plot the local clustering coefficient distribution
plt.hist(local_clustering_coefficient.values(),label='Local Clustering Coefficient Distribution', color='#fc6b03')
plt.axvline(av_local_clustering_coefficient,color='r',linestyle='dashed',label='Average Local Clustering Coefficient')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Padget Florentine Families: Local Clustering Coefficient',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()





# In[49]:


from networkx.algorithms.approximation import average_clustering
average_clustering = nx.algorithms.cluster.clustering(G)
print(average_clustering)


# In[50]:


print(nx.average_shortest_path_length(G))


# In[51]:


sum(list(nx.triangles(G).values()))/3


# In[52]:


print(nx.diameter(G))


# In[53]:


print(nx.radius(G))


# In[54]:


print(nx.periphery(G))


# In[55]:


print(nx.center(G))


# In[56]:


nx.transitivity(G)


# In[57]:


nx.average_clustering(G)


# In[58]:


nx.density(G)


# In[59]:


deg_centrality = nx.degree_centrality(G)


# In[60]:


print(sorted(nx.degree_centrality(G).items(),key= lambda x:x[1],
reverse=True))


# In[61]:


close_centrality = nx.closeness_centrality(G)


# In[62]:


from operator import itemgetter
print(sorted(nx.closeness_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[63]:


nx.average_node_connectivity(G)


# In[64]:


eigenvector_centrality = nx.eigenvector_centrality(G)


# In[65]:


from operator import itemgetter
print(sorted(nx.eigenvector_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[66]:


print(nx.number_connected_components)


# In[67]:


G= nx.complete_graph(100)
preds = nx.jaccard_coefficient(G, [(0, 1), (2, 5)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[68]:


G = nx.complete_graph(100)
preds = nx.adamic_adar_index(G, [(0,1), (2,5)])
for u, v, p in preds:
   print(f"({u}, {v}) -> {p:.8f}")


# In[69]:


G = nx.complete_graph(100)
preds = nx.preferential_attachment(G, [(0, 1), (4, 9)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p}")


# In[ ]:




