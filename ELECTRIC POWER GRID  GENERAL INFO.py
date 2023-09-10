#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

G =nx.read_gml('power.gml',label=None) #lable="label"
print(nx.info(G))


# In[2]:



print (nx.number_of_nodes(G))
print (nx.number_of_edges(G))

print (nx.is_directed(G))


# In[3]:



G =nx.read_gml('power.gml',label=None) #lable="label"
nx.info(G)


# In[4]:


density = nx.density(G)

print('The edge density is: ' + str(density))


# In[5]:


#the degree function in networkx returns a DegreeView object capable of iterating through (node, degree) pairs
degree = G.degree()

degree_list = []

for (n,d) in degree:
    degree_list.append(d)

av_degree = sum(degree_list) / len(degree_list)

print('The average degree is ' + str(av_degree))


# In[9]:



import matplotlib.pyplot as plt
from operator import itemgetter
bins=[0, 2, 4, 6, 8, 10 ]
plt.hist(degree_list,label='Degree Distribution',color='#fc6b03')
plt.axvline(av_degree,color='r',linestyle='dashed',label='Average Degree')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes'        ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Network: Average Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()




# In[8]:


plt.hist([v for k,v in nx.degree(G)], label ='Node Degree',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes' ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Networks: Node Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[11]:



degree_centrality = nx.centrality.degree_centrality(G)
plt.hist(nx.centrality.degree_centrality(G).values(),label='Degree Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Degree',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Network: Degree Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[ ]:



closeness_centrality = nx.centrality.closeness_centrality(G)
plt.hist(nx.centrality.closeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Closeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Network: Closeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[14]:



centrality = nx.eigenvector_centrality(G)
plt.hist(nx.centrality.eigenvector_centrality(G).values(),label='Eigenvector Centrality',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Eigenvector',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Network: Eigenvector Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[15]:



betwenness_centrality = nx.centrality.betweenness_centrality(G)
plt.hist(nx.centrality.betweeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Betweeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Networks: Betweeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()



# In[16]:



local_clustering_coefficient = nx.algorithms.cluster.clustering(G)

av_local_clustering_coefficient = sum(local_clustering_coefficient.values())/len(local_clustering_coefficient)

plt.hist(local_clustering_coefficient.values(),label='Local Clustering Coefficient Distribution', color='#fc6b03')
plt.axvline(av_local_clustering_coefficient,color='r',linestyle='dashed',label='Average Local Clustering Coefficient')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Electric Power Grid Networks: Local Clustering Coefficient',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[17]:


from networkx.algorithms.approximation import average_clustering
average_clustering = nx.algorithms.cluster.clustering(G)
print(average_clustering)


# In[18]:


from networkx.algorithms.community.modularity_max import greedy_modularity_communities
#preform the community detection
c = list(greedy_modularity_communities(G))

#Let's find out how many communities we detected
print(len(c))


# In[19]:


print(nx.average_shortest_path_length(G))


# In[20]:


sum(list(nx.triangles(G).values()))/3


# In[21]:


print(nx.diameter(G))


# In[34]:



print(nx.radius(G))


# In[35]:


print(nx.periphery(G))


# In[36]:


print(nx.center(G))


# In[37]:


nx.transitivity(G)


# In[38]:


nx.average_clustering(G)


# In[39]:


nx.density(G)


# In[22]:


deg_centrality = nx.degree_centrality(G)


# In[23]:


print(sorted(nx.degree_centrality(G).items(),key= lambda x:x[1],
reverse=True))[0:5]


# In[24]:


from operator import itemgetter
print(sorted(nx.degree_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[25]:


close_centrality = nx.closeness_centrality(G)


# In[26]:


print(sorted(nx.closeness_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[ ]:


nx.average_node_connectivity(G)


# In[20]:


eigenvector_centrality = nx.eigenvector_centrality(G)


# In[22]:


from operator import itemgetter
print(sorted(nx.eigenvector_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[25]:


print(nx.number_connected_components)


# In[30]:


G= nx.complete_graph(100)
preds = nx.jaccard_coefficient(G, [(0, 1), (2, 5)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[32]:


G = nx.complete_graph(100)
preds = nx.adamic_adar_index(G, [(0,1), (2,9)])
for u, v, p in preds:
   print(f"({u}, {v}) -> {p:.8f}")


# In[33]:


G = nx.complete_graph(100)
preds = nx.preferential_attachment(G, [(0, 1), (4, 9)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p}")


# In[ ]:




