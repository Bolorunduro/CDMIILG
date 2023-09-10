#!/usr/bin/env python
# coding: utf-8

# In[28]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


G = nx.read_pajek('USAir97.net')
print(nx.info(G))


# In[30]:



print (nx.number_of_nodes(G))
print (nx.number_of_edges(G))

print (nx.is_directed(G))


# In[31]:



G = nx.read_pajek('USAir97.net')
nx.info(G)


# In[32]:


density = nx.density(G)

print('The edge density is: ' + str(density))


# In[33]:


#the degree function in networkx returns a DegreeView object capable of iterating through (node, degree) pairs
degree = G.degree()

degree_list = []

for (n,d) in degree:
    degree_list.append(d)

av_degree = sum(degree_list) / len(degree_list)

print('The average degree is ' + str(av_degree))


# In[34]:



import matplotlib.pyplot as plt
from operator import itemgetter
bins=[0, 2, 4, 6, 8, 10 ]
plt.hist(degree_list,label='Degree Distribution',color='#fc6b03')
plt.axvline(av_degree,color='r',linestyle='dashed',label='Average Degree')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes'        ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('USAIR97 Networks: Average Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[35]:



plt.hist([v for k,v in nx.degree(G)], label ='Node Degree',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes' ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('USAIR97 Networks: Node Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[36]:


degree_centrality = nx.centrality.degree_centrality(G)
plt.hist(nx.centrality.degree_centrality(G).values(),label='Degree Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Degree',fontdict={'fontweight':'bold','fontsize':10})
plt.title('USAIR97 NETWORK: Degree Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[37]:



closeness_centrality = nx.centrality.closeness_centrality(G)
plt.hist(nx.centrality.closeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Closeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('USAIR97 NETWORKS: Closeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[38]:


centrality = nx.eigenvector_centrality(G)
plt.hist(nx.centrality.eigenvector_centrality(G).values(),label='Eigenvector Centrality',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Eigenvector',fontdict={'fontweight':'bold','fontsize':10})
plt.title('USAIR97 NETWORKS: Eigenvector Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[39]:



def node_betweenness(G,topk=15):
    #calculate node_betweenness using networkx
    #G is networkx type graph
    nodes_bet=nx.betweenness_centrality(G,normalized=True)
    np.save('node_betweenness.npy',nodes_bet)
    sorted_nodes_bet=sorted(nodes_bet.items(),key=lambda x:x[1],reverse=True)
    print(sorted_nodes_bet[:topk])


# In[40]:


#Now we can compute the local clustering coefficient
local_clustering_coefficient = nx.algorithms.cluster.clustering(G)

#lets find the average clustering coefficient
av_local_clustering_coefficient = sum(local_clustering_coefficient.values())/len(local_clustering_coefficient)

#similarly to the degree lets plot the local clustering coefficient distribution
plt.hist(local_clustering_coefficient.values(),label='Local Clustering Coefficient Distribution')
plt.axvline(av_local_clustering_coefficient,color='r',linestyle='dashed',label='Average Local Clustering Coefficient')
plt.legend()
plt.ylabel('Number of Nodes')
plt.title('Local Clustering Coefficient of USAIR97 NETWORKS')
plt.show()


# In[41]:


from networkx.algorithms.approximation import average_clustering
average_clustering = nx.algorithms.cluster.clustering(G)
print(average_clustering)


# In[42]:


print(nx.average_shortest_path_length(G))


# In[43]:


print(nx.diameter(G))


# In[44]:


print(nx.radius(G))


# In[45]:


print(nx.periphery(G))


# In[46]:


print(nx.center(G))


# In[47]:


nx.transitivity(G)


# In[48]:


nx.density(G)


# In[49]:


from operator import itemgetter
print(sorted(nx.degree_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[50]:


close_centrality = nx.closeness_centrality(G)


# In[79]:


print(sorted(nx.closeness_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[80]:


nx.average_node_connectivity(G)


# In[51]:


print(nx.average_shortest_path_length(G))


# In[52]:



eigenvector_centrality = nx.eigenvector_centrality(G)


# In[53]:


from operator import itemgetter
print(sorted(nx.eigenvector_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[54]:


sum(list(nx.triangles(G).values()))/3


# In[55]:


from networkx.algorithms.approximation import average_clustering
average_clustering = nx.algorithms.cluster.clustering(G)
print(average_clustering)


# In[56]:


print(nx.number_connected_components)


# In[57]:


G= nx.complete_graph(100)
preds = nx.jaccard_coefficient(G, [(0, 0), (1, 5)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[58]:


G = nx.complete_graph(100)
preds = nx.adamic_adar_index(G, [(0,0), (1,99)])
for u, v, p in preds:
   print(f"({u}, {v}) -> {p:.8f}")


# In[59]:


G = nx.complete_graph(100)
preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p}")

