#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
G =nx.read_gml('karate.gml',label=None) #lable="label"
print(nx.info(G))


# In[2]:


nx.average_node_connectivity(G)


# In[3]:


sum(list(nx.triangles(G).values()))/3


# In[4]:


density = nx.density(G)

print('The edge density is: ' + str(density))


# In[5]:


print (nx.number_of_nodes(G))
print (nx.number_of_edges(G))

print (nx.is_directed(G))


# In[6]:



G =nx.read_gml('karate.gml',label=None) #lable="label"
nx.info(G)


# In[7]:


#the degree function in networkx returns a DegreeView object capable of iterating through (node, degree) pairs
degree = G.degree()

degree_list = []

for (n,d) in degree:
    degree_list.append(d)

av_degree = sum(degree_list) / len(degree_list)

print('The average degree is ' + str(av_degree))


# In[8]:



import matplotlib.pyplot as plt
from operator import itemgetter
bins=[0, 2, 4, 6, 8, 10 ]
plt.hist(degree_list,label='Degree Distribution',color='#fc6b03')
plt.axvline(av_degree,color='r',linestyle='dashed',label='Average Degree')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes'        ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Zakary Karate Club: Average Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()



# In[9]:


plt.hist([v for k,v in nx.degree(G)], label ='Node Degree',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes' ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Zakary Karate Club: Node Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[10]:



degree_centrality = nx.centrality.degree_centrality(G)
plt.hist(nx.centrality.degree_centrality(G).values(),label='Degree Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Degree',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Zakary Karate Club: Degree Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()



# In[11]:



closeness_centrality = nx.centrality.closeness_centrality(G)
plt.hist(nx.centrality.closeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Closeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Zakary Karate Club: Closeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[12]:



centrality = nx.eigenvector_centrality(G)
plt.hist(nx.centrality.eigenvector_centrality(G).values(),label='Eigenvector Centrality',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Eigenvector',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Zakary Karate Club: Eigenvector Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()



# In[13]:



betwenness_centrality = nx.betweenness_centrality(G)
plt.hist(nx.betweeness_centrality(G).values(),label='Betweeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Betweeness',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Zakary Karate Club: Betweeness Centrality',fontdict={'fontweight':'bold','fontsize':15})
 
plt.show()


# In[14]:



local_clustering_coefficient = nx.algorithms.cluster.clustering(G)

#lets find the average clustering coefficient
av_local_clustering_coefficient = sum(local_clustering_coefficient.values())/len(local_clustering_coefficient)

#similarly to the degree lets plot the local clustering coefficient distribution
plt.hist(local_clustering_coefficient.values(),label='Local Clustering Coefficient Distribution', color='#fc6b03')
plt.axvline(av_local_clustering_coefficient,color='r',linestyle='dashed',label='Average Local Clustering Coefficient')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':12})
plt.ylabel('Number of Nodes',fontdict={'fontweight':'bold','fontsize':12})
plt.title('Zakary Karate Club: Local Clustering Coefficient',fontdict={'fontweight':'bold','fontsize':15})
 
plt.show()


# In[15]:


from networkx.algorithms.approximation import average_clustering
average_clustering = nx.algorithms.cluster.clustering(G)
print(average_clustering)


# In[16]:


from networkx.algorithms.community.modularity_max import greedy_modularity_communities

#preform the community detection
c = list(greedy_modularity_communities(G))

#Let's find out how many communities we detected
print(len(c))


# In[17]:


print(nx.average_shortest_path_length(G))


# In[18]:


sum(list(nx.triangles(G).values()))/3


# In[19]:


print(nx.diameter(G))


# In[20]:


print(nx.radius(G))


# In[21]:


print(nx.periphery(G))


# In[22]:


print(nx.center(G))


# In[23]:


nx.transitivity(G)


# In[24]:


nx.average_clustering(G)


# In[25]:


nx.density(G)


# In[26]:


deg_centrality = nx.degree_centrality(G)


# In[27]:


from operator import itemgetter
print(sorted(nx.degree_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[28]:


close_centrality = nx.closeness_centrality(G)


# In[29]:


print(sorted(nx.closeness_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[30]:


nx.average_node_connectivity(G)


# In[31]:


eigenvector_centrality = nx.eigenvector_centrality(G)


# In[32]:


from operator import itemgetter
print(sorted(nx.eigenvector_centrality(G).items(),key=itemgetter(1),
reverse=True))


# In[33]:


print(nx.number_connected_components)


# In[34]:


G= nx.complete_graph(100)
preds = nx.jaccard_coefficient(G, [(0, 0), (1, 5)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[35]:


G = nx.complete_graph(100)
preds = nx.adamic_adar_index(G, [(0,0), (1,99)])
for u, v, p in preds:
   print(f"({u}, {v}) -> {p:.8f}")


# In[36]:


G = nx.complete_graph(100)
preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p}")

