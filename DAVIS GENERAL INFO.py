#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import networkx as nx
from tqdm import tqdm


# In[28]:


#!/usr/bin/env python

import string
import networkx as nx

def davis_club_graph(create_using=None, **kwds):
    nwomen=18
    nclubs=14
    G=nx.generators.empty_graph(nwomen+nclubs,create_using=create_using,**kwds)
    G.clear()
    G.name="Davis Southern Club Women"

    women="""EVELYN
LAURA
THERESA
BRENDA
CHARLOTTE
FRANCES
ELEANOR
PEARL
RUTH
VERNE
MYRNA
KATHERINE
SYLVIA
NORA
HELEN
DOROTHY
OLIVIA
FLORA"""

    clubs="""E1
E2
E3
E4
E5
E6
E7
E8
E9
E10
E11
E12
E13
E14"""

    davisdat="""1 1 1 1 1 1 0 1 1 0 0 0 0 0
1 1 1 0 1 1 1 1 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0
1 0 1 1 1 1 1 1 0 0 0 0 0 0
0 0 1 1 1 0 1 0 0 0 0 0 0 0
0 0 1 0 1 1 0 1 0 0 0 0 0 0
0 0 0 0 1 1 1 1 0 0 0 0 0 0
0 0 0 0 0 1 0 1 1 0 0 0 0 0
0 0 0 0 1 0 1 1 1 0 0 0 0 0
0 0 0 0 0 0 1 1 1 0 0 1 0 0
0 0 0 0 0 0 0 1 1 1 0 1 0 0
0 0 0 0 0 0 0 1 1 1 0 1 1 1
0 0 0 0 0 0 1 1 1 1 0 1 1 1
0 0 0 0 0 1 1 0 1 1 1 1 1 1
0 0 0 0 0 0 1 1 0 1 1 1 1 1
0 0 0 0 0 0 0 1 1 1 0 1 0 0
0 0 0 0 0 0 0 0 1 0 1 0 0 0
0 0 0 0 0 0 0 0 1 0 1 0 0 0"""


    # women names
    w={}
    n=0
    for name in women.split('\n'):
        w[n]=name
        n+=1

    # club names
    c={}
    n=0
    for name in clubs.split('\n'):
        c[n]=name
        n+=1

    # parse matrix
    row=0
    for line in davisdat.split('\n'):
        thisrow=list(map(int,line.split(' ')))
        for col in range(0,len(thisrow)):
            if thisrow[col]==1:
                G.add_edge(w[row],c[col])
        row+=1
    return (G,list(w.values()),list(c.values()))

def project(B,pv,result=False,**kwds):
    """
    """
    if result:
        G=result
    else:
        G=nx.Graph(**kwds)
    for v in pv:
        G.add_node(v)
        for cv in B.neighbors(v):
            G.add_edges_from([(v,u) for u in B.neighbors(cv)])
    return G

if __name__ == "__main__":
    # return graph and women and clubs lists
    (G,women,clubs)=davis_club_graph()

    # project bipartite graph onto women nodes
    W=project(G,women)
    # project bipartite graph onto club nodes
    C=project(G,clubs)

    print("Degree distributions of projected graphs")
    print('')
    print("Member #Friends")
    for v in W:
        print('%s %d' % (v,W.degree(v)))

    print('')
    print("Clubs #Members")
    for v in C:
        print('%s %d' % (v,C.degree(v)))


# In[29]:


G = nx.davis_southern_women_graph()


# In[30]:


G.nodes()


# In[31]:


G.degree()


# In[32]:


degrees = ((node, G.degree(node)) for node in G.nodes())
degrees = ((node, degree) for node, degree in degrees if degree > 10)
print("Node\tDegree")
for node, degree in degrees:
    print("{}\t{}".format(node, degree))


# In[33]:


print(nx.average_shortest_path_length(G))


# In[34]:


print(nx.diameter(G))


# In[35]:


print(nx.radius(G))


# In[36]:


print(nx.periphery(G))


# In[37]:


print(nx.center(G))


# In[38]:


print(nx.number_connected_components)


# In[39]:


nx.density(G )


# In[40]:


print(nx.info(G))


# In[41]:


print (nx.number_of_nodes(G))
print (nx.number_of_edges(G))

print (nx.is_directed(G))


# In[42]:


#the degree function in networkx returns a DegreeView object capable of iterating through (node, degree) pairs
degree = G.degree()

degree_list = []

for (n,d) in degree:
    degree_list.append(d)

av_degree = sum(degree_list) / len(degree_list)

print('The average degree is ' + str(av_degree))


# In[43]:


#we now plot the degree distribution to get a better insight
import matplotlib.pyplot as plt
from operator import itemgetter
bins=[0, 2, 4, 6, 8, 10 ]
plt.hist(degree_list,label='Degree Distribution',color='#fc6b03')
plt.axvline(av_degree,color='r',linestyle='dashed',label='Average Degree')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes'        ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Davis Club: Average Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[44]:



plt.hist([v for k,v in nx.degree(G)], label ='Node Degree',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes' ,fontdict={'fontweight':'bold','fontsize':10})
plt.title('Davis Club: Node Degree',fontdict={'fontweight':'bold','fontsize':12} )
plt.show()


# In[45]:



degree_centrality = nx.centrality.degree_centrality(G)
plt.hist(nx.centrality.degree_centrality(G).values(),label='Degree Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Degree',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Davies Club: Degree Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[46]:


print(nx.degree_centrality(G))


# In[47]:


closeness_centrality = nx.centrality.closeness_centrality(G)
plt.hist(nx.centrality.closeness_centrality(G).values(),label='closeness Centrality', color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':12})
plt.ylabel('Number of Closeness',fontdict={'fontweight':'bold','fontsize':12})
plt.title('Davis Club: Closeness Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[48]:



print(nx.closeness_centrality(G))


# In[49]:


centrality = nx.eigenvector_centrality(G)
plt.hist(nx.centrality.eigenvector_centrality(G).values(),label='Eigenvector Centrality',color='#fc6b03');
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Eigenvector',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Davies Club: Eigenvector Centrality',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[50]:


centrality = nx.eigenvector_centrality(G)
print(nx.eigenvector_centrality(G))


# In[51]:


#Now we can compute the local clustering coefficient
local_clustering_coefficient = nx.algorithms.cluster.clustering(G)

#lets find the average clustering coefficient
av_local_clustering_coefficient = sum(local_clustering_coefficient.values())/len(local_clustering_coefficient)

#similarly to the degree lets plot the local clustering coefficient distribution
plt.hist(local_clustering_coefficient.values(),label='Local Clustering Coefficient Distribution', color='#fc6b03')
plt.axvline(av_local_clustering_coefficient,color='r',linestyle='dashed',label='Average Local Clustering Coefficient')
plt.legend()
plt.xlabel('Distributions',fontdict={'fontweight':'bold','fontsize':10})
plt.ylabel('Number of Nodes',fontdict={'fontweight':'bold','fontsize':10})
plt.title('Davis Club: Local Clustering Coefficient',fontdict={'fontweight':'bold','fontsize':12})
 
plt.show()


# In[52]:


nx.algorithms.cluster.clustering(G)


# In[53]:


print(nx.transitivity(G))


# In[54]:


nodes = list(range(100))

df =pd.DataFrame({'from': np.random.choice(nodes,100),
                 'to':np.random.choice(nodes,100)
                 })


# In[55]:


df


# In[56]:


G= nx.complete_graph(100)
preds = nx.jaccard_coefficient(G, [(0, 0), (1, 5)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[57]:


G = nx.complete_graph(100)
preds = nx.adamic_adar_index(G, [(0,0), (1,99)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p:.8f}")


# In[58]:


G = nx.complete_graph(100)
preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
for u, v, p in preds:
    print(f"({u}, {v}) -> {p}")

