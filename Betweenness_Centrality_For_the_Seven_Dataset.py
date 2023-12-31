# -*- coding: utf-8 -*-
"""BETWEENNESS CENTRALITY FOR THE SEVEN DATASET.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14luuFCwlHUhhaKrsDzdSU-AKaQq3WtbE
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import networkx as nx
from tqdm import tqdm

G = nx.read_gml('celegansneural.gml')

nx.betweenness_centrality(G)

G = nx.davis_southern_women_graph()

nx.betweenness_centrality(G)

G =nx.read_gml('power.gml',label=None) #lable="label"

nx.betweenness_centrality(G)

G = nx.read_edgelist('facebook_combined.txt')

nx.betweenness_centrality(G)

G = nx.florentine_families_graph()

nx.betweenness_centrality(G)

G = nx.read_pajek('USAir97.net')

nx.betweenness_centrality(G)

G = nx.karate_club_graph()

nx.betweenness_centrality(G)

