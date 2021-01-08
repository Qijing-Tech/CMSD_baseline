# -*- coding: utf-8 -*-
"""
@Time ：　2021/1/8
@Auth ：　xiaer.wang
@File ：　louvain_test.py
@IDE 　：　PyCharm
"""
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# define the graph
edge = [(1,2),(1,3),(1,4),(1,5),(1,6),(2,7),(2,8),(2,9)]
G = nx.Graph()
G.add_edges_from(edge)

# retrun partition as a dict
partition = community_louvain.best_partition(G)
print(partition)
# visualization
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=100,cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()