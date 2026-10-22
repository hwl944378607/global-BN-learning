import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import re
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from sklearn.utils import check_array

from Test.tools import todata
from confounders.getCausalOrders import getOrders, findOrder


def sublearning(G2, scc, datafile, skl):
    # 删除scc之间的边
    G = remove_edges(G2, scc)

    G2 = global_learning(scc, datafile, skl)
    # print(scc)
    # print(G)
    # print(G2)
    nodes = list(scc)
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            source_node = nodes[i]
            target_node = nodes[j]
            edge_weight = G2[i, j]

            # 添加有向边到 G
            if edge_weight != 0:
                G[source_node - 1, target_node - 1] = 1

    #print(G)
    return G


# 删除G中的内部边
def deleteG(G, scc):
    A = set(scc)
    # 将子集内的边删除
    edges_to_remove = []
    for edge in G.edges():
        u, v = edge
        if u in A and v in A:
            edges_to_remove.append(edge)
    G.remove_edges_from(edges_to_remove)  # 删除边
    print("删除的边", edges_to_remove)

    return G


def remove_edges(adj_matrix, node_set):
    A = set(node_set)
    sets = {int(x) - 1 for x in A}
    nodes = np.array(list(sets))
    adj_matrix[nodes[:, np.newaxis], nodes] = 0
    return adj_matrix


# 在子集上进行全局学习
def global_learning(scc, datafile, skl):
    # 用全局算法在子集内部计算
    # 数据划分
    # print("xxxxxx")
    # print(scc)
    # print(datafile.head())
    nodes = list(scc)
    data = datafile.iloc[:, [col_index - 1 for col_index in nodes]]
    # print(data.head())
    # 骨架获取
    selected_nodes = np.ix_(np.array(nodes) - 1, np.array(nodes) - 1)
    skeleton = skl[selected_nodes]
    # sg = getOrders(data, skeleton)
    sg = findOrder(data, skeleton)
    return sg