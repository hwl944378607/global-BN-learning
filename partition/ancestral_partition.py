import math

import networkx as nx

from partition.tarjan import tarjan_strongly_connected_subgraphs


def ancestralPartition(G, maxSize=30, minSize=2):
    n = len(G.nodes())

    if maxSize >= n:
        maxSize = n-1

    if maxSize <= 1:
        maxSize = 2

    # 使用Tarjan算法找到强连通子图
    scc = tarjan_strongly_connected_subgraphs(G)
    # 使用 sorted 函数进行升序排序，根据子图的节点数大小排序
    sorted_scc = sorted(scc, key=get_subgraph_size)

    # minSize = math.floor(n / 5)
    # minSize = max(2, minSize)
    # minSize = min(minSize, maxSize)
    # minSize = 1 + round(n/100)
    if len(sorted_scc) > 2:
        if minSize >= len(sorted_scc[0])+len(sorted_scc[1]):
            # 调用 merge_subgraphs 函数来连续合并子图
            sorted_scc = merge_subgraphs(sorted_scc, minSize, n)

    ps = []
    for i, subgraph in enumerate(sorted_scc):
        snodes = subgraph.nodes()
        print(f"强连通子图 {i+1}:")
        print("节点:", snodes)
        print("边:", subgraph.edges())
        print()
        if len(subgraph.nodes()) > 1:
            snodes = sorted(snodes)
            ps.append(snodes)

    return ps


def get_subgraph_size(subgraph):
    return len(subgraph)


def merge_subgraphs(subgraphs, maxSize, n, t=0):
    # 从 subgraphs 中找到最小的两个子图
    subgraphs = sorted(subgraphs, key=get_subgraph_size)
    min_subgraph1 = subgraphs[0]
    min_subgraph2 = subgraphs[1]

    # 计算这两个子图的节点数之和
    total_nodes = len(min_subgraph1.nodes()) + len(min_subgraph2.nodes())

    # 如果节点数之和小于 maxSize，则合并这两个子图
    go = t < (n/10) or n >= 100
    if total_nodes <= maxSize and go:
        merged_subgraph = nx.compose(min_subgraph1, min_subgraph2)
        # 从 subgraphs 中移除这两个子图
        subgraphs.remove(min_subgraph1)
        subgraphs.remove(min_subgraph2)
        # 将合并后的子图添加到 subgraphs 中
        subgraphs.append(merged_subgraph)
        # 递归调用 merge_subgraphs 继续尝试合并
        return merge_subgraphs(subgraphs, maxSize, n, t+1)
    else:
        # 如果节点数之和大于 maxSize，则停止合并
        return subgraphs
