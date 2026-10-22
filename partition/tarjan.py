import networkx as nx
import numpy as np

def tarjan_strongly_connected_subgraphs(graph):
    scc_subgraphs = []

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}

    def strongconnect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        for successor in graph.neighbors(node):
            if successor not in index:
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node], index[successor])

        if lowlink[node] == index[node]:
            connected_component = []
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node:
                    break
            scc_subgraphs.append(graph.subgraph(connected_component))

    for node in graph.nodes():
        if node not in index:
            strongconnect(node)

    return scc_subgraphs