import networkx as nx


def sachs():
    n = 11
    nodes = []
    for i in range(n):
        nodes.append(i+1)
    # 创建有向图
    G2 = nx.DiGraph()

    # 添加节点
    G2.add_nodes_from(nodes)

    G2.add_edge(2, 1)
    G2.add_edge(8, 1)
    G2.add_edge(4, 2)
    G2.add_edge(8, 2)
    G2.add_edge(4, 3)
    G2.add_edge(8, 3)
    G2.add_edge(8, 4)
    G2.add_edge(11, 4)
    G2.add_edge(8, 5)
    G2.add_edge(11, 5)
    G2.add_edge(7, 6)
    G2.add_edge(10, 6)
    G2.add_edge(10, 7)

    return G2

def alarm():
    n = 37
    nodes = []
    for i in range(n):
        nodes.append(i+1)

    TG = nx.DiGraph()
    TG.add_nodes_from(nodes)

    TG.add_edge(4, 5)
    TG.add_edge(4, 7)
    TG.add_edge(5, 2)
    TG.add_edge(5, 3)
    TG.add_edge(6, 1)
    TG.add_edge(6, 5)
    TG.add_edge(6, 7)
    TG.add_edge(7, 36)
    TG.add_edge(8, 9)
    TG.add_edge(11, 10)
    TG.add_edge(11, 12)
    TG.add_edge(13, 34)
    TG.add_edge(14, 15)
    TG.add_edge(15, 34)
    TG.add_edge(15, 37)
    TG.add_edge(17, 26)
    TG.add_edge(17, 31)
    TG.add_edge(19, 20)
    TG.add_edge(20, 21)
    TG.add_edge(21, 34)
    TG.add_edge(23, 22)
    TG.add_edge(23, 24)
    TG.add_edge(24, 21)
    TG.add_edge(25, 18)
    TG.add_edge(25, 24)
    TG.add_edge(25, 26)
    TG.add_edge(25, 32)
    TG.add_edge(27, 30)
    TG.add_edge(28, 29)
    TG.add_edge(29, 30)
    TG.add_edge(30, 26)
    TG.add_edge(30, 31)
    TG.add_edge(31, 16)
    TG.add_edge(31, 18)
    TG.add_edge(31, 32)
    TG.add_edge(32, 20)
    TG.add_edge(32, 33)
    TG.add_edge(33, 16)
    TG.add_edge(33, 34)
    TG.add_edge(34, 35)
    TG.add_edge(35, 9)
    TG.add_edge(35, 10)
    TG.add_edge(35, 12)
    TG.add_edge(35, 36)
    TG.add_edge(36, 37)

    return TG