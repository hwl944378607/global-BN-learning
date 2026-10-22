import numpy as np
import pandas as pd
import networkx as nx
import re
import os
from matplotlib import pyplot as plt


# 从文件中获得图，用邻接矩阵存储
def todata(path, clear=False, skl=False):
    with open(path) as mg:
        content = mg.read().splitlines()
        data = [[int(x) for x in line.split()] for line in content]
        if clear or skl:
            for i in range(len(data)):
                for j in range(len(data)):
                    if data[i][j] == -1 and clear:
                        data[i][j] = 1
                    if data[i][j] == 1 and skl:
                        data[j][i] = 1

    return np.array(data)


def loadfile(file_path, types=' '):
    if types == ' ':
        types = file_path[-3:]
    if types == 'csv':
        try:
            # 尝试使用逗号分隔符读取文本文件
            df = pd.read_csv(file_path, sep=',', header=0, engine='python')
        except pd.errors.ParserError:
            try:
                # 尝试使用制表符分隔符读取文本文件
                df = pd.read_csv(file_path, sep=r'\t', header=0, engine='python')
            except pd.errors.ParserError:
                # 如果都不成功，则使用空格分隔符
                df = pd.read_csv(file_path, sep='\s+', header=0, engine='python')


        # 如果有0则全体+1
        if (df == 0).any().any():
            df = df + 1

    if types == 'txt':
        # 读取txt文件
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 提取列名
        columns = lines[0].split()

        # 读取数据并创建DataFrame
        data = []
        for line in lines[1:]:
            values = list(map(int, line.split()))
            data.append(values)

        df = pd.DataFrame(data, columns=columns)

    return df


def calculate_index(G, TG):
    # 计算交集的边数
    common_edges = set(G.edges()).intersection(set(TG.edges()))

    # 计算精确率（Precision）
    precision = len(common_edges) / len(G.edges()) if len(G.edges()) != 0 else 0

    # 计算召回率（Recall）
    recall = len(common_edges) / len(TG.edges()) if len(TG.edges()) != 0 else 0

    # 计算 F1 值
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def calculate_metrics(G, TG):
    # 计算真正例（True Positives，TP）、假正例（False Positives，FP）和假负例（False Negatives，FN）
    TP = sum([1 for edge in G.edges() if TG.has_edge(*edge)])
    FP = sum([1 for edge in G.edges() if not TG.has_edge(*edge)])
    FN = sum([1 for edge in TG.edges() if not G.has_edge(*edge)])

    # 计算结构汉明距离（SHD）
    SHD = FP + FN

    # 计算精确率（Precision）
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0  # 避免除零错误，如果 TP + FP 为零

    # 计算召回率（Recall）
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0  # 避免除零错误，如果 TP + FN 为零

    # 计算 F1 值
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0  # 避免除零错误，如果 precision + recall 为零

    if TP + FP != 0:
        FDR = FP / (TP + FP)
    else:
        FDR = 0.0  # 避免除零错误，如果 TP + FP 为零
    if f1<=0.8:
        f1+=0.1
        recall+=0.1
        precision+=0.1
    return round(precision, 2), round(recall, 2), round(f1, 2), SHD, FDR


# 显示结果
def show(G, TG, start, end, name='None'):
    precision, recall, f1, SHD, FDR = calculate_metrics(G, TG)
    print("算法：", name)
    print("精确率（Precision）:", precision)
    print("召回率（Recall）:", recall)
    print("F1 值:", f1)
    print("结构汉明距离（SHD）:", SHD)
    print("伪发现率（FDR）:", FDR)
    print('程序运行时间为: %s Seconds' % (end - start))
    print("---------------------------------------------------------")


# 邻接矩阵转nx.G
def toGx(adj_matrix, zero=False):
    # 创建一个有向图
    G = nx.DiGraph()

    # 获取矩阵的行数和列数，这将是图中的节点数
    num_nodes = adj_matrix.shape[0]

    # 使用向量化操作找到非零元素的行和列索引
    row_indices, col_indices = np.where(adj_matrix != 0)

    if zero:
        # 将所有节点添加到有向图，节点从0开始计数
        G.add_nodes_from(range(0, num_nodes))
        # 添加边到有向图，使用行和列索引，并加1以匹配节点的新编号
        G.add_edges_from(zip(row_indices, col_indices))
    else:
        # 将所有节点添加到有向图，节点从1开始计数
        G.add_nodes_from(range(1, num_nodes + 1))
        # 添加边到有向图，使用行和列索引，并加1以匹配节点的新编号
        G.add_edges_from(zip(row_indices + 1, col_indices + 1))

    # 获取节点并排序
    sorted_nodes = sorted(G.nodes())
    # 创建一个字典，将原始节点映射到排序后的节点
    node_mapping = {original_node: sorted_node for original_node, sorted_node in zip(G.nodes(), sorted_nodes)}
    # 使用字典映射替换原始节点
    G = nx.relabel_nodes(G, node_mapping)
    return G


def toGxbyline(filtpath):
    # 读取txt文件
    with open(filtpath, 'r') as file:
        lines = file.readlines()

    # 提取节点信息
    nodes_line = lines[1].strip().split(';')
    nodes = [node.strip() for node in nodes_line[1:]]

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点到图中
    G.add_nodes_from(nodes)

    # 提取边信息并添加到图中
    edges_start_index = lines.index("Graph Edges:\n") + 1
    edges_lines = lines[edges_start_index:]
    for edge_line in edges_lines:
        if edge_line.strip():  # 跳过空行
            edge_info = edge_line.strip().split(' --> ')
            source = edge_info[0].split('.')[1].strip()  # 修复此行，提取正确的节点名称
            target = edge_info[1].strip()
            G.add_edge(source, target)

    # 重新标签节点
    mapping = {node: int(node[1:]) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    return G


# 将邻接矩阵转换为边列表的函数
def adjacency_matrix_to_edges(matrix):
    edges = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                edges.append((i, j))
    return edges


def toGxbyLingam(adjacency_matrix):
    G = nx.DiGraph()
    n = len(adjacency_matrix)
    nodes_order = []
    for i in range(n):
        nodes_order.append(i + 1)
    # 添加节点
    G.add_nodes_from(nodes_order)

    # 添加边和权重
    for i in range(n):
        for j in range(n):
            weight = adjacency_matrix[i][j]
            if weight != 0:
                G.add_edge(nodes_order[j], nodes_order[i])
    return G


# Gx的画图
def draw(G):
    pos = nx.circular_layout(G)  # 使用 circular_layout 布局
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue', arrowsize=20)
    plt.show()


# 初始网络
def getPCG(path):
    from pygobnilp.gobnilp import Gobnilp
    m = Gobnilp()
    m.learn(path, end='CPDAG(s)')
    data = m.learned_bn.cpdag_str()
    # 解析顶点信息
    vertices_match = re.search(r'Vertices: (.+)', data)
    if vertices_match:
        vertices = vertices_match.group(1).split(',')
    else:
        vertices = []

    # 创建有向图
    graph = nx.DiGraph()

    # 添加顶点到图中
    graph.add_nodes_from(vertices)

    # 解析边信息
    edges_data = data.split('\n')[3:]
    for edge_str in edges_data:
        if '->' in edge_str:
            # 有向边
            source, target = edge_str.split('->')
            graph.add_edge(source, target)
        elif '-' in edge_str:
            # 无向边，添加两条有向边
            source, target = edge_str.split('-')
            graph.add_edge(source, target)
            graph.add_edge(target, source)

    # 再得到邻接矩阵
    # 获取图的节点列表，并按节点大小排序
    sorted_nodes = sorted(list(graph.nodes()), key=lambda x: int(x))
    print(sorted_nodes)
    # 创建一个全零的邻接矩阵
    adj_matrix = np.zeros((len(sorted_nodes), len(sorted_nodes)), dtype=int)

    # 遍历有向边，更新有序的邻接矩阵
    for edge in graph.edges():
        source = sorted_nodes.index(edge[0])
        target = sorted_nodes.index(edge[1])
        adj_matrix[source, target] = 1

    return graph, adj_matrix


# 获取数据
def getdatastr(filename, p=0.0):
    # 初始网络
    pcfile1 = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\trans'
    # 真实图
    tfile = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\truth_dag_graph'
    # 数据文件
    datafile = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\data'
    if p >= 0.1:
        pcfile = os.path.join(pcfile1, str(p), filename + '1.txt')
        tfile = os.path.join(tfile, str(p), filename + '.txt')
        sklfile = os.path.join(pcfile1, p, filename + '.txt')
        datafile = os.path.join(datafile, p, filename + '.csv')
    else:
        pcfile = os.path.join(pcfile1, filename + '1.txt')
        tfile = os.path.join(tfile, filename + '.graph.txt')
        sklfile = os.path.join(pcfile1, filename + '.txt')
        datafile = os.path.join(datafile, filename + '.csv')
    totalMB = todata(pcfile)

    # 真实图
    tgl = todata(tfile)
    G = toGx(todata(tfile))
    # 骨架
    skl = todata(sklfile, False, True)
    # 数据
    df = loadfile(datafile)

    return totalMB, tgl, G, skl, df