import os

import networkx as nx
import random
import pandas as pd
import time
from Test.tools import *
from confounders.getCausalOrders import getAdj
from sklearn.utils import check_array
import numpy as np

def frobenius_norm_difference(G1, G2):
    # 为了确保两个矩阵大小一致，需要创建一个节点集合，包含两个图中的所有节点
    nodes = set(G1.nodes()).union(set(G2.nodes()))
    # 创建两个图的邻接矩阵
    A1 = nx.to_numpy_array(G1, nodelist=nodes)
    A2 = nx.to_numpy_array(G2, nodelist=nodes)

    # 计算两个邻接矩阵的差异矩阵
    diff = A1 - A2

    # 计算并返回Frobenius norm
    return np.linalg.norm(diff, 'fro')

def mse_adjacency(G1, G2):
    # 获取两个图节点的并集，确保邻接矩阵的大小一致
    nodes = set(G1.nodes()).union(G2.nodes())
    # 创建两个图对应的邻接矩阵
    A1 = nx.to_numpy_array(G1, nodelist=nodes)
    A2 = nx.to_numpy_array(G2, nodelist=nodes)

    # 计算邻接矩阵之间的差异
    diff = A1 - A2

    # 计算差异的平方
    diff_squared = np.square(diff)

    # 统计非零元素的数量用于计算均值
    non_zero_count = np.count_nonzero(diff_squared)

    # 为了避免除以0的情况，若矩阵中没有非零项，则将MSE设为0
    if non_zero_count == 0:
        return 0
# Hilbert-Schmidt independence criterion希尔伯特-施密特独立性指标
def hsic(kx, ky):
    kxy = np.dot(kx, ky)
    n = kxy.shape[0]
    h = np.trace(kxy) / n**2 + np.mean(kx) * np.mean(ky) - 2 * np.mean(kxy) / n
    return h * n ** 2 / (n-1)**2


# 变量x回归到y的残差
def _residual(x, y):  # 500*1f
    """The residual when x is regressed on y."""
    return x - (np.cov(x, y)[0, 1] / np.var(y)) * y


filename = r'hepar2'
p = r'0.1'
# 初始网络
pcfile1 = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\trans'
# pcfile = os.path.join(pcfile1, p, filename + '1.txt')
pcfile = os.path.join(pcfile1, filename + '1.txt')
totalMB = todata(pcfile)
pcG = toGx(totalMB)
# 真实图
tfile = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\truth_dag_graph'
# tfile = os.path.join(tfile, p, filename + '.txt')
tfile = os.path.join(tfile, filename + '.graph.txt')
tgl = todata(tfile, False, True)
G = toGx(todata(tfile))
# 骨架
# sklfile = os.path.join(pcfile1, p, filename + '.txt')
sklfile = os.path.join(pcfile1, filename + '.txt')
skl = todata(sklfile, False, True)
# 数据文件
datafile = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\data'
# datafile = os.path.join(datafile, p, filename + '.csv')
datafile = os.path.join(datafile, filename + '.csv')
df = loadfile(datafile)

start = time.time()

G_2 = np.zeros((len(totalMB), len(totalMB)))

adj = getAdj(skl)
# adj = getAdj(tgl)
# print(adj)
# 获取节点序
Data = check_array(df)
# print(df.head())
# print(Data[0])
# print(Data[1])
# print(Data[2])
n = len(Data[1])

nodes = []
for i in range(n):
    nodes.append(i + 1)
# print("子节点", nodes)
# print("相邻", adj)
print(nodes)
for xi in nodes:
    nei = adj[xi]
    for yi in nei:
        x = Data[:, xi - 1]
        x1 = x.reshape(-1, 1)
        y = Data[:, yi - 1]
        y1 = y.reshape(-1, 1)

        rx = _residual(x, y).reshape(-1, 1).T
        ry = _residual(y, x).reshape(-1, 1).T
        xy = hsic(x1, rx)
        # print(f"{v1}-{v2}:{xy}")
        yx = hsic(y1, ry)
        # print(f"{v2}-{v1}:{yx}")
        if xy >= yx:
            G_2[xi - 1][yi - 1] = 1
            # G_2[yi - 1][xi - 1] = 1
        else:
            G_2[yi - 1][xi - 1] = 1
            # G_2[xi - 1][yi - 1] = 1

        adj[yi].remove(xi)

end = time.time()
print(len(G_2))

G2 = toGx(G_2)
G22 = toGxbyLingam(G_2)
# draw(G2)
G = toGx(todata(tfile))
print("结果")
# p, recall, f1 = calculate_index(G2, G)
show(G2, G, start, end, name='M1')

show(G22, G, start, end, name='M2')




