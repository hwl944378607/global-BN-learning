import numpy as np
from causallearn.search.FCMBased import lingam
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array


def getOrders(data, skeleton, alpha=0.05):
    adj = getAdj(skeleton)
    # 获取节点序
    nodes = []
    n = len(check_array(data)[1])
    for i in range(n):
        nodes.append(i + 1)

    G = np.zeros((n, n), dtype=int)

    model = lingam.DirectLiNGAM()
    model.fit(data)
    matrix = model.adjacency_matrix_
    for i in range(n):
        for j in range(n):
            if matrix[j][i] != 0 and skeleton[i][j] == 1:
                G[i][j] = 1

    return G


def findOrder(data, skeleton, alpha=0.05):
    adj = getAdj(skeleton)
    # 获取节点序
    Data = check_array(data)
    n = len(Data[1])
    nodes = []
    for i in range(n):
        nodes.append(i + 1)
    # print("子节点", nodes)
    # print("相邻", adj)
    G = np.zeros((n, n), dtype=int)
    model = lingam.DirectLiNGAM()
    for xi in nodes:
        nei = adj[xi]
        if len(nei) > 0:
            for xj in nei:
                rij = model._residual(Data[:, xi - 1], Data[:, xj - 1])
                pij = hsic(getCore(Data[:, xj - 1]), getCore(rij))
                rji = model._residual(Data[:, xj - 1], Data[:, xi - 1])
                pji = hsic(getCore(Data[:, xi - 1]), getCore(rji))
                #if pij >= 0.000001 or pji >= 0.000001:
                if pij >= 2 * pji:
                    G[xi - 1][xj - 1] = 1
                if pji >= 2 * pij:
                    G[xj - 1][xi - 1] = 1
                """
                rij = model._residual(Data[:, xi-1], Data[:, xj-1])
                pij = hsic(Data[:, xj-1], rij)
                rji = model._residual(Data[:, xj-1], Data[:, xi-1])
                pji = hsic(Data[:, xi-1], rji)
                # if abs(pij-pji)>=alpha:
                if pij > pji and pij > 0:
                    G[xi-1][xj-1] = 1
                if pji > pij and pji > 0:
                    G[xj-1][xi-1] = 1
                """
                if xi in adj.get(xj, []):
                    adj[xj].remove(xi)

    return G

def hsic(Kx, Ky):

    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n ** 2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n ** 2 / (n - 1) ** 2

# def hsic(x, y):
#     Kx = x[:, np.newaxis]
#     Ky = y[np.newaxis, :]
#
#     Kxy = np.dot(Kx, Ky)
#     n = Kxy.shape[0]
#     h = np.trace(Kxy) / n ** 2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
#     return h * n ** 2 / (n - 1) ** 2


def getCore(x):
    Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    Kx = np.exp(- Kx ** 2)  # 计算核矩阵

    return Kx


def hsic2(xi, r, kernel="linear"):
    # 计算核矩阵
    K_xi = pairwise_kernels(xi.reshape(-1, 1), metric=kernel)
    K_r = pairwise_kernels(r.reshape(-1, 1), metric=kernel)

    # 计算 HSIC 统计量
    n = len(xi)
    hsic_statistic = np.trace((1 / n ** 2) * K_xi @ K_r @ K_xi)

    # 计算自由度调整项
    df_adjustment = (np.trace(K_xi) * np.trace(K_r)) / (n ** 2)

    # 计算最终的 HSIC 统计量
    hsic_statistic -= df_adjustment

    # 计算阈值
    threshold = np.sqrt((2 * hsic_statistic * (2 * n - 1)) / (n * (n - 1)))

    return hsic_statistic, threshold


def getAdj(graph_matrix):
    """
    返回无向图的相邻节点字典

    参数:
    - graph_matrix: 二维列表，表示无向图的邻接矩阵

    返回:
    - adj_dict: 字典，包含每个节点及其相邻节点列表
    """
    num_nodes = len(graph_matrix)
    adj_dict = {}

    for i in range(num_nodes):
        node = i + 1  # 假设节点编号为1, 2, 3, ...
        neighbors = [j + 1 for j in range(num_nodes) if graph_matrix[i][j] == 1]
        adj_dict[node] = neighbors

    return adj_dict
