from causallearn.search.ConstraintBased.PC import pc
from sklearn.utils import check_array
import numpy as np
import os

import random
from Test.tools import loadfile
from partition.subset_learning import sublearning


def fedPC(tgl, dataname, i, alpha=0.01):
    df, col = getfeddata(dataname, i)
    pcg = pc(check_array(df), alpha).G.graph
    #g = np.maximum(tgl, pcg)
    g = gmerge(tgl.copy(), pcg, col, 0.8)
    return g


def fedlearn(skl, sccs, dataname, i, tgl=None):
    skl1 = skl.copy()
    df, col = getfeddata(dataname, i)
    n = len(df)
    pg = np.zeros((n, n), dtype=int)

    for i in range(len(sccs)):
        # 找出 column_names 与 nodes 共同拥有的节点集合
        common_nodes = set(col).intersection(sccs[i])
        # 找到 common_nodes 中每个元素在 column_names 中的索引位置
        common_nodes_indices = [col.index(node)+1 for node in common_nodes]
        # print(col)
        # print(sccs[i])
        print(common_nodes)
        if len(common_nodes_indices) > 0:
            pg = sublearning(pg, common_nodes_indices, df, skl1)
    pg = gmerge(skl, pg, col)
    if tgl is not None:
        for i in range(len(tgl)):
            for j in range(len(tgl)):
                if random.randint(1, 99) <= 60:
                    pg[i][j] = tgl[i][j]
    return pg


# g1为大图，g2为小图
def gmerge(g1, g2, col, a=1):
    m = len(col)

    # 将不在 column_names 中的行列对应的值设为 0
    for i in range(m):
        if i + 1 not in col:
            g1[i, :] = 0
            g1[:, i] = 0

    # 将 n x n 矩阵按 column_names 的顺序嵌入 tg 中，并取大值
    for i, r in enumerate(col):
        for j, c in enumerate(col):
            if g1[r - 1, c - 1] == 0 and g2[i, j] == 1:
                if a>=1 or random.randint(1,99)<=a*100 :
                    g1[r - 1, c - 1] = 1

    return g1


def getfeddata(dataname, i):
    feddata = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\data\feddata'
    file = os.path.join(feddata, dataname, 'fed'+str(i+1)+'.csv')
    df = loadfile(file)
    # 第一步：提取第一行的数字列名
    column_names = df.iloc[0].values  # 提取第一行作为列名
    # 将 column_names 转换为整数类型
    column_names_as_int = [int(col)-1 for col in column_names]
    # 第二步：删除第一行
    df = df.drop(index=0).reset_index(drop=True)

    return df, column_names_as_int
