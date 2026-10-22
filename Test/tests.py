import os
import time
import networkx as nx
import numpy as np
from causallearn.search.FCMBased import lingam
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from sklearn.utils import check_array

from Test.tools import draw, todata, toGx, loadfile, toGxbyLingam, calculate_metrics, show, toGxbyline
# from pygobnilp.gobnilp import Gobnilp

from datas.truth_dag_graph.Large_Networks_graph import *
from datas.truth_dag_graph.Medium_Networks_Graph import *

# import bnlearn as bn

filename = r'Child'
p = r'0.2'
# 初始网络
pcfile1 = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\trans'
pcfile = os.path.join(pcfile1, p, filename + '1.txt')
# pcG = toGx(todata(pcfile))
# 真实图
tfile = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\truth_dag_graph'
# tfile = os.path.join(tfile, p, filename + '.txt')
tfile = os.path.join(tfile, filename + '.graph.txt')
tG = toGx(todata(tfile))
# tG = toGxbyline(tfile)
# 骨架
sklfile = os.path.join(pcfile1, p, filename + '.txt')
# skl = todata(sklfile, False, True)
# skl = todata(sklfile, False, True)
# pcG = toGx(todata(sklfile))
# # datafile = r'G:\codes\global-BN-learning\data\data2\Small Networks\earthquake5000.csv'
# 数据文件
datafile1 = r'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\data'
# datafile = os.path.join(datafile1, p, filename + '.csv')
datafile = os.path.join(datafile1, filename + '.csv')

# gofile = os.path.join(datafile1, 'gobulin', p, filename + '.txt')
gofile = os.path.join(datafile1, 'gobulin', filename + '.txt')

# draw(toGx(skl))
# show(pcG, tG, 0, 1, name='pc')
# show(toGx(skl), tG, 0, 1, name='skl')
#
n = len(tG.nodes())
df = loadfile(datafile)

# tg = win95pts_int()
# # tg = hepar2_int()
#
# # 获取节点列表并排序
# nodes = list(tg.nodes())
# nodes.sort()
#
# # 创建节点的映射字典
# node_mapping = {node: index for index, node in enumerate(nodes)}
#
# # 创建新的有序图
# sorted_tg = nx.DiGraph()
# sorted_tg.add_nodes_from(nodes)
#
# # 添加重新排序后的节点和边
# for edge in tg.edges():
#     sorted_tg.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
#
#
# print(sorted_tg.nodes())
# print(sorted_tg.edges())
#
#
#
# # show(tG, tg, 0, 1, name='tg')
# #获取邻接矩阵
# # adjacency_matrix = nx.adjacency_matrix(tg).todense()
# adjacency_matrix = nx.to_numpy_array(sorted_tg)
#
# # draw(toGx(adjacency_matrix))
# show(toGx(adjacency_matrix, zero=True), sorted_tg, 0, 1, name='tgg')
# # 保存邻接矩阵到txt文件
# np.savetxt(r'G:\codes\global-BN-learning\datas\truth_dag_graph\win95pts.graph.txt', adjacency_matrix, fmt='%d')


"""
totalMB = todata(sklfile)
pcG = toGx(totalMB)

# lingam
"""
# start = time.time()
# model = lingam.DirectLiNGAM()
# print(df.head())
# model.fit(df)
# G1 = toGxbyLingam(model.adjacency_matrix_)
# end = time.time()
# show(G1, tG, start, end, name='DirectLiNGAM')
# G1 = toGx(model.adjacency_matrix_)
# show(G1, tG, start, end, name='DirectLiNGAM_toGx')

# VARLiNGAM
# model = lingam.VARLiNGAM()
# model.fit(df)
# G11 = toGxbyLingam(model.adjacency_matrices_[0])
# # G11 = toGx(model.adjacency_matrices_[0])
# end = time.time()
# show(G11, tG, start, end, name='VARLiNGAM')

# PC
print(type(df.columns))
print(df.columns)
start = time.time()
cg = pc(check_array(df), 0.05)
end = time.time()
# print(type(cg.G.graph))
# print(type(todata(tfile)))
# print(cg.G.graph)
G2 = toGx(cg.G.graph)
show(G2, tG, start, end, name='PC')



# GIN
"""
# Gobnilp
m = Gobnilp()
start = time.time()
m.learn(gofile, score='DiscreteBIC', end='CPDAG(s)')
# m.learn(gofile)
bn = m.learned_bn
end = time.time()
network = np.zeros((n, n))
for child, local_score in bn.nodes(data='local_score'):
    pc = sorted(bn.predecessors(child))
    for i in range(len(pc)):
        network[int(pc[i]) - 1][int(child) - 1] = 1
G4 = toGx(network)
show(G4, tG, start, end, name='Gobnilp')

# Gobnilp2
start = time.time()
m.learn(gofile, end='CPDAG(s)')
bn = m.learned_bn
end = time.time()
network = np.zeros((n, n))
for child, local_score in bn.nodes(data='local_score'):
    pc = sorted(bn.predecessors(child))
    for i in range(len(pc)):
        network[int(pc[i]) - 1][int(child) - 1] = 1
G4 = toGx(network)
show(G4, tG, start, end, name='Gobnilp2')
"""
# GES
# start = time.time()
# res_map = ges(check_array(df), score_func='local_score_BIC', maxP=None, parameters=None)
# end = time.time()
# G3 = toGx(res_map['G'].graph)
# show(G3, tG, start, end, name='GES')

#draw(tG)