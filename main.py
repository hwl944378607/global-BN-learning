# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import d_separation

import partition
from Test.tools import *
from confounders.getConfounders import getUN
from partition.ancestral_partition import ancestralPartition
from partition.getSCCs import *
from partition.subset_learning import sublearning
from partition.tarjan import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
from causallearn.search.FCMBased import lingam

# bnlearn
# import bnlearn as bn
# import sys
# bn_learn_path = "F:/codes/BN_learn"    #需要用户自己设定
# sys.path.append(bn_learn_path)
# from add_bn_learn_tools_path import *
# add_bn_learn_tools_path(bn_learn_path)
# from import_bn_learn_tools import *

# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    filename = r'child'
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

    # draw(pcG)
    # show(pcG, G, 0, 1, name='pc')
    # show(toGx(skl), G, 0, 1, name='adl')
    # """
    # 约束划分
    ps = ancestralPartition(pcG)
    #N = len(pcG)
    G_2 = totalMB
    #
    # 子集学习
    for i in range(len(ps)):
        # print("子集", ps[i])
        G_2 = sublearning(G_2, ps[i], df, skl)

    # 不使用约束划分

    # G_2 = np.zeros((len(totalMB), len(totalMB)))
    # G_2 = sublearning(G_2, pcG.nodes(), df, tgl)
    # 隐变量检测
    # edgs = getUN(G_2, skl)
    # if len(edgs)>0:
    #     # 查找隐变量



    end = time.time()
    #
    G2 = toGx(G_2)


    # print("最终结果")
    # draw(G2)
    # print("真实图")

    show(pcG, G, 1, 1, name='pcG')
    show(toGx(skl), G, 1, 1, name='skl')

    print("结果")
    # p, recall, f1 = calculate_index(G2, G)
    precision, recall, f1, SHD, FDR = calculate_metrics(G2, G)
    print("精确率（Precision）:", precision)
    print("召回率（Recall）:", recall)
    print("F1 值:", f1)
    print("结构汉明距离（SHD）:", SHD)
    print("伪发现率（FDR）:", FDR)
    print('程序运行时间为: %s Seconds' % (end - start))
# """
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
