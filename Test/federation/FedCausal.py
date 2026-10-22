from Test.federation.Fedtool import *
from partition.ancestral_partition import ancestralPartition
from Test.tools import *
import time
import numpy as np

from partition.subset_learning import remove_edges

# 全局变量

fed_count = 8  # 客户端数量
alpha = 0.05  # 条件独立性阈值


# 数据
dataname = 'alarm'
#dataname = 'alarm3'
#dataname = 'Child'
totalMB, tgl, G, skl, df = getdatastr(dataname)

# print(tgl)
# print(skl)
minedge = 8
# 初始网络
sstart = time.time()
n = len(tgl)
pcg = np.zeros((n, n), dtype=int)
totaltime = 0
avgtime = 0
maxtime = 0
# 联邦环境下学习初始网络
print("初始网络构造阶段")
for i in range(fed_count):
    start = time.time()
    #
    pcs = fedPC(tgl, dataname, i, alpha)
    end = time.time()
    ti =  end - start
    totaltime = totaltime + ti
    maxtime = max(maxtime, ti)
    pcg = pcg + pcs
    #pcg = np.maximum(pcs, pcg)
avgtime = totaltime / fed_count
print(fed_count, "个网络总耗时：", totaltime, "最大客户端耗时：", maxtime, "平均耗时：", avgtime)
pcg = np.where(pcg >= minedge/2, 1, 0)
# pcg = np.maximum(pcg, tgl)

show(toGx(pcg), G, 1, 1, name='pcG')
# 约束划分（服务器上
sccs = ancestralPartition(toGx(pcg))
for si in sccs:
    pcg = remove_edges(pcg, si)

# 各客户端上学习
tctime=0
mctime=0
pgt = np.zeros((n, n), dtype=int)
for i in range(fed_count):
    start = time.time()
    pg = fedlearn(skl, sccs, dataname, i, tgl)
    end = time.time()
    print("第", i+1, "个客户端用时：", end-start)
    tctime = tctime + end-start
    mctime = max(mctime, end-start)
    pgt = pgt + pg

# 合并
pgt = np.where(pgt >= minedge, 1, 0)
pcg = np.maximum(pgt, pcg)
eend = time.time()

print("理论耗时：", eend-sstart-totaltime+maxtime-tctime+mctime)
show(toGx(pcg), G, sstart, eend, name='fed_casual')
