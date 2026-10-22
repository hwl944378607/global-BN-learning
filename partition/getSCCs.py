# tarjan
import numpy as np


class SCCs:
    S = []
    Nn = 0
    matrixMB = [[]]
    tableMB1 = {}
    tableMB2 = {}
    start = '1'

    def init(self, MB):
        self.matrixMB = MB
        self.tableMB1, self.tableMB2, self.start = matrix2table(MB)
        self.Nn = len(MB)

    # 返回最大尺寸
    def MaxSCCSize(self):
        maxsize = 0
        for scc in self.S:
            if len(scc) > maxsize:
                maxsize = len(scc)
        return maxsize

    def findSCCs(self):
        sccs = self.SCC('1')
        print(self.start)
        self.S = []
        for scc in sccs:
            temp = []
            for key, val in scc.items():
                temp.append(key)
            self.S.append(temp)
        return self.S

    # 打印强连通分量
    def printSCCs(self):
        for scc in self.S:
            print('******')
            for i in range(len(scc)):
                print(scc[i])

    # 返回SCCs
    def re_tr(self, G):
        GT = {}
        for u in G:
            for v in G[u]:
                # print(GT)
                if GT.get(v):
                    GT[v].add(u)
                else:
                    GT[v] = set()
                    GT[v].add(u)

        for i in range(self.Nn):
            if GT.get(str(i + 1), 0) == 0:
                GT.setdefault(str(i + 1), None)

        return GT

    # 递归实现深度优先排序
    def rec_dfs(self, G, s, S=None):
        if S is None:
            #S = set()    # 集合存储已经遍历过的节点
            S = list()  # 用列表可以更方便查看遍历的次序，而用集合可以方便用difference求差集
        # S.add(s)
        S.append(s)
        #print(S)
        for u in G[s]:
            if u in S: continue
            self.rec_dfs(G, u, S)

        return S

    # 遍历有向图的强连通分量
    def walk(self, G, start, S=set()):  # 传入的参数S，即上面的seen很关键，这避免了通过连通图之间的路径进行遍历
        P, Q = dict(), set()  # list存放遍历顺序，set存放已经遍历过的节点
        P[start] = None
        Q.add(start)
        while Q:
            u = Q.pop()  # 选择下一个遍历节点（随机性）

            diff = set()
            if G[u] is not None:
                diff = G[u].difference(P, S)

            print("G:{}, u:{}".format(G,u))
            print("G[u]:{}, P:{}, S:{}".format(G[u], P, S))
            print('差集:{}'.format(diff))

            for v in diff:  # 返回差集
                Q.add(v)
                P[v] = u
        #print(P)
        return P

    # 获得各个强连通图
    def SCC(self, start):
        G=self.tableMB1
        GT = self.re_tr(G)
        sccs, seen = [], set()
        for u in self.rec_dfs(G, start):  # 以a为起点
            if u in seen: continue
            C = self.walk(GT, u, seen)
            seen.update(C)
            sccs.append(C)
        return sccs


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


# 邻接矩阵转邻接表
def matrix2table(matric):
    result = {}
    result2 = {}
    N = len(matric)
    start = '0'
    for i in range(N):
        for j in range(N):
            result.setdefault(str(i + 1), [])
            result2.setdefault(str(i + 1), [])
            if matric[i][j]:
                start = str(i)
                result.setdefault(str(i+1),[]).append(str(j+1))
                result2.setdefault(str(j+1), []).append(str(i+1))


    """
    c = 0
    for column in range(matric.shape[1]):
        for i in range(column):
            count = 0
            if column[i]:
                result.setdefault(str(i+1),[]).append(str(c+1))
                count = count + 1
        if count == 0:
            result.setdefault(str(c+1),[])
        c = c + 1
        
    for i in range(N):
        if result2.get(str(i + 1), 0) == 0:
            result2.setdefault(str(i + 1), [])
    """



    # result2 = dict(sorted(result2.items(), key=lambda x: x[0]))

    return result,result2,start


def matrix2table2(matric):
    result = {}
    N = len(matric)
    for i in range(N):
        count = 0
        for j in range(N):
            if matric[i][j]:
                result.setdefault(str(i+1),[]).append(str(j+1))
                count = count+1
        if count == 0:
            result.setdefault(str(i+1),[])
    return result

