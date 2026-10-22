


# 查找有无为定向的边
def getUN(G, skl):
    edges = []

    # 检查邻接矩阵的维度是否相同
    if len(G) != len(skl) or len(G[0]) != len(skl[0]):
        print("Error: 邻接矩阵维度不相同")
        return

    # 遍历邻接矩阵
    for i in range(len(G)):
        for j in range(len(G[0])):
            # 如果在G1中存在边而在G2中不存在
            if G[i][j] == 1 and skl[i][j] == 0:
                edges.append((i, j))

    return edges


# 找隐变量
# def getCon(G, edges):
