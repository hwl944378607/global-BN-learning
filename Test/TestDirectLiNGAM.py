import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased import lingam


class TestDirectLiNGAM(unittest.TestCase):

    def test_DirectLiNGAM(self):

        # 替换文件路径为你的实际路径
        file_path = 'C:\Users\Guo\PycharmProjects\global-BN-learning\datas\Alarm1_s5000_v1.txt'

        # 使用pd.read_csv读取文本文件
        df = pd.read_csv(file_path, sep=' ', header=None)

        # 为DataFrame添加列名
        columns = ['x1', 'x2', 'x3', 'x4', 'x5']
        df.columns = columns

        # 打印DataFrame
        print(df)
