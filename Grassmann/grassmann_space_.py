################################################################################
# 本文件中是格拉斯曼流形上的核函数
################################################################################
# 导入模块
import json
import numpy as np
from scipy.linalg import svd
from scipy.linalg import eigh
from pathlib import Path
################################################################################
class GrassmannSubSpace:
    """
    将原始的图像集建模到格拉斯曼流形上
    [1] [1]王锐. 基于黎曼流形学习的图像集分类算法研究[D]. 江南大学, 2022. (22页, 图2.5)
    """
    def __init__(self):
        pass

    def orthogonal_subspace(self, data):
        """
        对格拉斯曼数据集进行QR分解
        :param data: [N, D, p]
        :return: [N, D, p]
        """
        D = []
        for d in data:
            q, _ = np.linalg.qr(d)
            D.append(q)
        return np.array(D)

    def compute_subspace(self, data, p=10):
        """
        对图像集进行SVD分解
        :param data: 图像集合列表
        :param p: 子空间阶数
        :return: 格拉斯曼数据集 [N, D, p]
        """
        sub = []
        for d in data:
            U, _, _ = svd(np.dot(d, d.T))
            sub.append(U[:, :p])
        return np.array(sub)
################################################################################
class GrassmannDimensionality:
    """
    计算低维格拉斯曼流形的维度
    在降维任务中，通过累积贡献率计算低维投影的维度
    """
    def __init__(self, ratio=0.95):
        """
        初始化函数
        :param ratio: 累积贡献率
        """
        self.ratio = ratio

    def stack(self, data):
        """
        计算样本集的均值
        :param data: [N, D, p]
        :return:
        """
        s = np.zeros((data[0].shape[0], data[0].shape[0])).astype(np.float32)
        for d in data:
            s += np.dot(d, d.T)
        s /= data.shape[0]
        return s

    def determine_dimensionality(self, data):
        """
        确定低维维度
        :param data: [N, D, p]
        :return: 低维维度 (int)
        """
        s = self.stack(data)
        values = eigh(s, eigvals_only=True)[::-1]
        total_variance = np.sum(values)
        explained_variance_ratio = values / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        self.components = int(np.argmax(cumulative_variance_ratio >= self.ratio) + 1)
        return self.components

    def save_low_dimensions(self, data_name):
        """
        在创建数据集时将数据集的维度存储在指定文件中
        :param data_name: 数据集的名字
        :return:
        """
        root = Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]
        leaf = ["DATA", "GRASSMANN", "Grassmann_data_paras.json"]
        root = list(root) + leaf
        json_path = "/".join(root)
        with open(json_path, 'r', encoding='utf-8') as paras:
            grassmann_paras = json.load(paras)
        paras.close()
        if grassmann_paras["low_dimensions"].get(data_name) is None:
            grassmann_paras["low_dimensions"][data_name] = dict()
        if not isinstance(grassmann_paras["low_dimensions"][data_name], dict):
            grassmann_paras["low_dimensions"][data_name] = dict()
        grassmann_paras["low_dimensions"][data_name][self.ratio] = self.components
        with open(json_path, 'w') as paras:
            json.dump(grassmann_paras, paras, indent=4)
        paras.close()
