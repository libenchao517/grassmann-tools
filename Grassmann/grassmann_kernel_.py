################################################################################
# 本文件中是格拉斯曼流形上的核函数
################################################################################
# 导入模块
import numpy as np
from Grassmann import GrassmannDistance
################################################################################
class GrassmannKernel:
    """
    格拉斯曼流形上的核函数
    [1] Shigenaka R, Raytchev B, Tamaki T, et al.
    Face sequence recognition using Grassmann distances and Grassmann kernels[C].
    The 2012 international joint conference on neural networks (IJCNN). IEEE, 2012: 1-7.
    [2] Jayasumana S, Hartley R, Salzmann M, et al.
    Kernel methods on Riemannian manifolds with Gaussian RBF kernels[J].
    IEEE transactions on pattern analysis and machine intelligence, 2015, 37(12): 2464-2477.
    """
    def __init__(self, gamma=0.2):
        """
        初始化函数
        :param gamma: 高斯投影核的超参数
        """
        self.gamma = gamma

    def pairwise_kernel(self, x, kernel):
        """
        计算成对核矩阵
        :param x: 样本集
        :param kernel: 核函数
        :return: 核矩阵 [N, N]
        """
        n = len(x)
        Kmatrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                Kmatrix[i, j] = Kmatrix[j, i] = kernel(x[i], x[j])
        return Kmatrix

    def non_pair_kernel(self, x, y, kernel):
        """
        计算非对称核矩阵
        :param x: 样本集1 [m, D, p]
        :param y: 样本集2 [n, D, p]
        :param kernel: 核函数
        :return: 核矩阵 [m, n]
        """
        if np.array_equal(x, y):
            return self.pairwise_kernel(x, kernel)
        m = len(x)
        n = len(y)
        Kmatrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                Kmatrix[i, j] = kernel(x[i], y[j])
        return Kmatrix

    def projection_kernel(self, a, b):
        """
        Projection Kernel [1]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        return np.square(np.linalg.norm(np.dot(a.T, b), ord='fro'))

    def binet_cauchy_kernel(self, a, b):
        """
        Binet Cauchy Kernel [1]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        return np.prod(np.square(costheta))

    def gaussian_projection_kernel(self, a, b):
        """
        Projection Gaussian Kernel [2]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        GD = GrassmannDistance()
        distsquare = GD.projection_metric_square(a, b)
        return np.exp(-self.gamma * distsquare)
