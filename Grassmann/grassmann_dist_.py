################################################################################
# 本文件中是格拉斯曼流形上的度量方式
################################################################################
# 导入模块
import numpy as np
################################################################################
class GrassmannDistance:
    """
    格拉斯曼流形上的距离
    [1] Hamm J, Lee D D.
    Grassmann discriminant analysis: a unifying view on subspace-based learning[C].
    Proceedings of the 25th international conference on Machine learning. 2008: 376-383.
    [2] Lui Y M, Beveridge J R, Draper B A, et al.
    Image-set matching using a geodesic distance and cohort normalization[C].
    2008 8th IEEE International Conference on Automatic Face & Gesture Recognition. IEEE, 2008: 1-6.
    [3] Jayasumana S, Hartley R, Salzmann M, et al.
    Kernel methods on Riemannian manifolds with Gaussian RBF kernels[J].
    IEEE transactions on pattern analysis and machine intelligence, 2015, 37(12): 2464-2477.
    [4] Harandi M, Sanderson C, Shen C, et al.
    Dictionary learning and sparse coding on Grassmann manifolds: An extrinsic solution[C].
    Proceedings of the IEEE international conference on computer vision. 2013: 3120-3127.
    [5] Wei D, Shen X, Sun Q, et al.
    Neighborhood preserving embedding on Grassmann manifold for image-set analysis[J].
    Pattern Recognition, 2022, 122: 108335.
    [6] Shigenaka R, Raytchev B, Tamaki T, et al.
    Face sequence recognition using Grassmann distances and Grassmann kernels[C].
    The 2012 international joint conference on neural networks (IJCNN). IEEE, 2012: 1-7.
    """
    def __init__(self):
        pass

    def pairwise_dist(self, x, metric):
        """
        计算成对距离矩阵
        :param x: 样本集
        :param metric: 度量方式
        :return: 距离矩阵 [N, N]
        """
        n = len(x)
        distance = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distance[i, j] = distance[j, i] = metric(x[i], x[j])
        return distance

    def non_pair_dist(self, x, y, metric):
        """
        计算非对称距离矩阵
        :param x: 样本集1 [m, D, p]
        :param y: 样本集2 [n, D, p]
        :param metric: 度量方式
        :return: 距离矩阵 [m, n]
        """
        if np.array_equal(x, y):
            return self.pairwise_dist(x, metric)
        m = len(x)
        n = len(y)
        distance = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                distance[i, j] = metric(x[i], y[j])
        return distance

    def f_norm_square(self, a, b):
        """
        Square F-Norm Distance / Square Projection Metric [3]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        return np.square(np.linalg.norm(np.dot(a, a.T) - np.dot(b, b.T))) / 2

    def gdist(self, a, b):
        """
        Projection Metric / Embedding Distance [4][5]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        return np.linalg.norm(np.dot(a, a.T) - np.dot(b, b.T)) / np.sqrt(2)

    def projection_metric(self, a, b):
        """
        Projection Metric [1]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = a.shape[1] - np.sum(np.square(costheta))
        return np.sqrt(dist)

    def projection_metric_square(self, a, b):
        """
        Square Projection Metric [3]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        return a.shape[1] - np.square(np.linalg.norm(np.dot(a.T, b)))

    def binet_cauchy(self, a, b):
        """
        Binet Cauchy Metric [1]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = 1 - np.prod(np.square(costheta))
        return np.sqrt(dist)

    def max_correlation(self, a, b):
        """
        Max Correlation [1]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.max(1 - np.square(costheta))
        return np.sqrt(dist)

    def min_correlation(self, a, b):
        """
        Min Correlation [1]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.min(1 - np.square(costheta))
        return np.sqrt(dist)

    def chordal_distance_fro(self, a, b):
        """
        Chordal Distance using F-norm [6]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sqrt(np.sum((1 - costheta) / 2)) * 2
        return dist

    def chordal_distance_2m(self, a, b):
        """
        Chordal Distance using 2-norm [6]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sqrt((1 - np.min(costheta)) / 2) * 2
        return dist

    def geodesic_distance(self, a, b):
        """
        Geodesic Distance [2]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sum(np.arccos(costheta) ** 2)
        return dist

    def mean_distance(self, a, b):
        """
        Mean Distance [6]
        :param a:
        :param b:
        :return:
        """
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sum(1 - costheta ** 2) / a.shape[1]
        return dist
