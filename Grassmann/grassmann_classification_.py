################################################################################
# 本文件中是格拉斯曼流形上的分类器
################################################################################
# 导入模块
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from Grassmann import GrassmannKernel
from Grassmann import GrassmannDistance
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder
################################################################################
class GrassmannKNN:
    """
    格拉斯曼流形上的最近邻分类器
    """
    def __init__(self, n_neighbors=1):
        """
        初始化函数
        :param n_neighbors: 邻居数
        """
        self.GD = GrassmannDistance()                             # 格拉斯曼度量生成器
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # 初始化最近邻分类器
        self.metric = self.GD.projection_metric                   # 定义格拉斯曼度量

    def fit(self, X, Y=None):
        """
        训练分类器
        :param X: 训练样本
        :param Y: 训练标签
        :return:
        """
        self.data_train = X
        dist = self.GD.pairwise_dist(X, self.metric)
        self.knn.fit(dist.T, Y)

    def predict(self, data_test):
        """
        对样本进行预测
        :param data_test: 测试样本
        :return: 预测的标签
        """
        dist = self.GD.non_pair_dist(self.data_train, data_test, self.metric)
        self.t_pred = self.knn.predict(dist.T)
        return self.t_pred
################################################################################
class GrassmannSVM(svm.SVC):
    """
    格拉斯曼流形上的支持向量机
    [1] Al-Samhi W, Al-Soswa M, Al-Dhabi Y.
    Time series data classification on grassmann manifold[C].
    Journal of Physics: Conference Series. IOP Publishing, 2021, 1848(1): 012037.
    [2] https://github.com/adamguos/arma-grassmann-classifier
    """
    def __init__(self, gamma=0.2):
        """
        初始化函数
        :param gamma: 高斯核函数的超参数
        """
        self.gamma = gamma
        self.GK = GrassmannKernel(gamma = gamma)
        kernel = lambda X, Y : self.GK.non_pair_kernel(X, Y, self.GK.gaussian_projection_kernel)
        super().__init__(kernel=kernel)

    def transform(self, X):
        """
        对测试样本进行预测
        :param X: 测试样本
        :return: 预测的标签
        """
        return super().predict(X)
################################################################################
class GrassmannKernelFDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    格拉斯曼流形上的核判别分析
    根据[2]进行实现
    [1] Hamm J, Lee D D.
    Grassmann discriminant analysis: a unifying view on subspace-based learning[C].
    Proceedings of the 25th international conference on Machine learning. 2008: 376-383.
    [2] https://github.com/concavegit/kfda/
    """
    def __init__(self, n_components=20, kernel=None, robustness_offset=1e-8):
        """
        初始化函数
        :param n_components:       低维维度
        :param kernel:             度量方式
        :param robustness_offset:
        """
        self.n_components = n_components
        self.robustness_offset = robustness_offset
        self.GK = GrassmannKernel()
        self.kernel = self.GK.projection_kernel if kernel is None else kernel

    def fit(self, X, y):
        """
        训练过程
        :param X: 训练样本
        :param y: 训练标签
        :return:
        """
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        y_onehot = OneHotEncoder().fit_transform(self.y_[:, np.newaxis])
        K = self.GK.pairwise_kernel(X, self.GK.projection_kernel)
        m_classes = y_onehot.T @ K / y_onehot.T.sum(1)
        indices = (y_onehot @ np.arange(self.classes_.size)).astype('i')
        N = K @ (K - m_classes[indices])
        N += eye(self.y_.size) * self.robustness_offset
        m_classes_centered = m_classes - K.mean(1)
        M = m_classes_centered.T @ m_classes_centered
        w, self.weights_ = eigsh(M, self.n_components, N, which='LM')
        return self

    def transform(self, X):
        """
        对测试样本进行降维
        :param X: 测试样本
        :return:
        """
        K = self.GK.non_pair_kernel(X, self.X_, self.GK.projection_kernel)
        return K @ self.weights_
################################################################################
class GrassmannKernelRDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    格拉斯曼流形上的正则化判别分析
    根据[2]进行实现
    [1] Hamm J, Lee D D.
    Grassmann discriminant analysis: a unifying view on subspace-based learning[C].
    Proceedings of the 25th international conference on Machine learning. 2008: 376-383.
    [2] https://github.com/daviddiazvico/scikit-kda
    """
    def __init__(self, kernel=None, lmb=0.001):
        """
        初始化函数
        :param kernel:  度量方式
        :param lmb:     超参数
        """
        self.lmb = lmb
        self.GK = GrassmannKernel()
        self.kernel = self.GK.projection_kernel if kernel is None else kernel

    def fit(self, X, y):
        """
        训练过程
        :param X: 训练数据
        :param y: 训练标签
        :return:
        """
        n = len(X)
        self._X = X
        self._H = np.identity(n) - 1 / n * np.ones(n) @ np.ones(n).T
        self._E = OneHotEncoder().fit_transform(y.reshape(n, 1))
        _, counts = np.unique(y, return_counts=True)
        K = self.GK.pairwise_kernel(X, self.kernel)
        C = self._H @ K @ self._H
        self._Delta = np.linalg.inv(C + self.lmb * np.identity(n))
        A = self._E.T @ C
        B = self._Delta @ self._E
        self._Pi_12 = np.diag(np.sqrt(1.0 / counts))
        P = self._Pi_12 @ A
        Q = B @ self._Pi_12
        R = P @ Q
        V, self._Gamma, self._U = np.linalg.svd(R, full_matrices=False)
        return self

    def transform(self, X):
        """
        对测试数据进行降维
        :param X: 测试数据
        :return:
        """
        _K = self.GK.non_pair_kernel(X, self._X, self.kernel)
        K = _K - np.mean(_K, axis=0)
        C = self._H @ K.T
        T = self._U @ self._Pi_12 @ self._E.T @ self._Delta
        Z = T @ C
        return Z.T
