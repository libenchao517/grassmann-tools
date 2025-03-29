################################################################################
# 本文件用于实现Grassmann Adaptive Local Learning及相关算法
################################################################################
# 导入模块
import scipy
import numpy as np
from sklearn.metrics import pairwise_distances
from Grassmann import GrassmannDistance
from Grassmann import GrassmannKernel
################################################################################
class GrassmannALL:
    """
    Grassmann Adaptive Local Learning
    [1] Wei D, Shen X, Sun Q, et al.
    Learning adaptive Grassmann neighbors for image-set analysis[J].
    Expert Systems with Applications, 2024, 247: 123316.
    """
    def __init__(
            self,
            n_components=76,
            p_grassmann=10,
            n_neighbors=5,
            train_size=5,
            random_state=517,
            converged_tol=1.0,
            drop_tol=1e-6,
            max_epoch=20,
            verbose=False
    ):
        """
        初始化函数
        :param n_components: 低维维度
        :param p_grassmann: 格拉斯曼子空间维度
        :param n_neighbors: 近邻数
        :param train_size: 训练数据比例
        :param random_state: 随机种子
        :param converged_tol: 迭代终止条件
        :param max_epoch: 最大迭代次数
        :param verbose: 可视化标志
        """
        self.n_components = n_components
        self.p_grassmann = p_grassmann
        self.n_neighbors = n_neighbors
        self.train_size = train_size
        self.random_state = random_state
        self.converged_tol = converged_tol
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.GD = GrassmannDistance()         # 初始化格拉斯曼度量
        self.GK = GrassmannKernel()           # 初始化格拉斯曼核函数
        self.metric = self.GD.f_norm_square   # 确定度量方式
        self.object_value = np.array([])      # 初始化损失列表

    def _init_p(self, data):
        """
        初始化投影矩阵
        :param data: 原始数据将矩阵 [N, D, p]
        :return: 投影矩阵 P [D, d]
        """
        p = np.eye(data.shape[1])
        return p[:, :self.n_components]

    def _update_p(self, h):
        """
        更新投影矩阵，论文中公式（21）
        :param h:
        :return:
        """
        eig_values, eig_vectors = scipy.linalg.eigh(h)
        sort_index_ = np.argsort(eig_values)
        index_ = sort_index_[:self.n_components]
        return eig_vectors[:, index_]

    def _init_w(self, data, target):
        """
        初始化权重矩阵
        :param data:
        :param target:
        :return:
        """
        W = np.zeros((1, 1))
        for t in np.unique(target):
            data_t = data[t == target]
            dist_t = self.GD.pairwise_dist(data_t, self.metric)
            dist_t = np.exp(-dist_t)
            W = self._diag(W, dist_t)
        W = W[1:, 1:]
        W_sum = np.tile(np.sum(W, axis=0), (W.shape[0], 1))
        return W / W_sum

    def _update_w(self, data, w, p, f, lambd=100):
        """
        更新权重矩阵，论文中公式（27）
        :param data:
        :param w:
        :param p:
        :param f:
        :param lambd:
        :return:
        """
        w_updated = np.zeros_like(w)
        alpha = np.zeros(data.shape[0])
        ita = np.zeros(data.shape[0])
        proj = self.transform(data, p)
        u = self.GD.pairwise_dist(proj, self.metric)
        v = np.square(pairwise_distances(f))
        z = u + lambd * v
        for i in range(data.shape[0]):
            alpha[i] = (self.n_neighbors * z[i, self.n_neighbors] - np.sum(z[i, :self.n_neighbors])) / 2
            ita[i] = 1/self.n_neighbors + np.sum(z[i, :self.n_neighbors]) / (2*self.n_neighbors*alpha[i])
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                w_updated[i, j] = ita[i] - z[i, j] / (2 * alpha[i])
        w_updated[w_updated < 0] = 0.001
        return w_updated

    def _init_f(self, w):
        """
        初始化F
        :param w:
        :return:
        """
        L = np.diag(np.sum(w, axis=1)) - (w + np.transpose(w)) / 2
        eig_values, eig_vectors = scipy.linalg.eigh(L)
        self.f_c = len(eig_values[eig_values < 1e-6])
        sort_index_ = np.argsort(eig_values)
        index_ = sort_index_[:self.f_c]
        return eig_vectors[:, index_]

    def _update_f(self, w):
        """
        更新F，论文中公式（23）
        :param w:
        :return:
        """
        L = np.diag(np.sum(w, axis=1)) - (w + np.transpose(w)) / 2
        eig_values, eig_vectors = scipy.linalg.eigh(L)
        sort_index_ = np.argsort(eig_values)
        index_ = sort_index_[:self.f_c]
        return eig_vectors[:, index_]

    def _update_h(self, data, w, p):
        """
        论文中公式（21）
        :param data:
        :param w:
        :param p:
        :return:
        """
        H = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                G = np.dot(data[i], np.transpose(data[i])) - np.dot(data[j], np.transpose(data[j]))
                H = w[i, j] * np.dot(np.dot(np.dot(G, p), np.transpose(p)), G)
        return H

    def _diag(self, a, b):
        """
        拼接矩阵a和b
        :param a:
        :param b:
        :return: [[a, 0], [0, b]]
        """
        return np.block([[a, np.zeros((a.shape[0], b.shape[1]))], [np.zeros((b.shape[0], a.shape[1])), b]])

    def _orthogonal_subspace(self, data, p):
        """
        对样本进行标准化
        :param data: 原始数据
        :param p:    投影矩阵
        :return:     标准化后的数据
        """
        D = []
        for d in data:
            q, r = np.linalg.qr(np.dot(np.transpose(p), d))
            D.append(np.dot(d, np.linalg.inv(r)))
        return np.array(D)

    def transform(self, data, p):
        """
        降维函数
        :param data: 高维样本
        :param p:    投影矩阵
        :return:     低维投影
        """
        embedding = []
        for d in data:
            embedding.append(np.dot(np.transpose(p), d))
        return np.array(embedding)

    def optimization(self, data, w, p, f, lambd=1000):
        """
        优化函数
        :param data:
        :param w:
        :param p:
        :param f:
        :param lambd:
        :return:
        """
        data = self._orthogonal_subspace(data, p)  # 标准化数据
        w = self._update_w(data, w, p, f, lambd)
        f = self._update_f(w)
        h = self._update_h(data, w, p)
        p = self._update_p(h)
        ob = np.trace(np.dot(np.dot(np.transpose(p), h), p))
        return data, w, p, f, ob

    def fit(self, data, target):
        """
        训练过程
        :param data:   训练样本
        :param target: 训练标签
        :return: GALL对象
        """
        self.coms = []
        p = self._init_p(data)
        w = self._init_w(data, target)
        f = self._init_f(w)
        self.coms.append(p)
        epoch = 0
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            data, w, p, f, current_object = self.optimization(data, w, p, f)
            self.coms.append(p)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            if epoch >= self.max_epoch:
                break
        return self

    def fit_transform(self, data_train, data_test, target_train, target_test):
        """
        主函数
        :param data_train: 训练样本
        :param data_test:  测试样本
        :param target_train: 训练集标签
        :param target_test:  测试集标签
        :return: 训练集的投影，测试集的投影
        """
        self.fit(data_train, target_train)
        for a in self.coms:
            data_train = self._orthogonal_subspace(data_train, a)
            data_test = self._orthogonal_subspace(data_test, a)
        embedding_train = self.transform(data_train, a)
        embedding_test = self.transform(data_test, a)
        return embedding_train, embedding_test
