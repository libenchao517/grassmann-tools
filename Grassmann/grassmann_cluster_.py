################################################################################
# 本文件中是格拉斯曼流形上的聚类方法
################################################################################
# 导入模块
import scipy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
from Grassmann import GrassmannDistance
from Grassmann import GrassmannKernel
################################################################################
class GrassmannKMeans:
    """
    格拉斯曼流形上的快速均值聚类
    代码来源于[2]
    [1] Stiverson S J.
    An Adaptation of K-Means-Type Algorithms to The Grassmann Manifold[D].
    Colorado State University, 2019.
    [2] https://github.com/sjstiver/Grassmannian_clustering
    """
    def __init__(
            self,
            center_select="data",
            eps=10e-6,
            center_count=5,
            n_epoch=10,
            mode="return_label"
    ):
        """
        初始化函数
        :param center_select: 选择聚类中心的方法
        :param eps: 允许的最小误差
        :param center_count: 聚类数量
        :param n_epoch: 最大迭代次数
        :param mode: 结果返回模式
        """
        self.center_select = center_select
        self.eps = eps
        self.distortion_change = []
        self.center_count = center_count
        self.n_epoch = n_epoch
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro
        self.mode = mode

    def recalculate_centers(self, C, X, t):
        """
        重新计算簇的中心
        :param C:
        :param X:
        :param t:
        :return:
        """
        U, S, Vh = np.linalg.svd(np.dot(np.dot(np.eye(C.shape[0]) - np.dot(C, C.T), X), np.linalg.pinv(np.dot(C.T, X))), full_matrices=False)
        Y = np.dot(C, np.dot(Vh.T, np.diag(np.cos(np.arctan(S) * t)))) + np.dot(U, np.diag(np.sin(np.arctan(S) * t)))
        Q = np.linalg.qr(Y)[0]
        return Q

    def cluster_distortion(self, dist, labels, center_count):
        """
        计算平均距离
        :param dist: 距离矩阵
        :param labels: 聚类标签
        :param center_count: 类别数量
        :return:
        """
        cluster_dist = []
        for i in range(center_count):
            idx = (labels == i).nonzero()
            cluster_dist.append(dist[i, idx].mean())
        return np.mean(cluster_dist)

    def fit(self, data):
        """
        训练过程
        :param data: 数据集
        :return: 聚类中心，标签
        """
        # 初始化聚类中心
        if self.center_select.lower() == "data":
            centers = data[np.random.choice(data.shape[0], self.center_count, replace=False)]
        elif self.center_select.lower() == "random":
            centers = []
            for i in range(self.center_count):
                centers.append(np.linalg.qr(np.random.random(data[0].shape))[0])
            centers = np.array(centers)
        else:
            print("Center selection algorithm is invalid.")
            return
        # 更新聚类中心
        count = 0
        dist = self.GD.non_pair_dist(centers, data, self.metric)
        labels = np.argmin(dist, axis=0)
        avg_dist = self.cluster_distortion(dist, labels, self.center_count)
        self.distortion_change.append(avg_dist)
        delta = 1
        n = np.zeros((1, self.center_count))[0]
        while count < self.n_epoch and delta > self.eps:
            for i in range(data.shape[0]):
                dist = self.GD.non_pair_dist(centers, data[i].reshape((1, data.shape[1], data.shape[2])), self.metric)
                label = np.argmin(dist, axis=0)[0]
                n[label] += 1
                centers[label] = self.recalculate_centers(centers[label], data[i], 1/(n[label]))
            dist = self.GD.non_pair_dist(centers, data, self.metric)
            labels = np.argmin(dist, axis=0)
            avg_dist = self.cluster_distortion(dist, labels, self.center_count)
            delta = (self.distortion_change[-1]-avg_dist)/avg_dist
            self.distortion_change.append(avg_dist)
            count += 1
        dist = self.GD.non_pair_dist(centers, data, self.metric)
        labels = np.argmin(dist, axis=0)
        avg_dist = self.cluster_distortion(dist, labels, self.center_count)
        self.distortion_change.append(avg_dist)
        return centers, labels

    def fit_transform(self, data):
        """
        主函数
        :param data: 数据集
        :return:
        """
        centers, labels = self.fit(data)
        if self.mode == "return_label":
            return labels
        elif self.mode == "return_center":
            return centers
        elif self.mode == "all":
            return centers, labels

class GrassmannLBG:
    """
    格拉斯曼流形上的LBG聚类
    [1] Stiverson S J.
    An Adaptation of K-Means-Type Algorithms to The Grassmann Manifold[D].
    Colorado State University, 2019.
    [2] https://github.com/sjstiver/Grassmannian_clustering
    """
    def __init__(
            self,
            center_select="data",
            eps=10e-6,
            center_count=5,
            n_epoch=10,
            mode="return_label"
    ):
        """
        初始化函数
        :param center_select: 选择聚类中心的方法
        :param eps: 允许的最小误差
        :param center_count: 聚类数量
        :param n_epoch: 最大迭代次数
        :param mode: 结果返回模式
        """
        self.center_select = center_select
        self.eps = eps
        self.distortion_change = []
        self.center_count = center_count
        self.n_epoch = n_epoch
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro
        self.mode = mode

    def flag_mean(self, X, r=None):
        """
        计算外部均值
        :param X: 样本集
        :param r: 格拉斯曼流形的子空间阶数
        :return:
        """
        if r is None:
            r = X[0].shape[1]
        A = X[0]
        for i in range(len(X) - 1):
            A = np.hstack((A, X[i + 1]))
        U = np.linalg.svd(A, full_matrices=False)[0]
        return U[:, :r]

    def cluster_distortion(self, dist, labels, center_count):
        """
        计算平均距离
        :param dist: 距离矩阵
        :param labels: 聚类标签
        :param center_count: 类别数量
        :return:
        """
        cluster_dist = []
        for i in range(center_count):
            idx = (labels == i).nonzero()
            cluster_dist.append(dist[i, idx].mean())
        return np.mean(cluster_dist)

    def init_cemters(self, data):
        """
        初始化聚类中心
        :param data: 数据集
        :return:
        """
        if self.center_select.lower() == "data":
            centers = data[np.random.choice(data.shape[0], self.center_count, replace=False)]
        elif self.center_select.lower() == "random":
            centers = []
            for i in range(self.center_count):
                centers.append(np.linalg.qr(np.random.random(data[0].shape))[0])
            centers = np.array(centers)
        else:
            print("Center selection algorithm is invalid.")
            return
        return centers

    def recalculate_label(self, data, centers):
        """
        计算标签
        :param data: 数据集
        :param centers: 策中心的列表
        :return:
        """
        dist = self.GD.non_pair_dist(centers, data, self.metric)
        labels = np.argmin(dist, axis=0)
        avg_dist = self.cluster_distortion(dist, labels, self.center_count)
        self.distortion_change.append(avg_dist)
        return labels

    def calculate_center(self, data, labels):
        """
        更新簇中心列表
        :param data: 数据集
        :param labels: 聚类标签
        :return:
        """
        centers = []
        for i in range(self.center_count):
            cluster_subset = data[labels == i]
            if cluster_subset.shape[0] != 0:
                centers.append(self.flag_mean(cluster_subset))
        centers = np.array(centers)
        return centers

    def fit(self, data):
        """
        训练过程
        :param data: 数据集
        :return:
        """
        # 初始化簇中心
        centers = self.init_cemters(data)
        labels = self.recalculate_label(data, centers)
        count = 0
        delta = 1
        # 更新簇中心
        while count < self.n_epoch and delta > self.eps:
            centers = self.calculate_center(data, labels)
            labels = self.recalculate_label(data, centers)
            count += 1
            delta = np.abs(self.distortion_change[-2] - self.distortion_change[-1]) / self.distortion_change[-1]
        return centers, labels

    def fit_transform(self, data):
        """
        主函数
        :param data: 数据集
        :return:
        """
        flag = True
        while flag:
            try:
                centers, labels = self.fit(data)
                flag = False
            except:
                print("重试！")
                flag = True
        if self.mode == "return_label":
            return labels
        elif self.mode == "return_center":
            return centers
        elif self.mode == "all":
            return centers, labels

class CGMKE:
    """
    Clustering on Grassmann Manifold via Kernel Embedding
    [1] Shirazi S, Harandi M T, Sanderson C, et al.
    Clustering on Grassmann manifolds via kernel embedding with application to action analysis[C].
    2012 19th IEEE international conference on image processing. IEEE, 2012: 781-784.
    """
    def __init__(self, center_count=5):
        """
        初始化函数
        :param center_count: 簇的数量
        """
        self.center_count = center_count
        self.KM = KMeans(n_clusters=center_count)   # 初始化欧氏空间中的K-Means
        self.GK = GrassmannKernel()                 # 初始化格拉斯曼流形上的核函数
        self.kernel = self.GK.projection_kernel     # 投影核函数

    def top_eigenvectors(self, K, n_components):
        """
        计算最大的特征值对应的特征向量
        :param K: 样本集
        :param n_components: 特征向量的个数
        :return:
        """
        eig_values, eig_vectors = scipy.linalg.eigh(K)
        sort_index_ = np.argsort(eig_values)[::-1]
        index_ = sort_index_[: n_components]
        return eig_vectors[:, index_]

    def trans_data(self, data):
        """
        将格拉斯曼流形上的样本投影到核空间中
        :param data: 样本集
        :return:
        """
        Kmetrix = self.GK.pairwise_kernel(data, self.kernel)
        D = np.diag(np.power(np.sum(Kmetrix, axis=1), -0.5))
        K = np.dot(np.dot(D, Kmetrix), D)
        n_components = self.center_count - 1 if self.center_count > 2 else self.center_count
        U = self.top_eigenvectors(K, n_components)
        return U

    def fit(self, data):
        """
        训练过程
        :param data: 数据集
        :return:
        """
        kernel_data = self.trans_data(data)
        self.KM.fit(kernel_data)
        return self.KM.labels_

    def fit_transform(self, data):
        """
        主函数
        :param data: 数据集
        :return:
        """
        labels = self.fit(data)
        return labels

class Shared_Nearest_Neighbor_DPC:
    """
    Shared Nearest Neighbor Density Peaks Clustering
    代码来源于[2]
    [1] Liu R, Wang H, Yu X.
    Shared-nearest-neighbor-based clustering by fast search and find of density peaks[J].
    information sciences, 2018, 450: 200-226.
    [2] https://github.com/liurui39660/SNNDPC
    """
    def __init__(self, n_cluster=10, n_neighbors=10):
        """
        初始化函数
        :param n_cluster: 簇的数量
        :param n_neighbors: 近邻数
        """
        self.n_cluster = n_cluster
        self.n_neighbors = n_neighbors
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro

    def fit(self, data):
        """
        训练过程
        :param data: 样本集
        :return:
        """
        unassigned = -1
        n = data.shape[0]
        # Compute distance
        distance = self.GD.pairwise_dist(data, self.metric)
        # Compute neighbor
        indexDistanceAsc = np.argsort(distance)
        indexNeighbor = indexDistanceAsc[:, :self.n_neighbors]
        # Compute shared neighbor
        indexSharedNeighbor = np.empty([n, n, self.n_neighbors], int)
        numSharedNeighbor = np.empty([n, n], int)
        for i in range(n):
            numSharedNeighbor[i, i] = 0
            for j in range(i):
                shared = np.intersect1d(indexNeighbor[i], indexNeighbor[j], assume_unique=True)
                numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = shared.size
                indexSharedNeighbor[j, i, :shared.size] = indexSharedNeighbor[i, j, :shared.size] = shared
        # Compute similarity
        similarity = np.zeros([n, n])
        for i in range(n):
            for j in range(i):
                if i in indexSharedNeighbor[i, j] and j in indexSharedNeighbor[i, j]:
                    indexShared = indexSharedNeighbor[i, j, :numSharedNeighbor[i, j]]
                    distanceSum = np.sum(distance[i, indexShared] + distance[j, indexShared])
                    similarity[i, j] = similarity[j, i] = numSharedNeighbor[i, j] ** 2 / distanceSum
        # Compute ρ
        rho = np.sum(np.sort(similarity)[:, -self.n_neighbors:], axis=1)
        # Compute δ
        distanceNeighborSum = np.empty(n)
        for i in range(n):
            distanceNeighborSum[i] = np.sum(distance[i, indexNeighbor[i]])
        indexRhoDesc = np.argsort(rho)[::-1]
        delta = np.full(n, np.inf)
        for i, a in enumerate(indexRhoDesc[1:], 1):
            for b in indexRhoDesc[:i]:
                delta[a] = min(delta[a], distance[a, b] * (distanceNeighborSum[a] + distanceNeighborSum[b]))
        delta[indexRhoDesc[0]] = -np.inf
        delta[indexRhoDesc[0]] = np.max(delta)
        # Compute γ
        gamma = rho * delta
        # Compute centroid
        indexAssignment = np.full(n, unassigned)
        indexCentroid = np.sort(np.argsort(gamma)[-self.n_cluster:])
        indexAssignment[indexCentroid] = np.arange(self.n_cluster)
        # Assign non-centroid step 1
        queue = indexCentroid.tolist()
        while queue:
            a = queue.pop(0)
            for b in indexNeighbor[a]:
                if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= self.n_neighbors / 2:
                    indexAssignment[b] = indexAssignment[a]
                    queue.append(b)
        # Assign non-centroid step 2
        indexUnassigned = np.argwhere(indexAssignment == unassigned).flatten()
        while indexUnassigned.size:
            numNeighborAssignment = np.zeros([indexUnassigned.size, self.n_cluster], int)
            for i, a in enumerate(indexUnassigned):
                for b in indexDistanceAsc[a, :self.n_neighbors]:
                    if indexAssignment[b] != unassigned:
                        numNeighborAssignment[i, indexAssignment[b]] += 1
            if most := np.max(numNeighborAssignment):
                temp = np.argwhere(numNeighborAssignment == most)
                indexAssignment[indexUnassigned[temp[:, 0]]] = temp[:, 1]
                indexUnassigned = np.argwhere(indexAssignment == unassigned).flatten()
            else:
                self.n_neighbors += 1
        return indexAssignment

    def fit_transform(self, data):
        """
        主函数
        :param data: 数据集
        :return:
        """
        labels = self.fit(data)
        return labels

class Grassmann_AffinityPropagation:
    """
    Affinity Propagation on Grassmann Manifold
    [1] https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
    """
    def __init__(self):
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro

    def fit(self, data):
        dist = -self.GD.pairwise_dist(data, self.metric)
        AP = AffinityPropagation(affinity='precomputed')
        AP.fit(dist)
        return AP.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels

class Grassmann_AgglomerativeClustering:
    """
    Agglomerative Clustering on Grassmann Manifold
    [1] https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
    """
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro

    def fit(self, data):
        dist = self.GD.pairwise_dist(data, self.metric)
        AC = AgglomerativeClustering(n_clusters=self.n_cluster, linkage="average", metric='precomputed')
        AC.fit(dist)
        return AC.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels

class Grassmann_DBSCAN:
    """
    DBSCAN on Grassmann Manifold
    [1] https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
    """
    def __init__(self):
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro

    def fit(self, data):
        dist = self.GD.pairwise_dist(data, self.metric)
        DBS = DBSCAN(metric='precomputed')
        DBS.fit(dist)
        return DBS.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels

class Grassmann_HDBSCAN:
    """
    HDBSCAN on Grassmann Manifold
    [1] https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
    """
    def __init__(self):
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro

    def fit(self, data):
        dist = self.GD.pairwise_dist(data, self.metric)
        HDBS = HDBSCAN(metric='precomputed')
        HDBS.fit(dist)
        return HDBS.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels

class Grassmann_OPTICS:
    """
    OPTICS on Grassmann Manifold
    [1] https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
    """
    def __init__(self):
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro

    def fit(self, data):
        dist = self.GD.pairwise_dist(data, self.metric)
        OPT = OPTICS(metric='precomputed')
        OPT.fit(dist)
        return OPT.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels

class Grassmann_SpectralClustering:
    """
    Spectral Clustering on Grassmann Manifold
    [1] https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
    """
    def __init__(self, n_clusters=5, neighbors=5):
        self.n_clusters = n_clusters
        self.neighbors = neighbors
        self.GK = GrassmannKernel()
        self.metric = self.GK.projection_kernel

    def fit(self, data):
        kernel = self.GK.pairwise_kernel(data, self.metric)
        SC = SpectralClustering(n_clusters=self.n_clusters, n_neighbors=self.neighbors, affinity='precomputed')
        SC.fit(kernel)
        return SC.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels
