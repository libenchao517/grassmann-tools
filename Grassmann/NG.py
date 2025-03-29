################################################################################
# 本文件中是Nested Grassmanns for Dimensionality Reduction
################################################################################
# 导入模块
import numpy as np
import torch as tc
import pymanopt as mo
import pymanopt.manifolds as pm
from pymanopt.optimizers import SteepestDescent
################################################################################
class Nested_Grassmann:
    """
    Nested Grassmann 和 Supervised Nested Grassmann 的代码来源于[2]
    [1] Yang C H, Vemuri B C.
    Nested grassmannians for dimensionality reduction with applications[J].
    The journal of machine learning for biomedical imaging, 2022, 2022: 002.
    [2] https://github.com/cvgmi/NestedGrassmann
    """
    def __init__(self):
        pass

    def dist_proj(self, X, Y):
        Px = tc.matmul(X, tc.matmul(tc.inverse(tc.matmul(X.conj().t(), X)), X.conj().t()))
        Py = tc.matmul(Y, tc.matmul(tc.inverse(tc.matmul(Y.conj().t(), Y)), Y.conj().t()))
        if tc.is_complex(X) or tc.is_complex(Y):
            P = Px - Py
            return tc.sqrt(tc.sum(tc.matmul(P, P.conj().t()))).real / np.sqrt(2)
        else:
            return tc.norm(Px - Py) / np.sqrt(2)

    def affinity_matrix(self, dist_m, y, v_w, v_b):
        N = dist_m.shape[0]
        affinity = np.eye(N)
        for i in range(N):
            for j in range(i):
                tmp1 = np.argsort(dist_m[i, y == y[i]])[v_w]
                tmp2 = np.argsort(dist_m[j, y == y[j]])[v_w]
                g_w = int((y[i] == y[j]) and (dist_m[i, j] < np.maximum(tmp1, tmp2)))
                tmp1 = np.argsort(dist_m[i, y != y[i]])[v_b - 1]
                tmp2 = np.argsort(dist_m[j, y != y[j]])[v_b - 1]
                g_b = int((y[i] != y[j]) and (dist_m[i, j] < np.maximum(tmp1, tmp2)))
                affinity[i, j] = g_w - g_b
                affinity[j, i] = affinity[i, j]
        return affinity

    def NG_dr(self, X, m, verbosity=0):
        """
        X: array of N points on Gr(n, p); N x n x p array
        aimc to represent X by X_hat (N points on Gr(m, p), m < n)
        where X_hat_i = R^T X_i, R \in St(n, m)
        minimizing the projection error (using projection F-norm)
        """
        N, n, p = X.shape
        # true if X is complex-valued
        cpx = np.iscomplex(X).any()
        if cpx:
            man = pm.Product([pm.ComplexGrassmann(n, m), pm.Euclidean(n, p, 2)])
        else:
            man = pm.Product([pm.Grassmann(n, m), pm.Euclidean(n, p)])
        X_ = tc.from_numpy(X)
        @mo.function.pytorch(man)
        def cost(A, B):
            AAT = tc.matmul(A, A.conj().t())  # n x n
            if cpx:
                B_ = B[:, :, 0] + B[:, :, 1] * 1j
            else:
                B_ = B
            IAATB = tc.matmul(tc.eye(n, dtype=X_.dtype) - AAT, B_)  # n x p
            d2 = 0
            for i in range(N):
                d2 = d2 + self.dist_proj(X_[i], tc.matmul(AAT, X_[i]) + IAATB) ** 2 / N
            return d2

        solver = SteepestDescent(verbosity=verbosity)
        problem = mo.Problem(manifold=man, cost=cost)
        theta = solver.run(problem)
        A = theta.point[0]
        B = theta.point[1]
        if cpx:
            B_ = B[:, :, 0] + B[:, :, 1] * 1j
        else:
            B_ = B
        tmp = np.array([A.conj().T for i in range(N)])
        X_low = tmp @ X
        X_low = np.array([np.linalg.qr(X_low[i])[0] for i in range(N)])
        return X_low

    def NG_sdr(self, X, y, m, v_w=5, v_b=5, verbosity=0, *args, **kwargs):
        """
        X: array of N points on complex Gr(n, p); N x n x p array
        aim to represent X by X_hat (N points on Gr(m, p), m < n)
        where X_hat_i = R^T X_i, W \in St(n, m)
        minimizing the projection error (using projection F-norm)
        """
        N, n, p = X.shape
        cpx = np.iscomplex(X).any()
        # true if X is complex-valued
        if cpx:
            gr = pm.ComplexGrassmann(n, p)
            man = pm.ComplexGrassmann(n, m)
        else:
            gr = pm.Grassmann(n, p)
            man = pm.Grassmann(n, m)
        # distance matrix
        dist_m = np.zeros((N, N))
        for i in range(N):
            for j in range(i):
                dist_m[i, j] = gr.dist(X[i], X[j])
                dist_m[j, i] = dist_m[i, j]
        # affinity matrix
        values, counts = np.unique(y, return_counts=True)
        v_w = np.minimum(v_w, np.min(counts)-1)
        v_b = np.minimum(v_b, np.min(counts)-1)
        affinity = self.affinity_matrix(dist_m, y, v_w, v_b)
        X_ = tc.from_numpy(X)
        affinity_ = tc.from_numpy(affinity)
        @mo.function.pytorch(man)
        def cost(A):
            dm = tc.zeros((N, N))
            for i in range(N):
                for j in range(i):
                    dm[i, j] = self.dist_proj(tc.matmul(A.conj().t(), X_[i]), tc.matmul(A.conj().t(), X_[j])) ** 2
                    dm[j, i] = dm[i, j]
            d2 = tc.mean(affinity_ * dm)
            return d2
        solver = SteepestDescent(verbosity=verbosity)
        problem = mo.Problem(manifold=man, cost=cost)
        A = solver.run(problem)
        tmp = np.array([A.point.conj().T for i in range(N)])  # N x m x n
        X_low = tmp @ X
        # N x m x p
        X_low = np.array([np.linalg.qr(X_low[i])[0] for i in range(N)])
        return X_low
