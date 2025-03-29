Grassmann是一个软件包，包含了在格拉斯曼流形上进行分类和聚类任务的一些必要工具。
- Grassmann KNN 分类器
```python
from Grassmann import GrassmannKNN
KNN = GrassmannKNN()
KNN.fit(data_train, target_train)
t_pred = KNN.predict(data_test)
```
- Grassmann SVM 分类器 [1-2]
```python
from Grassmann import GrassmannSVM
SVM = GrassmannSVM()
SVM.fit(data_train, target_train)
t_pred = SVM.transform(data_test)
```
- Kernel Discriminant Analysis on Grassmann Manifold [3-4]
```python
from Grassmann import GrassmannKernelFDA
from Grassmann import GrassmannKernel
n_cluster = len(np.unique(target))
n_components = n_cluster - 1 if n_cluster>2 else n_cluster
GK = GrassmannKernel()
GKDA = GrassmannKernelFDA(n_components = n_components, kernel = GK.projection_kernel)
GKDA.fit(data_train, target_train)
embedding_train = GKDA.transform(data_train)
embedding_test = GKDA.transform(data_test)
```
- Regularized Discriminant Analysis on Grassmann Manifold [5]
```python
from Grassmann import GrassmannKernelRDA
GK = GrassmannKernel()
GKDA = GrassmannKernelRDA(kernel = GK.projection_kernel)
GKDA.fit(data_train, target_train)
embedding_train = GKDA.transform(data_train)
embedding_test = GKDA.transform(data_test)
```
- Grassmann Adaptive Local Learning [6]
```python
from Grassmann import GrassmannALL
GA = GrassmannALL(
    n_components = n_components,
    p_grassmann = p_grassmann,
    n_neighbors = n_neighbors,
    train_size = train_size,
    random_state = random_state,
    converged_tol = converged_tol,
    drop_tol = drop_tol,
    max_epoch = max_epoch,
    verbose = verbose
)
embedding_train, embedding_test = GA.fit_transform(data_train, data_test, target_train, target_test)
```
- Nested Grassmann [7-8]
```python
from Grassmann import Nested_Grassmann
embedding_ = Nested_Grassmann().NG_dr(data, m = n_components)
```
- Supervised Nested Grassmann [7-8]
```python
from Grassmann import Nested_Grassmann
embedding_ = Nested_Grassmann().NG_sdr(data, target, m = n_components)
```
- K-Means on Grassmann Manifold [9-10]
```python
from Grassmann import GrassmannKMeans
GKM = GrassmannKMeans(center_count = len(np.unique(target)), n_epoch = 100)
_label = GKM.fit_transform(embedding_)
```
- LBG on Grassmann Manifold [9-10]
```python
from Grassmann import GrassmannLBG
GLBG = GrassmannLBG(center_count = len(np.unique(target)), n_epoch = 100)
_label = GLBG.fit_transform(embedding_)
```
- Clustering on Grassmann Manifold via Kernel Embedding [11]
```python
from Grassmann import CGMKE
CGMKE = CGMKE(center_count = len(np.unique(target)))
_label = CGMKE.fit_transform(embedding_)
```
- Shared Nearest Neighbor Density Peaks Clustering on Grassmann Manifold [12-13]
```python
from Grassmann import Shared_Nearest_Neighbor_DPC
SNNDPCG = Shared_Nearest_Neighbor_DPC(n_cluster = len(np.unique(target)), n_neighbors = 5)
_label = SNNDPCG.fit_transform(embedding_)
```
- Affinity Propagation on Grassmann Manifold
```python
from Grassmann import Grassmann_AffinityPropagation
GAF = Grassmann_AffinityPropagation()
_label = GAF.fit_transform(embedding_)
```
- Agglomerative Clustering on Grassmann Manifold
```python
from Grassmann import Grassmann_AgglomerativeClustering
GAG = Grassmann_AgglomerativeClustering(n_cluster = len(np.unique(target)))
_label = GAG.fit_transform(embedding_)
```
- DBSCAN on Grassmann Manifold
```python
from Grassmann import Grassmann_DBSCAN
GDBS = Grassmann_DBSCAN()
_label = GDBS.fit_transform(embedding_)
```
- HDBSCAN on Grassmann Manifold
```python
from Grassmann import Grassmann_HDBSCAN
GHDBS = Grassmann_HDBSCAN()
_label = GHDBS.fit_transform(embedding_)
```
- OPTICS on Grassmann Manifold
```python
from Grassmann import Grassmann_OPTICS
GOPT = Grassmann_OPTICS()
_label = GOPT.fit_transform(embedding_)
```
- Grassmann Spectral Clustering
```python
from Grassmann import Grassmann_SpectralClustering
GSC = Grassmann_SpectralClustering(n_clusters = len(np.unique(target)), neighbors = 5)
_label = GSC.fit_transform(embedding_)
```
- Generalized Relevance Learning Grassmann Quantization [14-15]
```python
from Grassmann import GRLGQ_Run
t_pred, accuracy = GRLGQ_Run(
    dim_of_subspace=p_grassmann,
    nepochs=250
).fit(
    data_train=data_train,
    data_test=data_test,
    target_train=target_train,
    target_test=target_test
)
```
参考文献
```
[1] Al-Samhi W, Al-Soswa M, Al-Dhabi Y. Time series data classification on grassmann manifold[C]. Journal of Physics: Conference Series. IOP Publishing, 2021, 1848(1): 012037.
[2] https://github.com/adamguos/arma-grassmann-classifier
[3] Hamm J, Lee D D. Grassmann discriminant analysis: a unifying view on subspace-based learning[C]. Proceedings of the 25th International Conference on Machine Learning. 2008: 376-383.
[4] https://github.com/concavegit/kfda/
[5] https://github.com/daviddiazvico/scikit-kda
[6] Wei D, Shen X, Sun Q, et al. Learning adaptive Grassmann neighbors for image-set analysis[J]. Expert Systems with Applications, 2024, 247: 123316.
[7] Yang C H, Vemuri B C. Nested grassmannians for dimensionality reduction with applications[J]. The Journal of Machine Learning for Biomedical Imaging, 2022, 2022: 002.
[8] https://github.com/cvgmi/NestedGrassmann
[9] Stiverson S J. An Adaptation of K-Means-Type Algorithms to The Grassmann Manifold[D]. Colorado State University, 2019.
[10] https://github.com/sjstiver/Grassmannian_clustering
[11] Shirazi S, Harandi M T, Sanderson C, et al. Clustering on Grassmann manifolds via kernel embedding with application to action analysis[C]. 2012 19th IEEE International Conference on Image Processing. IEEE, 2012: 781-784.
[12] Liu R, Wang H, Yu X. Shared-nearest-neighbor-based clustering by fast search and find of density peaks[J]. Information Sciences, 2018, 450: 200-226.
[13] https://github.com/liurui39660/SNNDPC
[14] Mohammadi M, Babai M, Wilkinson M H F. Generalized relevance learning grassmann quantization[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
[15] https://github.com/mohammadimathstar/GRLGQ
```
