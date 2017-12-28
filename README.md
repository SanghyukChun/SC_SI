# SC_SI

Python implementation of SC_SI (Subspace Clustering with Scalable and Iterative Algorithm).
SC_SI solves multiple R-PCA (Robust PCA) problem to clustering.

You can check full paper in the link: http://library.kaist.ac.kr/mobile/book/view.do?bibCtrlNo=649637

## Usage

Almost same as sci-kit learn K-means [link](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
```
import numpy as np
from sc_si.clustering.sc_si import SC_SI

n_data = 10000
dim_data = 500
X = np.random.random((n_data, dim_data))

n_clusters = 10
n_components = 20
model = SC_SI(n_clusters=n_clusters, n_components=n_components, alpha=1.0, init='sc_si', n_init=3, max_iter=100, verbose=True)
labels = model.fit_predict(X)

n_data2 = 1000
X_unseen = np.random.random((n_data2, dim_data))
labels_unseen = model.predict(X_unseen)
```

## Optimization Hints

1. alpha handles 'robustness' of the objective function. The objective function is more robust with less alpha.
2. Use alpha=1.0. Theoretically alpha can be any number between (0, 2] but practically, I recommend to choose alpha as 1
3. Use default initialization named SC_IN if size of dataset is not too large. Otherwise, use 'random' initialization with large n_init.
  - Or, sampling datasets to initialization
4. Use default svd_algorithm (subspace iteration). It is much faster and use less memory than exact SVD.
5. For datasets with less outliers, use large beta (e.g. 10) otherwise, set beta = alpha
