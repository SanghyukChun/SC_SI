# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse.linalg import svds


def _findnnspace(X, center, N0, k, n_components):
    Y = X - center
    dist = np.sum(Y ** 2, axis=1)
    indices = np.argsort(dist)
    randp = np.random.choice(N0, k)
    indices = indices[randp]

    _X = X[indices]
    _center = np.mean(_X, axis=0)
    _Y = _X - _center
    _, _, U = svds(_Y, n_components)

    return _center, U


def _findinitcenter(X, n_clusters, centers, subspaces, beta, choose_max=False):
    assert beta > 0
    n_samples, n_features = X.shape
    dist = np.zeros((n_clusters, n_samples))
    for idx in range(n_clusters):
        Y = X - centers[idx]
        V = subspaces[idx]
        Y = Y - (Y.dot(V.T)).dot(V)
        dist[idx] = np.linalg.norm(Y, axis=1) ** beta
    dist = np.min(dist, axis=0)
    if choose_max:
        idx = np.argmax(dist)
    else:
        sum_dist = np.sum(dist)
        if sum_dist > 0:
            dist /= sum_dist
            idx = np.random.choice(n_samples, p=dist)
        else:
            idx = np.random.choice(n_samples)
    return X[idx]


class SC_IN:
    def __init__(self, n_clusters=8, n_components=10, beta=1, verbose=0):
        '''example useage:
                sc_in = SC_IN(verbose=1)
                sc_in.fit(np.random.random((10000, 784)))'''
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.beta = beta
        self.verbose = verbose

    def fit(self, X, y=None, N0=None, k=None):
        ''' X : array-like matrix, shape=(n_samples, n_features)
        Training instances to cluster.
        y : Ignored'''
        X = np.array(X)
        n_samples, n_features = X.shape

        if N0 is None:
            N0 = int(n_samples / (self.n_clusters ** 2))
        N0 = max(N0, self.n_components * 2)
        if k is None:
            k = int(0.9 * N0)

        centers = np.zeros((self.n_clusters, n_features))
        subspaces = np.zeros((self.n_clusters, self.n_components, n_features))
        for i in range(self.n_clusters):
            if i == 0:
                _center = X[np.random.randint(n_samples)]
            else:
                _center = _findinitcenter(X, i, centers, subspaces, self.beta)
            _center, _subspace = _findnnspace(X, _center, N0, k, self.n_components)
            if self.verbose:
                print('[SC_IN] %s / %s' % (i + 1, self.n_clusters))
            centers[i] = _center
            subspaces[i] = _subspace
        self.centers = centers
        self.subspaces = subspaces
        return centers, subspaces
