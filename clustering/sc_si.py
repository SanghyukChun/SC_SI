# -*- coding: utf-8 -*-
import time
import multiprocessing

import numpy as np
from scipy.sparse.linalg import svds

from _init_methods import SC_IN, _findnnspace


def mysvds(X, U, k, n_iter):
    for _ in range(n_iter):
        U = X.dot(X.T.dot(U))
        U, _ = np.linalg.qr(U)
    return U[:, :k]


def __compute_subspace(X, U, n_components, opt):
    n_samples, n_features = X.shape
    if n_samples <= n_components:
        return U
    if opt.get('svd_algorithm', 'subspace_iteration') == 'subspace_iteration':
        return mysvds(X, U, n_components, opt.get('n_subspace_iter', 1))
    else:
        U, _, _ = svds(X, n_components)
        return U


def __single_iteration__(chunks):
    ret_centers, ret_subspaces = {}, {}
    for cluster_idx, _X, _D, centers, subspaces, n_components, opt in chunks:
        _sumDW = np.sum(_D)

        Xj = _X.T
        DW = _D.T
        DW.shape = (len(DW), 1)
        Yj = np.multiply(Xj, DW.T)
        Yj = Yj - (Xj.dot(DW)).dot(DW.T / _sumDW)

        b = Xj.dot(DW) / _sumDW
        b.shape = len(b)
        ret_centers[cluster_idx] = b

        ret_subspaces[cluster_idx] =\
            __compute_subspace(Yj, subspaces[cluster_idx].T, n_components, opt).T
    return ret_centers, ret_subspaces


def _compute_norm(chunks, max_n=100000):
    def compute_norm(X, U):
        X = X - (X.dot(V.T)).dot(U)
        return np.linalg.norm(X, axis=1)
    dist = {}
    for idx, Y, V in chunks:
        n_data = Y.shape[0]
        if n_data > max_n:
            _dist = dist.setdefault(idx, [])
            for i in range(int(n_data / max_n) + 1):
                _Y = Y[max_n * i: max_n * (i + 1)]
                _dist.extend(compute_norm(_Y, V))
        else:
            dist[idx] = compute_norm(Y, V)
    return dist


class SC_SI(object):
    def __init__(self, n_clusters=8, n_components=10, init='sc_in', init_space=None, max_iter=100,
                 verbose=0, compute_labels=True, random_state=None,
                 svd_algorithm='subspace_iteration', alpha=1.0, n_subspace_iter=1,
                 tol=1e-7, tol2=3, n_jobs=1,
                 beta=None, n_init=3):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.init = init
        self.init_space = init_space
        self.max_iter = max_iter
        self.verbose = verbose
        self.compute_labels = compute_labels
        self.random_state = random_state
        self.svd_algorithm = svd_algorithm
        self.alpha = alpha
        self.beta = beta
        self.n_subspace_iter = n_subspace_iter
        self.tol = tol
        self.tol2 = tol2
        max_cpu = multiprocessing.cpu_count()
        if n_jobs < 0:
            n_jobs = max_cpu
        self.n_jobs = min(max_cpu, n_jobs)
        self.n_init = n_init

        if hasattr(self.init, '__array__') and hasattr(self.init_space, '__array__'):
            self.centers = self.init
            self.subspaces = self.init_space

    def _compute_subspace(self, X, U):
        n_samples, n_features = X.shape
        if n_samples <= self.n_components:
            return U
        if self.svd_algorithm == 'subspace_iteration':
            return mysvds(X, U, self.n_components, self.n_subspace_iter)
        else:
            U, _, _ = svds(X, self.n_components)
            return U

    def __iteration__(self, X, centers, subspaces, labels, D):
        for cluster_idx in range(self.n_clusters):
            _idx = (labels == cluster_idx)
            _X = X[_idx]
            _D = D[cluster_idx][_idx]
            _center, _subspace = __single_iteration__(
                [(cluster_idx, _X, _D, centers, subspaces, self.n_components, {})]
            )
            centers[cluster_idx] = _center[cluster_idx]
            subspaces[cluster_idx] = _subspace[cluster_idx]
        obj, labels, D = self._compute_obj(X, centers, subspaces)
        return centers, subspaces, labels, D, obj

    def __multi_iteration__(self, X, centers, subspaces, labels, D):
        n_samples, n_features = X.shape
        n_jobs = min(self.n_jobs, self.n_clusters)
        if n_jobs == 1:
            return self.__iteration__(X, centers, subspaces, labels, D)
        chunks = {}
        for cluster_idx in range(self.n_clusters):
            _idx = (labels == cluster_idx)
            _X = X[_idx]
            _D = D[cluster_idx][_idx]
            _chunks = cluster_idx, _X, _D, centers, subspaces, self.n_components, {}
            chunks.setdefault(cluster_idx % n_jobs, []).append(_chunks)
        try:
            pool = multiprocessing.Pool(n_jobs)
            rets = pool.map(__single_iteration__, chunks.values())
        except:
            pool.terminate()
            raise
        pool.close()
        pool.join()
        centers = np.zeros((self.n_clusters, n_features))
        subspaces = np.zeros((self.n_clusters, self.n_components, n_features))
        for _centers, _subspaces in rets:
            for cluster_idx, _center in _centers.iteritems():
                centers[cluster_idx] = _center
            for cluster_idx, _subspace in _subspaces.iteritems():
                subspaces[cluster_idx] = _subspace
        obj, labels, D = self._compute_obj(X, centers, subspaces)
        return centers, subspaces, labels, D, obj

    def _compute_obj(self, X, centers, subspaces):
        n_samples, n_features = X.shape
        n_jobs = min(self.n_jobs, self.n_clusters)
        if n_jobs == 1:
            rets = {}
            for idx in range(self.n_clusters):
                Y = X - centers[idx]
                V = subspaces[idx]
                _chunks = idx, Y, V
                rets[idx] = _compute_norm([_chunks])[idx]
            rets = [rets]
        else:
            chunks = {}
            '''
            WARNING: for large datasets, load full chunks on memory
            may cause OOM error
            '''
            for idx in range(self.n_clusters):
                Y = X - centers[idx]
                V = subspaces[idx]
                _chunks = idx, Y, V
                chunks.setdefault(idx % n_jobs, []).append(_chunks)
            try:
                pool = multiprocessing.Pool(n_jobs)
                rets = pool.map(_compute_norm, chunks.values())
            except:
                pool.terminate()
                raise
            pool.close()
            pool.join()
        dist = np.zeros((self.n_clusters, n_samples))
        for ret in rets:
            for idx, _dist in ret.iteritems():
                dist[idx] = _dist
        # prevent nan error
        dist += 1e-100 * np.ones((self.n_clusters, n_samples))

        _dist = np.min(dist ** self.alpha, axis=0)
        obj = np.sum(_dist)
        labels = np.argmin(dist ** self.alpha, axis=0)
        D = (self.alpha / 2.0) * (dist ** (self.alpha - 2))
        return obj, labels, D

    def _init_clusters(self, X, sc_in_N0=None, sc_in_k=None):
        n_samples, n_features = X.shape
        subspaces = None
        if sc_in_N0 is None:
            sc_in_N0 = int(n_samples / (self.n_clusters ** 2))
        sc_in_N0 = max(sc_in_N0, self.n_components * 2)
        if sc_in_k is None:
            sc_in_k = int(0.9 * sc_in_N0)
        if self.init == 'sc_in':
            beta = self.beta if self.beta else self.alpha
            sc_in = SC_IN(n_clusters=self.n_clusters,
                          n_components=self.n_components,
                          beta=beta, verbose=self.verbose)
            centers, subspaces = sc_in.fit(X, N0=sc_in_N0, k=sc_in_k)
        elif self.init == 'random':
            seeds = np.random.permutation(n_samples)[:self.n_clusters]
            centers = X[seeds]
        elif hasattr(self.init, '__array__'):
            centers = np.array(self.init, dtype=X.dtype)
            if self.init_space is not None:
                subspaces = self.init_space
        elif callable(self.init):
            centers = self.init(X, self.n_clusters, random_state=self.random_state)
            centers = np.asarray(centers, dtype=X.dtype)
        if subspaces is None:
            subspaces = np.zeros((self.n_clusters, self.n_components, n_features))
            for idx, center in enumerate(centers):
                _center, U = _findnnspace(X, center, sc_in_N0, sc_in_k, self.n_components)
                centers[idx] = _center
                subspaces[idx] = U
        return centers, subspaces

    def _initialize(self, X, sc_in_N0, sc_in_k):
        t = time.time()
        best_obj = np.inf
        centers, subspaces = None, None
        labels, D = None, None
        for n_iter in range(self.n_init):
            if self.verbose:
                print '[SC_SI] Initialization %s / %s' % (n_iter + 1, self.n_init)
            _centers, _subspaces = self._init_clusters(X, sc_in_N0, sc_in_k)
            _obj, _labels, _D = self._compute_obj(X, _centers, _subspaces)
            if _obj < best_obj:
                best_obj = _obj
                centers = _centers
                subspaces = _subspaces
                labels = _labels
                D = _D
        if self.verbose:
            print('[SC_SI] Initialization Done. takes %s secs' % (time.time() - t))
            print('[SC_SI] Start Training')
            print('[SC_SI] 0 / %s, obj: %s, improvement: -, # changed labels: -' %
                  (self.max_iter, best_obj))
        self._check_subspace(subspaces)
        return best_obj, centers, subspaces, labels, np.ones(D.shape)

    def _check_subspace(self, subspaces):
        assert subspaces.shape[0] == self.n_clusters
        assert subspaces.shape[1] == self.n_components
        assert len(subspaces.shape) == 3
        I = np.diag(np.ones((self.n_components)))
        for vv in subspaces:
            for v in subspaces:
                assert np.sum((I - v.dot(v.T)) ** 2) < 1e-5, 'subspace is not orthogonal, %s' % v.dot(v.T)

    def predict(self, X):
        _, labels, _ = self._compute_obj(X, self.centers, self.subspaces)
        return labels

    def score(self, X):
        obj, _, _ = self._compute_obj(X, self.centers, self.subspaces)
        return obj

    def _fit(self, X, y=None, sc_in_N0=None, sc_in_k=None):
        ''' X : array-like matrix, shape=(n_samples, n_features)
        Training instances to cluster.
        y : Ignored'''
        X = np.array(X)
        assert self.n_components < X.shape[1],\
            'too large subspace dimension %s, %s' % (self.n_components, X.shape)
        prev_obj, centers, subspaces, prev_labels, D = self._initialize(X, sc_in_N0, sc_in_k)
        n_iter, converged, n_unchanged = 0, False, 0
        t = time.time()
        while(not converged):
            n_iter += 1
            centers, subspaces, labels, D, obj =\
                self.__multi_iteration__(X, centers, subspaces, prev_labels, D)
            self._check_subspace(subspaces)
            n_changed = np.sum(prev_labels != labels)
            if n_changed == 0:
                n_unchanged += 1
            else:
                n_unchanged = 0
            prev_labels = labels
            improvement = (prev_obj - obj) / prev_obj
            prev_obj = obj
            if self.verbose:
                print('[SC_SI] %s / %s, obj: %s, improvement: %s, # changed labels: %s' %
                      (n_iter, self.max_iter, obj, improvement, n_changed))
            if n_iter >= self.max_iter:
                if self.verbose:
                    print('reached to max iteration. Break')
                converged = True
            if improvement < 0:
                print 'WARNING: objective function should be MONOTONICALLY decreased!!'
            if np.abs(improvement) < self.tol:
                if self.verbose:
                    print('improvement is less than %s. Break' % self.tol)
                converged = True
            if n_unchanged >= self.tol2:
                if self.verbose:
                    print('labels are unchanged during recent %s iters. Break' % self.tol2)
                converged = True
        if self.verbose:
            print('[SC_SI] Training Done. takes %s secs' % (time.time() - t))
        self.centers = centers
        self.subspaces = subspaces
        return obj, labels

    def fit(self, X, y=None, sc_in_N0=None, sc_in_k=None):
        return self._fit(X, y, sc_in_N0, sc_in_k)[0]

    def fit_predict(self, X, y=None, sc_in_N0=None, sc_in_k=None):
        return self._fit(X, y, sc_in_N0, sc_in_k)[1]


class MiniBatchSC_SI(SC_SI):
    def __init__(self, n_clusters=8, n_components=10, init='sc_in', init_space=None, max_iter=100,
                 verbose=0, compute_labels=True, random_state=None,
                 svd_algorithm='subspace_iteration', alpha=1.0, n_subspace_iter=1,
                 tol=1e-7, tol2=3, n_jobs=1,
                 beta=None, n_init=3):
        super(MiniBatchSC_SI, self).__init__(n_clusters,
                                             n_components,
                                             init,
                                             init_space,
                                             max_iter,
                                             verbose,
                                             compute_labels,
                                             random_state,
                                             svd_algorithm,
                                             alpha,
                                             n_subspace_iter,
                                             tol, tol2,
                                             n_jobs,
                                             beta,
                                             n_init)

    def __multi_iteration__(self, X, centers, subspaces, labels, D, w=0.1):
        for cluster_idx in range(self.n_clusters):
            _idx = (labels == cluster_idx)
            _X = X[_idx]
            _D = D[cluster_idx][_idx]
            _center, _subspace = __single_iteration__(
                [(cluster_idx, _X, _D, centers, subspaces, self.n_components, {})]
            )
            centers[cluster_idx] = centers * (1 - w) + _center[cluster_idx] * w
            subspaces[cluster_idx] = _subspace[cluster_idx]
        obj, labels, D = self._compute_obj(X, centers, subspaces)
        return centers, subspaces, labels, D, obj
