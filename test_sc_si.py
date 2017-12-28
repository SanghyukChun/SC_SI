# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print 'failed to load matplotlib'

from mnist import MNIST
from clustering.sc_si import SC_SI


class Test:
    def make_3d_dataset(self, n_per_k, d, r, k, theta=0.05):
        U1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        X = np.zeros((n_per_k * k, d))
        labels = []
        X0 = np.random.random((n_per_k, 2))
        X0 = X0.dot(U1)
        X[:n_per_k, :] = X0
        labels.extend([0] * n_per_k)
        for i in range(1, k):
            _t = theta * i
            T = np.array([[np.cos(_t), -np.sin(_t), 0.0],
                          [np.sin(_t), np.cos(_t), 0.0],
                          [0.0, 0.0, 1.0]])
            _X = T.dot(X0.T)
            _X = _X.T
            X[n_per_k * i: n_per_k * (i + 1), :] = _X
            labels.extend([i] * n_per_k)
        return X, np.array(labels), None, None

    def make_random_dataset(self, n_per_k, d, r, k, scale=0.5, noise=0.1):
        X = np.zeros((n_per_k * k, d))
        labels = []
        bs = []
        Us = []
        assert r * k < d
        Q, _ = np.linalg.qr(np.random.random((d, d)))
        for i in range(k):
            U = Q[:, r * i: r * (i + 1)]
            b = (np.ones(r) * 0.5).dot(U.T)
            _Z = np.random.random((n_per_k, r))
            _b = np.mean(_Z, axis=0)
            _Z = (_Z - _b).dot(U.T) + b
            _X = scale * _Z \
                + np.random.random((n_per_k, d)) * noise
            labels.extend([i] * n_per_k)
            X[n_per_k * i: n_per_k * (i + 1), :] = _X
            bs.append(b)
            Us.append(U.T)
        return X, np.array(labels), np.array(bs), np.array(Us)

    def eval_cluster(self, pred, true):
        '''http://scikit-learn.org/stable/modules/clustering.html#clustering
        '''
        print 'homogeneity score', metrics.homogeneity_score(pred, true)
        print 'completeness score', metrics.completeness_score(pred, true)
        print 'v measure score', metrics.v_measure_score(pred, true)
        print 'fowlkes mallows score', metrics.fowlkes_mallows_score(pred, true)

    def vis_cluster(self, X, names, models, preds, sample=2000):
        if X.shape[1] > 3:
            X = PCA(n_components=3).fit_transform(X)

        IDX = np.random.choice(X.shape[0], sample)
        X = X[IDX]

        fig = plt.figure()
        n = len(models)
        if n == 1:
            ax = Axes3D(fig)
            ax.set_title(names[0])
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=preds[0][IDX])
        else:
            for idx, (name, model, pred) in enumerate(zip(names, models, preds)):
                pos = '%s1%s' % (n, idx + 1)
                ax = fig.add_subplot(int(pos), projection='3d')
                ax.set_title(name)
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred[IDX])
        plt.show()

    def test_mnist(self):
        mnist = MNIST()
        mnist.download()
        X, true = mnist.load('train')
        model = SC_SI(n_clusters=10, n_components=20, verbose=True, n_init=3,
                      beta=10)
        pred = model.fit_predict(X)
        print 'sc si'
        self.eval_cluster(pred, true)
        print Counter(pred)

        model = KMeans(n_clusters=10, n_init=3)
        pred = model.fit_predict(X)
        print 'kmeans'
        self.eval_cluster(pred, true)
        print Counter(pred)

    def test(self, n=10000, d=120, r=10, k=3):
        X, true, centers, spaces = self.make_random_dataset(n, d, r, k)

        model = SC_SI(n_clusters=k, n_components=r, verbose=True, n_init=3)
        pred = model.fit_predict(X)
        print 'sc si'
        self.eval_cluster(pred, true)

        model = KMeans(n_clusters=k, n_init=3)
        pred = model.fit_predict(X)
        print 'kmeans'
        self.eval_cluster(pred, true)

    def test_3d_dataset(self, n=10000, verbose=False):
        d, k, r = 3, 2, 2
        X, true, centers, spaces = self.make_3d_dataset(n, d, r, k)
        model_sc_si = SC_SI(n_clusters=k, n_components=r, verbose=verbose, n_init=10)
        pred_sc_si = model_sc_si.fit_predict(X)
        print '[Benchmark Test] SC_SI (SC_IN):'
        self.eval_cluster(pred_sc_si, true)
        print ''

        model = SC_SI(n_clusters=k, n_components=r, verbose=verbose, init='random', n_init=10)
        pred = model.fit_predict(X)
        print '[Benchmark Test] SC_SI (Random Init):'
        self.eval_cluster(pred, true)
        print ''

        model = KMeans(n_clusters=k)
        pred = model.fit_predict(X)
        print '[Benchmark Test] K-means:'
        self.eval_cluster(pred, true)

        self.vis_cluster(X, ['SC-SI', 'K-means'],
                         [model_sc_si, model],
                         [pred_sc_si, pred])


if __name__ == '__main__':
    try:
        import fire
    except ImportError:
        print 'Failed to import fire'
        exit(-1)
    fire.Fire(Test)
