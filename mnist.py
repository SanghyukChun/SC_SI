# -*- coding: utf-8 -*-
import os
import gzip
import urllib2
import numpy as np


class MNIST:
    '''base code are from
    https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
    '''
    def __init__(self, base_dir='./'):
        self.base_dir = base_dir
        self.url = 'http://yann.lecun.com/exdb/mnist/%s'
        self.images = {'train': 'train-images-idx3-ubyte.gz',
                       'test': 't10k-images-idx3-ubyte.gz'}
        self.labels = {'train': 'train-labels-idx1-ubyte.gz',
                       'test': 't10k-labels-idx1-ubyte.gz'}
        self.image_size = 28
        self.n_images = {'train': 60000,
                         'test': 10000}
        self.pixel_depth = 255

    def download(self):
        mnist_dir = os.path.join(self.base_dir, 'mnist')
        if not os.path.exists(mnist_dir):
            os.mkdir(mnist_dir)
        for phase in ('train', 'test'):
            image_fname = os.path.join(mnist_dir, self.images[phase])
            label_fname = os.path.join(mnist_dir, self.labels[phase])
            if not os.path.exists(image_fname):
                print 'download image from', self.url % self.images[phase]
                response = urllib2.urlopen(self.url % self.images[phase])
                open(image_fname + '.tmp', 'w').write(response.read())
                os.rename(image_fname + '.tmp', image_fname)

                print 'download label from', self.url % self.labels[phase]
                response = urllib2.urlopen(self.url % self.labels[phase])
                open(label_fname + '.tmp', 'w').write(response.read())
                os.rename(label_fname + '.tmp', label_fname)

    def load(self, phase, nrz=False):
        assert phase in ('train', 'test'), 'invalid phase %s should be (train / test)' % phase
        mnist_dir = os.path.join(self.base_dir, 'mnist')
        image_fname = os.path.join(mnist_dir, self.images[phase])
        label_fname = os.path.join(mnist_dir, self.labels[phase])

        with gzip.open(label_fname) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * self.n_images[phase])
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        with gzip.open(image_fname) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(self.image_size ** 2 * self.n_images[phase])
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            if nrz:
                data = (data - (self.pixel_depth / 2.0)) / self.pixel_depth
            data = data.reshape(self.n_images[phase], self.image_size ** 2)
        return data, labels
