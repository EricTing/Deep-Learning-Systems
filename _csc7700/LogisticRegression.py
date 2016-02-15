#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T


class LogisticReg:
    def __init__(self, learning_rate=0.1, C=1.0, max_iter=100):
        "docstring"
        self._learning_rate = learning_rate
        self._C = C
        self._max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            init_weights = np.random.randn(X.shape[1])
        else:
            init_weights = sample_weight

        init_bias = 0.0

        x = T.matrix('x')
        t = T.vector('t')
        w = theano.shared(init_weights, name="w")
        b = theano.shared(init_bias, name="b")

        probas = 1 / (1 + T.exp(-T.dot(x, w) - b))
        nll = (-t * T.log(probas) -
               (1 - t) * T.log(1 - probas)).sum() + self._C * (w**2).sum()

        gw, gb = T.grad(nll, [w, b])

        d = self._learning_rate

        d = 0.01
        myupdaets = [(w, w - d * gw), (b, b - d * gb)]
        train = theano.function(inputs=[x, t], outputs=nll, updates=myupdaets)

        for i in range(self._max_iter):
            cost = train(X, y)
            print cost, w.get_value(), b.get_value()


def test():

    D = np.random.random((3, 3))
    t = np.random.binomial(1, 0.5, D.shape[0])

    log = LogisticReg()
    log.fit(D, t)


def main():
    pass


if __name__ == '__main__':
    main()
