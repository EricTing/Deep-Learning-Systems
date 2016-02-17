#!/usr/bin/env python


import numpy as np
import theano
import theano.tensor as T


class LogisticReg:
    """A simple implementation of Logistic Regression,
    Only works for two class classification, labels must 0 and 1
    """

    def __init__(self, learning_rate=0.01, C=1.0, max_iter=200, penalty=None):
        self._learning_rate = learning_rate
        self._C = C
        self._max_iter = max_iter
        self._parameters = {}
        self._penalty = penalty

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

        # must use mean() rather than the sum() of individual cost, or there will be overflow
        if self._penalty == "l1":
            mean_nll = (-t * T.log(probas) - (1 - t) *
                        T.log(1 - probas)).mean() + self._C * T.abs_(w).sum()
        elif self._penalty == "l2":
            mean_nll = (-t * T.log(probas) - (1 - t) *
                        T.log(1 - probas)).mean() + self._C * (w**2).sum()
        elif self._penalty is None:
            mean_nll = (-t * T.log(probas) - (1 - t) *
                        T.log(1 - probas)).mean()
        else:
            raise Exception("unknown penalty %s" % self._penalty)

        gw, gb = T.grad(mean_nll, [w, b])

        d = self._learning_rate

        myupdaets = [(w, w - d * gw), (b, b - d * gb)]
        train = theano.function(inputs=[x, t],
                                outputs=mean_nll,
                                updates=myupdaets)

        costs, steps = [], []
        for i in range(self._max_iter):
            cost = train(X, y)
            costs.append(cost)
            steps.append(i)

        self._parameters["costs"] = costs
        self._parameters["steps"] = steps
        self._parameters["probas"] = probas
        calculate_probas = theano.function(inputs=[x], outputs=probas)
        self._parameters["calculate_probas"] = calculate_probas
        self._parameters["weights"] = w.get_value()

    def predict_proba(self, X):
        return self._parameters["calculate_probas"](X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.array([1 if p > 0.5 else 0 for p in probas])

    def get_parameters(self):
        return self._parameters


def test():
    import pandas as pd

    data = pd.read_pickle('prostate.df')

    y = data.values[:, -1]
    x = data.values[:, :-1]

    from sklearn.cross_validation import KFold
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    t = encoder.fit_transform(y)  # targets
    kf = KFold(t.size, 10, shuffle=True)

    train_index, test_index = list(kf)[0]
    x_train, x_test = x[train_index], x[test_index]
    t_train, t_test = t[train_index], t[test_index]

    logistc_reg = LogisticReg(max_iter=300, C=1.0, learning_rate=0.01)
    logistc_reg.fit(x_train.astype('float'), t_train.astype('float'))

    parameters = logistc_reg.get_parameters()
    costs = parameters['costs']
    steps = parameters['steps']
    import matplotlib.pyplot as plt
    plt.plot(steps, costs)
    plt.show()

    predictions = logistc_reg.predict(x_test.astype("float"))

    from sklearn.metrics import accuracy_score
    print(accuracy_score(predictions, t_test))

    # D = np.random.random((100, 10))
    # t = np.random.binomial(1, 0.5, D.shape[0])

    # log = LogisticReg(max_iter=1000)
    # log.fit(D, t)
    # print t
    # print log.predict(D)


def main():
    pass


if __name__ == '__main__':
    main()
