import cPickle, gzip
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class HiddenLayer:
    def __init__(self, intput, n_in, n_out):
        self.input = input

        W_values = 4 * numpy.random.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=-numpy.sqrt(6. / (n_in, n_out)),
            size=(n_in, n_out))

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        # It is a safe practice (and a good idea) to use borrow=True in a shared variable
        # constructor when the shared variable stands for a large object (in terms of memory
        # footprint) and you do not want to create copies of it in memory.
        self.b = theano.shared(value=numpy.zeros((n_out, )),
                               name='b',
                               borrow=True)
        self.params = [self.W, self.b]

        net = T.dot(self.input, self.W) + self.b
        self.output = T.nnet.sigmoid(net)


class MultiLogisticRegression:
    def __init__(self, input, n_in, n_out):
        self.input = input

        self.W = theano.shared(value=numpy.zeros((n_in, n_out)),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out, )),
                               name='b',
                               borrow=True)
        self.params = [self.W, self.b]
        self.prob = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.predict = T.argmax(self.prob, axis=1)

    def nll(self, y):
        return -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])  # TODO


class ConvPoolLayer:
    def __init__(self, input, filter_shape, image_shape, poolsize):
        self.input = input

        n_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
        n_out = (filter_shape[0] * filter_shape[2] * filter_shape[3]) / (
            poolsize[0] * poolsize[1])

        W_bound = numpy.sqrt(6. / (n_in + n_out))  # TODO:

        self.W = theano.shared(
            numpy.random.uniform(low=-W_bound,
                                 high=W_bound,
                                 size=filter_shape),
            borrow=True)

        self.b = theano.shared(value=numpy.zeros((filter_shape[0], )),
                               borrow=True)
        self.params = [self.W, self.b]

        conv_out = conv.conv2d(input=self.input,
                               filters=self.W,
                               filter_shape=filter_shape,
                               image_shape=image_shape)

        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize,
                                            ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
