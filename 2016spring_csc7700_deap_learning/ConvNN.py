import cPickle, gzip
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        W_values = 4 * numpy.random.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out))
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out, )),
                               name='b',
                               borrow=True)
        self.params = [self.W, self.b]

        net = T.dot(self.input, self.W) + self.b
        self.output = T.nnet.sigmoid(net)


class MultiLogisticRegression(object):
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
        return -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])


class ConvPoolLayer(object):
    def __init__(self, input, filter_shape, image_shape, poolsize):

        self.input = input

        n_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
        n_out = (filter_shape[0] * filter_shape[2] * filter_shape[3]) / (
            poolsize[0] * poolsize[1])
        W_bound = numpy.sqrt(6. / (n_in + n_out))
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


batch_size = 250
learning_rate = 0.1
nkerns = [10, 10]

x = T.matrix('x')
y = T.ivector('y')

layer0_input = x.reshape((batch_size, 1, 28, 28))
layer0 = ConvPoolLayer(input=layer0_input,
                       image_shape=(batch_size, 1, 28, 28),
                       filter_shape=(nkerns[0], 1, 5, 5),
                       poolsize=(2, 2))

layer1 = ConvPoolLayer(input=layer0.output,
                       image_shape=(batch_size, nkerns[0], 12, 12),
                       filter_shape=(nkerns[1], nkerns[0], 5, 5),
                       poolsize=(2, 2))
layer1_output = layer1.output.flatten(2)

layer2 = HiddenLayer(input=layer1_output, n_in=nkerns[1] * 4 * 4, n_out=50, )

layer3 = MultiLogisticRegression(input=layer2.output, n_in=50, n_out=10)

cost = layer3.nll(y)

model_predict = theano.function([x], layer3.predict)

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function([x, y], cost, updates=updates)

dataset = './HW2/digits.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_set_x, train_set_y = train_set
test_set_x, test_set_y = test_set
train_set_y = train_set_y.astype(numpy.int32)

ix = []
for i in range(10):
    ix.append(numpy.nonzero(train_set_y == i)[0][:500])
ix = numpy.concatenate(ix)
train_set_x = train_set_x[ix]
train_set_y = train_set_y[ix]
ix = numpy.random.permutation(train_set_x.shape[0])
train_set_x = train_set_x[ix]
train_set_y = train_set_y[ix]

n_batches = train_set_x.shape[0]
n_batches /= batch_size

n_epochs = 200
c = numpy.zeros((n_epochs, ))
for i in range(n_epochs):
    err = 0
    for b in range(n_batches):
        err += train_model(train_set_x[b * batch_size:(b + 1) * batch_size],
                           train_set_y[b * batch_size:(b + 1) * batch_size])
    print 'iteration:', i, ', nll =', err
    c[i] = err

plot(c)

n_testbatches = test_set_x.shape[0] / batch_size
err = 0
for b in range(n_testbatches):
    yp = model_predict(test_set_x[b * batch_size:(b + 1) * batch_size])
    yy = test_set_y[b * batch_size:(b + 1) * batch_size]
    err += len(np.nonzero(yp - yy)[0])

print 1.0 * err / len(test_set_y)
