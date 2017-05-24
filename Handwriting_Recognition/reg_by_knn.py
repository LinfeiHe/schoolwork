"""
recognize handwriting by using k-NearestNeighbor algorithm
"""
# Author: helinfei@ecust

import numpy as np
import scipy.io as scio
import random
import collections
import time
import scipy.optimize as opt


def load_data(sub_data=1, num_train=100, num_test=100):
    data = scio.loadmat('mnist_all.mat')
    train_in = []
    test_in = []
    train_out = []
    test_out = []
    for i in range(0, 10):
        if sub_data:
            sel_train = random.sample(range(len(data['train%d' % i])),  num_train)
            sel_test = random.sample(range(len(data['test%d' % i])), num_test)
            train_in.append(np.array(data['train%d' % i][sel_train]))
            test_in.append(np.array(data['test%d' % i][sel_test]))
            train_out.append(i * np.ones((num_train, 1)))
            test_out.append(i * np.ones((num_test, 1)))
        else:
            train_in.append(np.array(data['train%d' % i]))
            test_in.append(np.array(data['test%d' % i]))
            train_out.append(i * np.ones((data['train%d' % i].shape[0], 1)))
            test_out.append(i * np.ones((data['test%d' % i].shape[0], 1)))
    train_in = np.concatenate(train_in) / 255.
    test_in = np.concatenate(test_in) / 255.
    train_out = np.concatenate(train_out)
    test_out = np.concatenate(test_out)
    return train_in, train_out, test_in, test_out


def pca_mxn(sample, test, degree=0.99, white=0, zca=0, eps=10e-5):
    # sample should be m_samples with n_features

    m, n = sample.shape
    sample_mean = np.mean(sample, 1)
    test_mean = np.mean(test, 1)
    sample_new = sample - np.transpose([sample_mean])
    test_new = test - np.transpose([test_mean])
    sigma = np.dot(sample_new.T, sample_new) / m
    u, s, v = np.linalg.svd(sigma)
    lambda_ = np.sqrt(s)
    if 0 < degree < 1:
        k = np.nonzero(np.cumsum(lambda_) / np.sum(lambda_) >= degree)[0][0]
    elif degree > 1 and isinstance(degree, int):
        k = degree
        print 'degree is %s' % (np.cumsum(lambda_) / np.sum(lambda_))[k - 1]
    else:
        print 'wrong degree value'
    sample_reduce = np.dot(sample_new, u[:, :k])
    test_reduce = np.dot(test_new, u[:, :k])
    if white and not zca:
        sample_pca_white = 1 / (lambda_[:k] + eps) * sample_reduce
        test_pca_white = 1 / (lambda_[:k] + eps) * test_reduce
        return sample_pca_white, test_pca_white
    if zca:
        sample_pca_white = 1 / (lambda_ + eps) * np.dot(sample_new, u)
        test_pca_white = 1 / (lambda_ + eps) * np.dot(test_new, u)
        sample_zca_white = np.dot(sample_pca_white, u.T)
        test_zca_white = np.dot(test_pca_white, u.T)
        return sample_zca_white, test_zca_white

    return sample_reduce, test_reduce


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


def valid(train_in, train_out, n):
    samples, features = train_in.shape
    samples_random = random.sample(range(samples), samples)
    acc_all = []
    k_range = [1, 3, 5, 7]
    for k in k_range:
        acc = []
        for i in range(n):
            valid_train_in = []
            valid_train_out = []
            valid_test_in = []
            valid_test_out = []

            valid_train_in.append(train_in[samples_random[:i * samples / n], :])
            valid_train_in.append(train_in[samples_random[(i + 1) * samples / n:], :])
            valid_train_out.append(train_out[samples_random[:i * samples / n], :])
            valid_train_out.append(train_out[samples_random[(i + 1) * samples / n:], :])

            valid_test_in.append(train_in[samples_random[i * samples / n:(i + 1) * samples / n], :])
            valid_test_out.append(train_out[samples_random[i * samples / n:(i + 1) * samples / n], :])

            valid_train_in_ = np.concatenate(valid_train_in)
            valid_train_out_ = np.concatenate(valid_train_out)
            valid_test_in_ = np.concatenate(valid_test_in)
            valid_test_out_ = np.concatenate(valid_test_out)

            nn = Knn()
            nn.train(valid_train_in_, valid_train_out_)
            out = nn.predict(valid_test_in_, valid_test_out_, k)
            acc.append(out[1])
        print 'k=%d accuracy is %s' % (k, acc)
        acc_all.append(np.mean(acc))
    print acc_all
    print 'max valiation accuracy is %s' % np.max(acc_all)
    k_out = k_range[np.argmax(acc_all)]
    print 'best k is %d' % k_out
    return k_out


class AutoEncoder:
    """
    auto_encoder using L_BFGS training algorithm
    """
    def __init__(self, hidden_size, filename=None, lambda_=3e-3, sparsity_param=0.05, beta=3, max_iter=400):
        self.features = 0
        self.hidden_size = hidden_size
        self.w1 = []
        self.b1 = []
        self.samples = 0
        self.lambda_ = lambda_
        self.sparsity_param = sparsity_param
        self.beta = beta
        self.max_iter = max_iter
        if filename is not None:
            self.load_net(filename)

    def train(self, input_data, valid_grad=0):
        """
        train auto-encoder with L_BFGS algorithm
        :param input_data: numpy_array(n x m), n_features with m_samples
        :param valid_grad: whether to validate gradient
        :return: none, set best theta to class variable
        """
        self.features, self.samples = input_data.shape
        theta = self.init_theta()
        if valid_grad:
            cost, grad = self.cost_func(theta, input_data, self.lambda_, self.sparsity_param, self.beta)
            num_grad = self.valid_grad(self.cost_func, theta, input_data)
            err = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
            if err <= 1e-8:
                print 'validate successfully'
            else:
                raise 'validate failed'

        res = opt.minimize(self.cost_func, theta,
                           args=(input_data, self.lambda_, self.sparsity_param, self.beta),
                           method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': self.max_iter})
        self.w1, _, self.b1, _ = self.unroll_theta(res.x[:, None])
        print res

    def get_features(self, input_data):
        """
        :param input_data: numpy_array(n x m), n_features with m_samples
        :return: numpy_array(k x m), auto-encoder k_features
        """
        return sigmoid(np.dot(self.w1, input_data) + self.b1)

    def cost_func(self, theta, input_data, lambda_, sparsity_param, beta):
        n, m = input_data.shape
        w1, w2, b1, b2 = self.unroll_theta(theta)
        # forward
        z2 = np.dot(w1, input_data) + np.reshape(b1, (self.hidden_size, 1))
        a2 = sigmoid(z2)
        h = np.dot(w2, a2) + np.reshape(b2, (self.features, 1))
        p = np.sum(a2, 1) / m
        sparsity = np.sum(sparsity_param * np.log(sparsity_param / p) +
                          (1 - sparsity_param) * np.log((1 - sparsity_param) / (1 - p)))
        cost = 0.5 / m * np.sum(np.power(h - input_data, 2)) + 0.5 * lambda_ * (np.sum(np.power(w1, 2)) +
                                                                                np.sum(
                                                                                    np.power(w2, 2))) + beta * sparsity
        # backward
        delta3 = h - input_data
        temp = beta * (-sparsity_param / p + (1 - sparsity_param) / (1 - p))
        delta2 = np.dot(w2.T, delta3) + np.transpose([temp])
        delta2 = delta2 * a2 * (1 - a2)
        w2_grad = np.dot(delta3, a2.T) / m + lambda_ * w2
        b2_grad = np.transpose([np.sum(delta3, 1) / m])
        w1_grad = np.dot(delta2, input_data.T) / m + lambda_ * w1
        b1_grad = np.transpose([np.sum(delta2, 1) / m])

        grad = self.roll_theta(w1_grad, w2_grad, b1_grad, b2_grad)
        return cost, grad

    def init_theta(self):
        k = self.hidden_size
        n = self.features
        r = np.sqrt(6) / np.sqrt(n + k + 1)
        w1 = np.random.rand(k, n) * 2 * r - r
        w2 = np.random.rand(n, k) * 2 * r - r
        b1 = np.zeros((k, 1))
        b2 = np.zeros((n, 1))
        return self.roll_theta(w1, w2, b1, b2)

    def valid_grad(self, func, theta, input_data, epsilon=0.0001):
        num_grad = np.zeros(theta.shape)
        n = theta.size
        theta_p = theta + epsilon * np.eye(n)
        theta_m = theta - epsilon * np.eye(n)
        for j in range(n):
            num_grad[j] = (func(theta_p.T[j][:, None], input_data, self.lambda_, self.sparsity_param, self.beta)[0] -
                           func(theta_m.T[j][:, None], input_data, self.lambda_, self.sparsity_param, self.beta)[0]) / (
                              2 * epsilon)
        return num_grad

    def unroll_theta(self, theta):
        n = self.features
        k = self.hidden_size
        w1 = np.reshape(theta[:n * k], (k, n))
        w2 = np.reshape(theta[n * k:2 * n * k], (n, k))
        b1 = theta[2 * n * k:2 * n * k + k]
        b2 = theta[2 * n * k + k:]
        return w1, w2, b1, b2

    def roll_theta(self, w1, w2, b1, b2):
        n = self.features
        k = self.hidden_size
        theta = np.concatenate((np.reshape(w1, (n * k, 1)), np.reshape(w2, (n * k, 1))))
        theta = np.concatenate((theta, b1))
        theta = np.concatenate((theta, b2))

        return theta

    def save_net(self):
        scio.savemat('auto_encoder.mat', {'w1': self.w1, 'b1': self.b1})

    def load_net(self, filename):
        data = scio.loadmat(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.hidden_size, self.features = self.w1.shape


class Knn:
    """
    k-NearestNeighbor algorithm, default k=1
    """
    def __init__(self, k=1):
        self.k = k
        self.train_in = []
        self.train_out = []

    def __del__(self):
        pass

    def train(self, train_in_, train_out_):
        """
        :param train_in_: numpy_array, m_samples with n_features
        :param train_out_: numpy_array, m_samples with t_classes
        :return: none
        """
        self.train_in = train_in_
        self.train_out = train_out_

    def predict(self, test_in_, test_out=None, k=None):
        '''
        :param test_in_: numpy_array, m_samples with n_features
        :param test_out: numpy_array, m_samples with t_classes
        :param k: k numbers nearest neighbor
        :return: if test_out=None, -> res;
                else -> res, acc
        '''
        if k:
            num = k
        else:
            num = self.k
        res = np.zeros((test_in_.shape[0], 1))
        for i, rows in enumerate(test_in_):
            dist = np.sum(np.abs(self.train_in - rows), 1)
            min_index = np.argsort(dist)[:num]
            y = self.train_out[min_index, 0]
            res[i] = collections.Counter(y).most_common(1)[0][0]  # find the largest number of occurrences in y
        if test_out is not None:
            acc = np.mean(res == test_out)
            return res, acc
        else:
            return res


if __name__ == "__main__":
    train_in, train_out, test_in, test_out = load_data(0)
    #train_in, test_in = pca_mxn(train_in, test_in, 0.9)
    print train_in.shape, test_in.shape, train_out.shape, test_out.shape
    '''
    # auto_encoder
    auto_encoder = AutoEncoder(345)
    # auto_encoder.train(train_in)
    # auto_encoder.save_net()
    auto_encoder.load_net('auto_encoder.mat')
    train_in = auto_encoder.get_features(train_in.T)
    test_in = auto_encoder.get_features(test_in.T)
    print train_in.shape, test_in.shape
    '''

    nn = Knn(5)
    #nn.k = valid(train_in, train_out, 5)

    train_start = time.clock()
    nn.train(train_in, train_out)
    train_end = time.clock()

    test_start = time.clock()
    res, acc = nn.predict(test_in, test_out)
    test_end = time.clock()

    print res, acc
    print 'Training Run time: %s Seconds' % (train_end - train_start)
    print 'Testing Run time: %s Seconds' % (test_end - test_start)

