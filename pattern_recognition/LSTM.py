"""
Introduction: Minimal character-level LSTM model.
Author: helinfei@ecust
Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LSTM(object):
    def __init__(self, sequence_size, hidden_size, learning_rate=0.1, max_iter=1000):
        self.sequence_size = sequence_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        # hidden variable
        self.W = np.random.randn(4 * hidden_size, sequence_size) * 0.01
        self.U = np.random.randn(4 * hidden_size, hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        # y_pred variable
        self.V = np.random.randn(sequence_size, hidden_size) * 0.01
        self.by = np.zeros((sequence_size, 1))
        self.theta = self.roll_theta(self.W, self.U, self.b, self.V, self.by)

    def train(self, inputs, labels, h_pre, c_pre, valid_grad=1):
        theta = self.roll_theta(self.W, self.U, self.b, self.V, self.by)
        if valid_grad:
            h_pre = np.zeros((self.hidden_size, 1))
            c_pre = np.zeros((self.hidden_size, 1))
            _, grad, _, _, _ = self.loss_func(inputs, labels, h_pre, c_pre)
            num_grad = self.valid_grad(theta, inputs, labels, h_pre, c_pre)
            err = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
            print err
            if err <= 1e-8:
                print 'validate successfully'
            else:
                raise 'validate failed'
        # adam optimization algorithm
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        for n in xrange(self.max_iter):
            loss, grad, ps, h_pre, c_pre = self.loss_func(inputs, labels, h_pre, c_pre)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            theta += - self.learning_rate * m / (np.sqrt(v) + eps)
            self.W, self.U, self.b, self.V, self.by = self.unroll_theta(theta)
            print 'loss is %s' % loss
            print '_______iter: %s ______' % n

    def predict(self, test_in, test_out):
        y_pred = self.loss_func(test_in, test_out, np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1)))[2]
        count = 0
        res = []
        for t in xrange(len(test_in)):
            temp = np.argmax(y_pred[t])
            res.append(temp)
            if temp == test_out[t]:
                count += 1
        acc = count / len(test_in)
        return res, acc

    def loss_func(self, inputs, labels, h_pre, c_pre):
        # x state, hidden state, cell state, y state, probability state
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        gate_i, gate_f, gate_o, a = {}, {}, {}, {}
        hs[-1] = np.copy(h_pre)
        cs[-1] = np.copy(c_pre)
        loss = 0
        # forward
        for t in xrange(len(inputs)):
            xs[t] = np.zeros((self.sequence_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            total_gate = np.dot(np.concatenate((self.W, self.U, self.b), 1), np.concatenate((xs[t], hs[t - 1], [[1]])))
            a[t] = np.tanh(total_gate[0:self.hidden_size])
            gate_i[t] = sigmoid(total_gate[self.hidden_size:2 * self.hidden_size])
            gate_f[t] = sigmoid(total_gate[2 * self.hidden_size:3 * self.hidden_size])
            gate_o[t] = sigmoid(total_gate[3 * self.hidden_size:4 * self.hidden_size])
            cs[t] = gate_f[t] * cs[t - 1] + gate_i[t] * a[t]
            hs[t] = gate_o[t] * np.tanh(cs[t])
            ys[t] = np.dot(self.V, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][labels[t]])
        # backward
        dW, dU, db = np.zeros_like(self.W), np.zeros_like(self.U), np.zeros_like(self.b)
        dV, dby = np.zeros_like(self.V), np.zeros_like(self.by)
        dhnext, fnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(gate_f[0]), np.zeros_like(cs[0])
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[labels[t]] -= 1
            dV += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.V.T, dy) + dhnext
            dc = dh * gate_o[t] * (1 - np.tanh(cs[t]) ** 2) + dcnext * fnext
            da = dc * gate_i[t] * (1 - a[t] ** 2)
            di = dc * a[t] * gate_i[t] * (1 - gate_i[t])
            df = dc * cs[t - 1] * gate_f[t] * (1 - gate_f[t])
            do = dh * np.tanh(cs[t]) * gate_o[t] * (1 - gate_o[t])
            dgates = np.concatenate((da, di, df, do), 0)
            dhnext = np.dot(self.U.T, dgates)
            fnext = gate_f[t]
            dcnext = dc
            dW += np.dot(dgates, xs[t].T)
            dU += np.dot(dgates, hs[t - 1].T)
            db += dgates
        for dparam in [dW, dU, db, dV, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        grad = self.roll_theta(dW, dU, db, dV, dby)
        return loss, grad, ps, hs[len(inputs) - 1], cs[len(inputs) - 1]

    def valid_grad(self, theta, inputs, labels, h_pre, c_pre, epsilon=0.00001):
        num_grad = np.zeros(theta.shape)
        n = theta.size
        theta1 = theta + epsilon * np.eye(n)
        theta2 = theta - epsilon * np.eye(n)
        for j in range(n):
            self.W, self.U, self.b, self.V, self.by = self.unroll_theta(theta1.T[j][:, None])
            y1 = self.loss_func(inputs, labels, h_pre, c_pre)
            self.W, self.U, self.b, self.V, self.by = self.unroll_theta(theta2.T[j][:, None])
            y2 = self.loss_func(inputs, labels, h_pre, c_pre)
            num_grad[j] = (y1[0] - y2[0]) / (2 * epsilon)
        return num_grad

    def unroll_theta(self, theta):
        W = np.reshape(theta[:4 * self.hidden_size * self.sequence_size], self.W.shape)
        U = np.reshape(theta[4 * self.hidden_size * self.sequence_size:
            4 * self.hidden_size * (self.sequence_size + self.hidden_size)], self.U.shape)
        b = np.reshape(theta[4 * self.hidden_size * (self.sequence_size + self.hidden_size):
            4 * self.hidden_size * (self.sequence_size + self.hidden_size + 1)], self.b.shape)
        V = np.reshape(theta[4 * self.hidden_size * (self.sequence_size + self.hidden_size + 1):
            4 * self.hidden_size * (self.sequence_size + self.hidden_size + 1) +
            self.sequence_size * self.hidden_size], self.V.shape)
        by = np.reshape(theta[4 * self.hidden_size * (self.sequence_size + self.hidden_size + 1) +
                              self.sequence_size * self.hidden_size:], self.by.shape)
        return W, U, b, V, by

    def roll_theta(self, dW, dU, db, dV, dby):
        temp = np.concatenate((dW.flatten(), dU.flatten(), db.flatten(), dV.flatten(), dby.flatten()))
        return np.array([temp]).T

    def sample(self, h_pre, c_pre, seed_ix, sequence_length):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((self.sequence_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in xrange(sequence_length):
            total_gate = np.dot(np.concatenate((self.W, self.U, self.b), 1), np.concatenate((x, h_pre, [[1]])))
            a = np.tanh(total_gate[0:self.hidden_size])
            gate_i = sigmoid(total_gate[self.hidden_size:2 * self.hidden_size])
            gate_f = sigmoid(total_gate[2 * self.hidden_size:3 * self.hidden_size])
            gate_o = sigmoid(total_gate[3 * self.hidden_size:4 * self.hidden_size])
            cs = gate_f * c_pre + gate_i * a
            hs = gate_o * np.tanh(cs)
            ys = np.dot(self.V, hs) + self.by
            ps = np.exp(ys) / np.sum(np.exp(ys))
            ix = np.random.choice(range(self.sequence_size), p=ps.ravel())
            x = np.zeros((self.sequence_size, 1))
            x[ix] = 1
            ixes.append(ix)
            h_pre = hs
            c_pre = cs
        return ixes

SEQ_LENGTH = 25
HIDDEN_SIZE = 100
cache = 0
if __name__ == '__main__':
    # data I/O
    data = open('test2.py', 'r').read()  # should be simple plain text file
    chars = list(set(data))
    print chars
    data_size, vocab_size = len(data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # lstm iteration
    p = 0
    n = 0
    hprev = np.zeros((HIDDEN_SIZE, 1))
    cprev = np.zeros((HIDDEN_SIZE, 1))
    lstm = LSTM(vocab_size, HIDDEN_SIZE)
    smooth_loss = -np.log(1.0 / vocab_size) * SEQ_LENGTH
    while True:
        # read data
        if p + SEQ_LENGTH + 1 >= len(data):
            hprev = np.zeros((HIDDEN_SIZE, 1))  # reset RNN memory
            p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p + SEQ_LENGTH]]
        labels = [char_to_ix[ch] for ch in data[p + 1:p + SEQ_LENGTH + 1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = lstm.sample(hprev, cprev, inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print '----\n %s \n----' % (txt,)

        # forward seq_length characters through the net and fetch gradient
        loss, grad, ps, hprev, cprev = lstm.loss_func(inputs, labels, hprev, cprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss)  # print progress

        # perform parameter update with Adagrad
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        loss, grad, ps, hprev, cprev = lstm.loss_func(inputs, labels, hprev, cprev)
        cache += grad ** 2
        lstm.theta += - lstm.learning_rate * grad / (np.sqrt(cache) + eps)
        lstm.W, lstm.U, lstm.b, lstm.V, lstm.by = lstm.unroll_theta(lstm.theta)


        p += SEQ_LENGTH  # move data pointer
        n += 1  # iteration counter