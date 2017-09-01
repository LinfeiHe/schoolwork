"""
Introduction: NaiveBayes algorithm demon, <<Statistical learning method>> 4.2.2
Author: helinfei@ecust
"""


import numpy as np
import collections

S = 1
M = 2
L = 3
x1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
x2 = np.array([S, M, M, S, S, S, M, M, L, L, L, M, M, L, L])
y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
x = np.concatenate([[x1], [x2]])


class NaiveBayes(object):
    def __init__(self):
        self.plk = {} #store p(xi=l|y=k)
        self.k = [] # store k_class values
        self.py = {} # store p(y=k)

    def train(self, x, y):
        n, m = x.shape # n_features, m_samples
        y_k = collections.Counter(y) # counter k_class in y
        py = np.zeros(len(y_k)) # py has k elements
        for k in y_k:
            self.k.append(k)
            self.py[k] = y_k[k] / float(len(y))
        for i, rows in enumerate(x):
            xi_l = collections.Counter(x[i, :]) # counter l different values in x[i, :]
            for l in xi_l:
                for k in y_k:
                    temp1 = rows == l
                    temp2 = y == k
                    count = 0
                    for ii in range(m):
                        if temp1[ii] == True and temp2[ii] == True:
                            count += 1
                    self.plk['%d%d%d' %(i,l,k)] = count / (self.py[k] * m) # p(xi=l|y=k)

    def predict(self, x):
        n, m = x.shape
        ck = np.zeros(m)
        for index, rows in enumerate(x.T):
            max = 0
            for k in self.k:
                temp = 1
                for i in range(n):
                    temp *= self.plk['%s%s%s' % (i, x[i][index], k)]
                res = self.py[k] * temp
                if res >= max:
                    max = res
                    ck[index] = k
        return ck

if __name__ == '__main__':
    cls = NaiveBayes()
    cls.train(x, y)
    x = np.array([[2], [S]])
    res = cls.predict(x)
    print res