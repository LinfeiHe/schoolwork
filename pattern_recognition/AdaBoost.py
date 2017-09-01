"""
Introduction: AdaBoost algorithm demon, <<Statistical learning method>> 8.1.3
Author: helinfei@ecust
"""


import numpy as np

# create data
x = np.linspace(0, 9, 10)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])


class weak_classifier(object):
    def __ini__(self):
        self.v = 0
        self.left = 1

    def base_classifier(self, x, y, weights):
        y_pred = np.zeros(np.size(x))
        max_acc1 = 0
        max_acc2 = 0
        for v in x:
            for k in range(len(x)):
                if x[k] <= v:
                    y_pred[k] = 1
                else:
                    y_pred[k] = -1
            acc = np.mean(np.dot(y_pred == y, weights))
            if acc > max_acc1:
                best_v1 = v
                max_acc1 = acc
        for v in x:
            for k in range(len(x)):
                if x[k] <= v:
                    y_pred[k] = -1
                else:
                    y_pred[k] = 1
            acc = np.mean(np.dot(y_pred == y, weights))
            if acc > max_acc2:
                best_v2 = v
                max_acc2 = acc
        if max_acc1 >= max_acc2:
            self.left = 1
            return best_v1, max_acc1, 1
        else:
            self.left = -1
            return best_v2, max_acc2, -1

    def base_classifier_pred(self, x, v, left):
        x_temp = x.copy()
        x_temp[x <= v] = left
        x_temp[x > v] = -left
        return x_temp


class AdaBoost(object):
    def __init__(self, base_cls):
        self.base_cls = base_cls
        self.final_cls = []
        self.alpha = []
        self.left = []

    def train(self, x, y):
        weights = np.ones(np.size(x)) / len(x)
        res = []
        alpha = []
        while 1:
            v, acc, left = self.base_cls.base_classifier(x, y, weights)
            y_pred = self.predict(x)
            if np.mean(y_pred == y) == 1:
                break
            err = 1 - acc
            alpha_temp = 0.5 * np.log(acc / err)

            alpha.append(alpha_temp)
            res.append(v)

            weights = weights * np.exp(-alpha_temp * y * self.base_cls.base_classifier_pred(x, v, left))
            weights /= np.sum(weights)
            self.final_cls = res
            self.alpha = alpha
            self.left.append(left)

    def predict(self, x):
        y_pred = 0
        for i in range(len(self.alpha)):
            y_pred += self.alpha[i] * self.base_cls.base_classifier_pred(x, self.final_cls[i], self.left[i])
        return np.sign(y_pred)


if __name__ == '__main__':
    classifier = AdaBoost(weak_classifier())
    classifier.train(x, y)
    print 'result:', classifier.predict(x)
    print 'alpha[i]:', classifier.alpha
    print 'weak classifier, separate point:', classifier.final_cls
    print 'classifier.left:', classifier.left
