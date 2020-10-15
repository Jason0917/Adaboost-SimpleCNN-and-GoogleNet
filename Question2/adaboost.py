from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math


class AdaBoost:
    def __init__(self, T=50, weakClf=DecisionTreeClassifier()):
        self.T = T
        self.alpha = []
        self.classifierList = []
        for t in range(0, self.T):
            self.classifierList.append(weakClf)

    def train(self, X, Y):
        data_size = X.shape[0]
        D = np.full(data_size, 1 / data_size)
        for t in range(0, self.T):
            clf = self.classifierList[t]
            clf = clf.fit(X, Y, D)
            y_pre = clf.predict(X)
            # calculate error value
            error = 1e-5
            for i in range(data_size):
                if y_pre[i] != Y[i]:
                    error += D[i]
            alphat = 0.5 * math.log((1 - error) / error)
            self.alpha.append(alphat)
            # update D
            Z_t = 0
            for k in range(data_size):
                Z_t += D[k] * math.exp(-alphat * y_pre[k] * Y[k])
            for j in range(data_size):
                D[j] = (D[j] * math.exp(-alphat * y_pre[j] * Y[j])) / Z_t

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for t in range(self.T):
            y_pre = self.alpha[t] * self.classifierList[t].predict(X)
            y = y + y_pre

        for i in range(X.shape[0]):
            if y[i] > 0:
                y[i] = 1
            else:
                y[i] = -1

        return y


def main():
    # if x>0, y=1, else y=-1
    x = np.array([-1, -2, -3, -4, -5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, -1, -2, -3, 4, 5])
    y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    adaboost = AdaBoost(weakClf=DecisionTreeClassifier())
    adaboost.train(X_train.reshape(-1, 1), y_train)
    y_pred = adaboost.predict(X_test.reshape(-1, 1))

    loss = (y_test - y_pred).sum()
    print(loss)


if __name__ == '__main__':
    main()
