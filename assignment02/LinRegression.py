import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import random

from sklearn.datasets import make_blobs
X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]], n_features=2, random_state=2019)
indices = np.arange(X.shape[0])
random.seed(2020)
random.shuffle(indices)
indices[:10]

X_train = X[indices[:800],:]
X_val = X[indices[800:1200],:]
X_test = X[indices[1200:],:]
t_train = t[indices[:800]]
t_val = t[indices[800:1200]]
t_test = t[indices[1200:]]

t2_train = t_train == 1
t2_train = t2_train.astype('int')
t2_val = (t_val == 1).astype('int')
t2_test = (t_test == 1).astype('int')

#Plotting the training sets
def plotting(X, y, marker='.'):
    labels = set(y)
    for lab in labels:
        plt.plot(X[y == lab][:, 1], X[y == lab][:, 0],marker, label="class {}".format(lab))
    plt.legend()
    plt.show()


class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""

    def accuracy(self, X_test, y_test, **kwargs):
        pred = self.predict(X_test, **kwargs)
        if len(pred.shape) > 1:
            pred = pred[:,0]
        return sum(pred == y_test) / len(pred)


def add_bias(X):
    sh = X.shape
    if len(sh) == 1:
        return np.concatenate([np.array([1]), X])
    else:
        m = sh[0]
        bias = np.ones((m, 1))
        return np.concatenate([bias, X], axis = 1)

def mse(y, y_pred):
    sum_errors = 0.
    for i in range(0,len(y)):
        sum_errors += (y[i] - y_pred[i])**2
    mean_squared_error = sum_errors/len(y)
    return mean_squared_error

class NumpyLinRegClass(NumpyClassifier):

    def fit(self, X_train, t_train, gamma=0.1, epochs=1, diff=0.00001):
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""

        (k, m) = X_train.shape
        X_train = add_bias(X_train)

        self.theta = theta = np.zeros(m+1)
        update = 1

        while update > diff:
            error = mse(t_train, X_train @ theta)
            theta -= gamma / k *  X_train.T @ (X_train @ theta - t_train)
            next_error = mse(t_train, X_train @ theta)
            update = error - next_error
            epochs+=1


    def predict(self, x, threshold=0.5):
        z = add_bias(x)
        score = z @ self.theta
        return score > threshold

#training and validating the data
def runingLinReg(X_train,t2_train,X_val,t2_val,printing):
    print("Running the linear regression classifier")

    bestAcc_Lin = 0
    diffLin = 0
    for i in range(1,10,1):
        d = 1/10**(i)
        LinReg = LinRegClassifier()
        LinReg.fit(X_train,t2_train,diff=d)
        accuracy = LinReg.accuracy(X_val,t2_val)
        if printing == True:
            print(f'diff = {d:.9f} | accuracy = {accuracy:.5f} | runs = {LinReg.run}')

        if accuracy > bestAcc_Lin:
            bestAcc_Lin = accuracy
            diffLin = d

    return bestAcc_Lin,diffLin

bestAcc_Lin,diffLin = runingLinReg(X_train,t2_train,X_val,t2_val,True)
print(f"Best accuracy: {bestAcc_Lin:4.5f} with diff = {diffLin}")
