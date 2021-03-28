import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import random
from sklearn.datasets import make_blobs
X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]],n_features=2, random_state=2019)

indices = np.arange(X.shape[0])
random.seed(2020)
random.shuffle(indices)

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

eta = 0.01
dim_hidden = 6
dim_in =  X_train.shape[1]
dim_out = len(set(t_train))

def logistic(x):
    return 1/(1+np.exp(-x))

#tranforming the t_train from (800,1) -> (800,3)
def transforming(t_train):
    t_train_modified = np.zeros((len(t_train), len(set(t_train))))
    if len(set(t_train)) == 3:
        for t in t_train:
            if t==0:
                i = np.where(t_train==t)
                t_train_modified[i] = [1,0,0]
            elif t==1:
                i = np.where(t_train==t)
                t_train_modified[i] = [0,1,0]
            elif t==2:
                i = np.where(t_train==t)
                t_train_modified[i] = [0,0,1]
        return t_train_modified
    else:
        for t in t_train:
            if t==0:
                i = np.where(t_train==t)
                t_train_modified[i] = [1,0]
            elif t==1:
                i = np.where(t_train==t)
                t_train_modified[i] = [0,1]
        return t_train_modified

#function to scale the data
def scaling(X,X_train=X):
    m = np.mean(X_train)
    s = np.std(X_train)
    return (X-m)/s

class MNNClassifier():
    """A multi-layer neural network with one hidden layer"""

    def __init__(self,eta = 0.01, dim_hidden = 6):
        """Initialize the hyperparameters"""
        self.eta = eta
        self.dim_hidden = dim_hidden

    def fit(self, X_train, t_train, epochs = 100):
        """Initialize the weights. Train *epochs* many epochs."""

        # Initilaization
        self.dim_in = dim_in= X_train.shape[1]
        self.dim_out = dim_out = len(set(t_train))

        #weights:
        self.weight1 = weight1 = np.random.rand(dim_in,self.dim_hidden)
        self.weight2 = weight2 = np.random.rand(self.dim_hidden,dim_out)

        #biases:
        self.bias_1 = bias_1 = -np.ones(self.dim_hidden)
        self.bias_2 = bias_2 = -np.ones(dim_out)

        #transforming t_train from (800,) -> (800,3) dim
        self.t= t = transforming(t_train)

        for e in range(epochs):
            # Run one epoch of forward-backward
            hidden_activations,output_activations = self.forward(X_train)
            self.backward(hidden_activations,output_activations,X_train)

    def forward(self, X):
        """Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        input1 = self.bias_1 + (X @ self.weight1)
        hidden_activations = logistic(input1)

        input2 = self.bias_2 + (hidden_activations @ self.weight2)
        output_activations = logistic(input2)

        return hidden_activations,output_activations

    def backward(self,hidden_activations,output_activations,X):
        #computing delta

        self.delta_output=delta_output = (output_activations - self.t)*output_activations*(1-output_activations)
        self.delta_hidden=delta_hidden = hidden_activations * (1-hidden_activations) * (delta_output @ self.weight2.T)

        #updating the weights
        self.weight1 -= self.eta*(X.T @ delta_hidden)
        self.weight2 -= self.eta*(hidden_activations.T @ delta_output)

    def accuracy(self,X_test, y_test, **kwargs):

        predictedValues = []
        for a in X_test:
            var = self.predict(a,**kwargs)
            predictedValues.append(var)
        equal = []
        for b,c in zip(predictedValues,y_test):
            if b==c:
                equal.append((b,c))
        accuracyVariable = len(equal)/len(y_test)
        return accuracyVariable

    def predict(self,X):
        hidden_activations,output_activations = self.forward(X)
        #returning the max of the output_activations
        return output_activations.argmax()

#scaling the data:
X_train_scaled = scaling(X_train)
X_val_scaled = scaling(X_val,X_train)

mnnc = MNNClassifier()
mnnc.fit(X_train_scaled,t2_train)
a = mnnc.accuracy(X_val_scaled,t2_val)
print(a)
