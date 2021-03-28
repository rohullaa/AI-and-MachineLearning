from collections import Counter

def majority(a):
    counts = Counter(a)
    return counts.most_common()[0][0]

def distance_L2(a, b):
    s = sum((x - y) ** 2 for (x,y) in zip(a,b))
    return s ** 0.5

class PyClassifier():
    def accuracy(self,X_test, y_test, **kwargs):
        predictedValues = []
        for a in X_test:
            var = self.predict(a,**kwargs)
            predictedValues.append(var)
        equal = []
        for b,c in zip(predictedValues,y_test):
            if b == c:
                equal.append((b,c))
        accuracyVariable = len(equal)/len(y_test)
        return accuracyVariable

class kNNClassifier(PyClassifier):
    def __init__(self, k=3, dist=distance_L2):
        self.k = k
        self.dist = dist

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, a):
        X = self.X_train
        y = self.y_train
        distances = [(self.dist(a, b), b, c) for (b, c) in zip(X, y)]
        distances.sort()
        predictors = [c for (_,_,c) in distances[0: k]]
        return majority(predictors)
print("Running the kNN classifier")
best_k = 0
bestAccKnn = 0

for k in range(1,20):
    knnCL = kNNClassifier(k=k)
    knnCL.fit(X_train,t2_train)
    acc = knnCL.accuracy(X_val,t2_val)
    if True:
        print(f'k = {k:4.1f} | Accuracy = {acc:.5f}')
    if acc > bestAccKnn:
        bestAccKnn = acc
        best_k = k
print(f"Best accuracy: {bestAccKnn:4.5f} at k = {best_k}")

print("Running the kNN classifier")
bestAcc = 0
bestK = 0

for k in range(1,20):
    knnCL = kNNClassifier(k=k)
    knnCL.fit(X_train,t_train)
    acc = knnCL.accuracy(X_val,t_val)
    print(f'k = {k:4.1f} | Accuracy = {acc:.5f}')
    if acc > bestAcc:
        bestAcc = acc
        bestK = k

print()
print(f"Best accuracy: {bestAcc:4.5f} at k = {bestK}")
