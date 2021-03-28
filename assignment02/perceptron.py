class PerceptronClassifier(PyClassifier):
    """Simple perceptron python classifier"""

    def fit(self, X_train, y_train, eta=1, epochs=1):
        """Train the self.weights on the training data with learning
        rate eta, running epochs many epochs"""
        X_train = [[1] + list(x) for x in X_train]
        dim = len(X_train[0])
        weights = [0 for _ in range(dim)]
        self.dim = dim
        self.weights = weights

        for e in range(epochs):
            for x, t in zip(X_train,y_train):
                y = int(self.forward(x)>0)
                for i in range(dim):
                    weights[i] -= eta * (y - t) * x[i]

    def forward(self, x):
        """Calculate the score for the item x"""
        score = sum([self.weights[i]*x[i] for i in range(self.dim)])
        return score

    def predict(self, x):
        """Predict the value for the item x"""
        x = [1] + list(x)
        score = self.forward(x)
        return int(score > 0)

def runingPer(X_train,t2_train,X_val,t2_val,printing):
    print("Running the Perceptron classifier")
    bestAccPer = 0
    bestEpochPer = 0
    for i in range(1,20):
        perCL = PerceptronClassifier()
        perCL.fit(X_train,t2_train,epochs = i)
        acc = perCL.accuracy(X_val,t2_val)
        if printing == True:
            print(f"Epoch: {i:4.1f} | Accuracy = {acc}")
        if acc > bestAccPer:
            bestAccPer = acc
            bestEpochPer = i
    return bestAccPer,bestEpochPer

bestAccPer,bestEpochPer =runingPer(X_train,t2_train,X_val,t2_val,True)
print(f"Best accuracy: {bestAccPer:4.5f} after {bestEpochPer} epochs")

    
