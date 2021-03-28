def logistic(x):
    return 1/(1+np.exp(-x))

class LogRegClassifier(NumpyClassifier):
    def fit(self, X_train, t_train, eta = 0.1, epochs=10,diff=0.001):
        (k,m) = X_train.shape
        X_train = add_bias(X_train)

        self.weights = weights = np.zeros(m+1)

        #defining the update with a large number
        update = 100

        self.run = 0

        #the while loops goes on until the update is less than diff
        while update > diff:
            oldMSE = MSE(t_train,X_train @ weights)
            weights -= eta / k *  X_train.T @ (self.forward(X_train) - t_train)
            newMSE = MSE(t_train,X_train @ weights)
            update = oldMSE - newMSE
            self.run += 1

    def forward(self,X_train):
        z = X_train @ self.weights
        return logistic(z)

    def predict(self,X):
        z = add_bias(X)
        score = self.forward(z)
        return (score>0.5).astype('int')

#training and validating the data
def runingLogReg(X_train,t2_train,X_val,t2_val,printing):
    print("Running the logistic regression classifier")
    bestAccLog = 0
    best_diff = 0

    for i in range(1,10,1):
        d = 1/10**(i)
        LogReg = LogRegClassifier()
        LogReg.fit(X_train,t2_train,diff=d)
        accuracy = LogReg.accuracy(X_val,t2_val)
        if printing == True:
            print(f'diff = {d:.9f} | accuracy = {accuracy:.5f} | runs = {LinReg.run}')

        if accuracy > bestAccLog:
            bestAccLog = accuracy
            best_diff = d
    return bestAccLog,best_diff

bestAccLog,best_diff = runingLogReg(X_train,t2_train,X_val,t2_val,True)
print(f"Best accuracy: {bestAccLog:4.5f} with diff = {best_diff}")

#To modify the target values from scalars to arrays, we use the following technique (look at output)

t = t_train[0:10]
a = (t_train[0:10]==0).astype('int')
b = (t_train[0:10]==1).astype('int')
c = (t_train[0:10]==2).astype('int')
print("t-vector -> t-matrix")
print()
for i in range(10):
    print("|",t[i],"|"," -> |",a[i]," " ,b[i]," ",c[i],"|")

#training and finding the acccuracy of the classifier

def oneVsRest(X_train,t_train,X_val,t_val):
    bestAccuracy = -1
    for i in range(3):
        #converting from scalar to array:
        t_train1 = (t_train==i).astype('int')
        t_val1 = (t_val==i).astype('int')

        #fitting and calculating accuracy:
        logReg = LogRegClassifier()
        logReg.fit(X_train,t_train1)
        accuracy = logReg.accuracy(X_val,t_val1)

        if accuracy > bestAccuracy:
            bestAccuracy= accuracy
    return bestAccuracy

bestAccuracy = oneVsRest(X_train,t_train,X_val,t_val)

print(f'Best accuracy for one-vs-rest approach: {bestAccuracy}')
