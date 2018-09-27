import os
import time
import numpy as np
import matplotlib.pyplot as plt

#*****************************************************************************
#Getting raw data
#*****************************************************************************
#Working with
def Get_Data():
    fname = 'breast-cancer-wisconsin.data'
    data = np.loadtxt(fname, delimiter=',') #Removed instances of missing data
    data = data.reshape(683,11)
    data_labels = data[:,(10)]
    data_labels[4]
    data = data[:,(1,2,3,4,5,6,7,8,9)]
    return data, data_labels
'''
Logistic regression
Given x, want y$ = P(y=1|x)

output y$ = sigmoid(w.T * x + b)

w = weights
x = input
sigmoid(z) =  1 / (1 + e^(-z))

'''
def Vectorized_regression(data, data_labels, epocs):

    n = np.size(data,0)
    feature_size = np.size(data,1)
    testing_size = n-int(n*.1)

    x = data[0:n-int(n*.1),]
    y = data_labels[0:testing_size,].reshape(testing_size,1)

    m = np.size(x,0)

    x_validation = data[testing_size: n,]
    y_validation = data_labels[testing_size: n,]

    w = np.zeros(feature_size).reshape(1,feature_size)
    w
    b = np.zeros(1)


    #foward propogation
    for j in range(epocs):
        print('Epoc: ' + str(j))
        J = 0
        X = x.T
        X
        X.shape
        w.shape
        z = np.dot(w,X) + b
        z.shape
        z

        A = sigmoid(z)
        A.shape
        A

        Y = y.T
        Y.shape


        J = (1.0/m) * np.sum(cost_fuct(A,Y))
        J


        dZ = A - Y
        dZ.shape
        X.shape

        dW = (1.0/m)*(np.dot(X,dZ.T))
        dW.shape

        db = (1.0/m)*np.sum(dZ)
        db

        alpha = .1      #Learning rate
        w
        w = w - alpha* dW.T
        w
        w.shape
        b = b - alpha*db
        b


        print ('cost: ' + str(J))
        #Validation
        accuracy = 0.0
        for i in range(np.size(x_validation,0)):
            X = x_validation[i]

            z =  np.dot(w,X) + b        #vectorization


            A = sigmoid(z)

            if (A >= .5 and y_validation[i] == 1):
                accuracy+=1.0
            elif( A < .5 and y_validation[i] == 0):
                accuracy+=1.0

        accuracy /= np.size(x_validation,0)
        print("validation acc: " + str(accuracy))

def regression(data, data_labels, epocs):

    n = np.size(data,0)
    feature_size = np.size(data,1)
    testing_size = n-int(n*.1)

    x = data[0:n-int(n*.1),]
    y = data_labels[0:testing_size,]
    m = np.size(x,0)

    x_validation = data[testing_size: n,]
    y_validation = data_labels[testing_size: n,]

    w = np.zeros(feature_size).reshape(1,feature_size)
    w
    b = np.zeros(1)


    #foward propogation
    for j in range(epocs):
        J = 0
        print('Epoc: ' + str(j))
        for i in range(m):

            #foward propogation
            #z = wx + b
            X = x[i]
            X
            z =  np.dot(w,X) + b        #vectorization
            z

            #print(z)
            A = sigmoid(z)
            A


            J += cost_fuct(A,y[i])
            J
            #print('Input z: '+str(z)+ ' Output: ' +str(A) + ' Cost ' +str(J))
            #backward propogation
            Y = y[i].T
            Y
            dZ = A - Y
            dZ

            dW = (1.0/m)*(X*dZ)
            dW
            db = (1.0/m)*np.sum(dZ)

            alpha = 1
            w = w - alpha*dW
            b = b - alpha*db
        print ('cost: ' + str((1.0/m)*J))
        #Validation
        accuracy = 0.0
        for i in range(np.size(x_validation,0)):
            X = x_validation[i]
            z =  np.dot(w,X) + b        #vectorization
            A = sigmoid(z)

            if (A >= .5 and y_validation[i] == 1):
                accuracy+=1.0
            elif( A < .5 and y_validation[i] == 0):
                accuracy+=1.0

        accuracy /= np.size(x_validation,0)
        print('validation acc: ' +str(accuracy))

def sigmoid(z):
    z
    h = 1.0 + np.exp(-z)
    h
    sig = 1.0 /h

    sig
    return sig

'''
Logistic regression cost function
given {(x^1,y^1),....,(x^m,y^m)}, want y$ approximately y^i

Loss(error) function
L(y$, y) = -(ylog(y$) + (1-y)log(1-y$))

Cost function
J(w,b) = 1/m Sigma[(i =1) -> m] L(y$^i, y^i)
'''
def cost_fuct(A,Y):
    A = A - .000000001
    error = -(Y * np.log(A) +(1.0-Y)*np.log(1.0-A))
    return error

'''
repeat {
    w := w - alpha*[(dJ(w,b)/dw)]
    b := b - alpha*[(dJ(w,b)/db)]
    where apla is the learning rate

}
'''
def gradient_descent():
    pass

def fix_labels(y):

    for i in range(np.size(y,0)):
        if y[i] == 2:
            y[i] = 0.0
        else:
            y[i] = 1
    return y

if __name__ == '__main__':
    data, data_labels = Get_Data()
    data_labels
    data_labels = fix_labels(data_labels)
    data_labels

    tic1 = time.time()
    print('None Vectorized_regression: ')
    regression(data,data_labels,epocs=100)
    toc1 = time.time()
    tic2 = time.time()
    print('\nVectorized regression: ')
    Vectorized_regression(data,data_labels,epocs=100)
    toc2 = time.time()

    print("None vectorized time: " + str(1000*(toc1 - tic1) ) + 'ms')
    print("Vectorized time: " + str(1000*(toc2 - tic2) ) + 'ms')
