import numpy as np
import sklearn.datasets

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

def class_mean_vector(D, L):
    mean_vector = []
    # ATTENTION! I used range(3) because i know there are 3 classes, but it's not the case all the time
    for i in set(list(L)):
        mean = D[:, L == i].mean(1).reshape(D.shape[0], 1)
        mean_vector.append(mean)
    #print(mean_vector)

    return np.hstack(mean_vector)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()[
        'target']
    return D, L

def compute_accuracy_error(predicted_labels, true_labels): #Lp is the predicted label and true_labels is the true label
    BooleanPredictionResults = predicted_labels == true_labels
    Success = sum(BooleanPredictionResults)
    Accuracy = Success / len(true_labels)
    Error = 1 - Accuracy
    #print("Accuracy: ", Accuracy, "Error: ", Error)   #TODO correct rounding 
    return Accuracy, Error
