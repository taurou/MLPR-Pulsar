import numpy as np
import matplotlib.pyplot as plt
import function_lib.lib as lib
import scipy.special
import math

#UTILITIES

def logpdf_GAU_ND(x, mu, c):
    M = x.shape[0]
    logdet_sign, logdet = np.linalg.slogdet(c) #returns the sign and the log-determinant
    const = -M*0.5*np.log(2*np.pi) -0.5*logdet
    c_inv = np.linalg.inv(c) #inversion of the covariance matrix
    return_val = [ const - 0.5*np.dot( np.dot((x[:,i:i+1]-mu).T, c_inv), x[:,i:i+1]-mu) for i in range (x.shape[1]) ]
    return np.array(return_val).ravel()

def compute_Mean_CovMatr(D):
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    D_centered = D - mu
    #computing the covariance matrix
    C = np.dot(D_centered, D_centered.T)/D_centered.shape[1]
    return mu, C

def ll(x, mu, c):
    x_gaussian_density = logpdf_GAU_ND(x,mu,c)
    return x_gaussian_density.sum()

def computeSbSw(D,L):  # returns Sb,Sw
    #computing means for the whole dataset and for each class
    mean_vector = []
    # ATTENTION! I used range(3) because i know there are 3 classes, but it's not the case all the time
    for i in set(list(L)):
        mean = D[:, L == i].mean(1).reshape(D.shape[0], 1)
        mean_vector.append(mean)
    #print(mean_vector)

    class_mean = np.hstack(mean_vector)
    overall_mean = D.mean(1).reshape(D.shape[0], 1)
    #print(class_mean, overall_mean)
    N = D.shape[1]  # number of total elements N
    # list containing the number of elements per class
    Nc = [D[:, L == i].shape[1] for i in set(list(L))]

    #between class covariance
    Sb = 0
    for i in set(list(L)):
        Sb = Sb + np.dot((class_mean[:, i].reshape(D.shape[0], 1)-overall_mean),
                            (class_mean[:, i].reshape(D.shape[0], 1)-overall_mean).T)*Nc[i]
    Sb = Sb / N
    #within class covariance
    Sw = 0
    for i in set(list(L)):  # numero di classi è 3
        for j in range(D[:, L == i].shape[1]):  # number of elements for this class
            x = D[:, L == i][:, j].reshape(D.shape[0], 1)
            Sw = Sw + np.dot((x-class_mean[:, i].reshape(D.shape[0], 1)),
                                (x-class_mean[:, i].reshape(D.shape[0], 1)).T)

    Sw = Sw / N
    return Sb, Sw



def KfoldCrossValidation(D, L, k, prior, classifierFunction, seed=0): #passing the classifier function
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) #randomizes the order of the data

    group_size = int(math.ceil((D.shape[1]/k)))
    goodEstimation = 0
    totalEstimation = 0

    for i in range(k):
        idxTest = idx[i*group_size:(i*group_size + group_size)]
        idxTrain = [i for i in idx if i not in set(idxTest)]
        predicted_L = classifierFunction(D[:,idxTrain], L[idxTrain], D[:,idxTest], L[idxTest], prior)
        BooleanPredictionResults = predicted_L == L[idxTest]
        goodEstimation += sum(BooleanPredictionResults)
        totalEstimation += len(predicted_L)
    
    Accuracy = goodEstimation / totalEstimation
    Error = 1 - Accuracy
    print("KFOLD Accuracy: ", Accuracy, "Error: ", Error)   #TODO correct rounding 


def LeaveOneOutCrossValidation(D, L, prior, classifierFunction, seed=0):
    KfoldCrossValidation(D, L, D.shape[1], prior, classifierFunction, seed)


#if diagonal is set to true, only the diagonal will be kept and the other elements will be zeroed

def compute_Mean_CovMatr_perClass(D, L, diagonal=False): #put diagonal = true optionally if we want to use just the diagonal of the cov matrix
    mean_cov_class = {}
    # generating a diagonal matrix with ones with numClass * numClass dimension
    for i in set(list(L)):  # potrebbero essere anche parole...
        (mu, C) = compute_Mean_CovMatr(D[:, L == i])
        if (diagonal):
            C = np.diag(np.diag(C))
        mean_cov_class[i] = (mu, C)
    return mean_cov_class
    
#takes as input the dictionary label : (mu, C) and the data.

def compute_logDensity_AllSamples(D, mu_C_tuple):
    matrix = []
    (mu, c) = mu_C_tuple
    matrix.append(logpdf_GAU_ND(D, mu, c))
    return np.array(matrix)


def MVG_basePredictor(eval_D, eval_L, mu_C, prior): #TODO correct eval_L ! It's never used!!

    logDens = np.zeros((len(mu_C), eval_D.shape[1]))
    logSJoint = np.zeros((len(mu_C), eval_D.shape[1]))
    
    for label, mean_covmatr in mu_C.items():
        logDens[label, :] = compute_logDensity_AllSamples(eval_D, mean_covmatr)
        logSJoint[label, :] =logDens[label, :] + np.log(prior[label, :])

    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    CPosterior = np.exp(logSPost)

    llr = logDens[1, :] - logDens[0, :] #compute llr between class 1 and class 0.
    # Computes the LABEL with max Class-Posterior-Probability
    predicted_labels = np.argmax(CPosterior, axis=0)
    return predicted_labels, llr #return the predicted labels 

    
def TiedCovariance_logMVG(train_D, train_L, eval_D, eval_L, prior):
    _, Sw = computeSbSw(train_D, train_L) #computing the empirical within covariance matrix
    class_means = lib.class_mean_vector(train_D, train_L)
    mu_C = { i:(lib.vcol(class_means[:,i:i+1]),Sw) for i in set(list(train_L))} #formatting the right input for the base function
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)

def TiedCovariance_logNB(train_D, train_L, eval_D, eval_L, prior):
    _, Sw = computeSbSw(train_D, train_L) #computing the empirical within covariance matrix
    Sw = np.diag(np.diag(Sw))
    class_means = lib.class_mean_vector(train_D, train_L)
    mu_C = { i:(lib.vcol(class_means[:,i:i+1]),Sw) for i in set(list(train_L))} #formatting the right input for the base function
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)

def logMVG(train_D, train_L, eval_D, eval_L, prior):
    mu_C = compute_Mean_CovMatr_perClass(train_D, train_L)
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)

def logNaiveBayes(train_D, train_L, eval_D, eval_L, prior):
    mu_C = compute_Mean_CovMatr_perClass(train_D, train_L, diagonal=True)
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)


