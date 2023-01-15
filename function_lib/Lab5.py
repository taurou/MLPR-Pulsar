import Lab4 as LAB4
import Lab3_LDA as LDA
import numpy as np
import matplotlib.pyplot as plt
import lib
import scipy.special
import math


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
        (mu, C) = LAB4.compute_Mean_CovMatr(D[:, L == i])
        if (diagonal):
            C = np.diag(np.diag(C))
        mean_cov_class[i] = (mu, C)
    return mean_cov_class

#takes as input the dictionary label : (mu, C) and the data.

def compute_logDensity_AllSamples(D, mu_C_tuple):
    matrix = []
    (mu, c) = mu_C_tuple
    matrix.append(LAB4.logpdf_GAU_ND(D, mu, c))
    return np.array(matrix)

#takes prior (a column vector with the prior probabilities for each class)

def compute_LOGjoint_density(D, mu_C, prior):
    logDensity = compute_logDensity_AllSamples(D, mu_C)
    return np.log(prior)+logDensity
#same as the previous function: computes the joint density between the sample and the mean+covmatr

def compute_joint_density(D, mu_C, prior):
    logDensity = compute_logDensity_AllSamples(D, mu_C)
    density = np.exp(logDensity)
    return prior*density

def MVG_basePredictor(eval_D, eval_L, mu_C, prior, usingLog = False): #TODO correct eval_L ! It's never used!!
    if(usingLog):
        logSJoint = np.zeros((len(mu_C), eval_D.shape[1]))
        for label, mean_covmatr in mu_C.items():
            logSJoint[label, :] = compute_LOGjoint_density(
                eval_D, mean_covmatr, prior.ravel()[label])
        logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        CPosterior = np.exp(logSPost)
    else:        
        SJoint = np.zeros((len(mu_C), eval_D.shape[1]))
        for label, mean_covmatr in mu_C.items():
            SJoint[label, :] = compute_joint_density(
                eval_D, mean_covmatr, prior.ravel()[label])
        SMarginal = lib.vrow(SJoint.sum(0))
        CPosterior = SJoint / SMarginal

    # Computes the LABEL with max Class-Posterior-Probability
    SPost = np.argmax(CPosterior, axis=0)
    """ BooleanPredictionResults = SPost == eval_L
    Accuracy = sum(BooleanPredictionResults) / len(eval_L)
    Error = 1 - Accuracy
    print("Accuracy: ", Accuracy, "Error: ", Error) """
    return SPost #return the predicted labels 

""" 

def MVG_Predictor(eval_D, eval_L, mu_C, prior):
    SJoint = np.zeros((len(mu_C), eval_D.shape[1]))
    for label, mean_covmatr in mu_C.items():
        SJoint[label, :] = compute_joint_density(
            eval_D, mean_covmatr, prior.ravel()[label])
    SMarginal = lib.vrow(SJoint.sum(0))
    CPosterior = SJoint / SMarginal
    # Computes the LABEL with max Class-Posterior-Probability
    SPost = np.argmax(CPosterior, axis=0)
    BooleanPredictionResults = SPost == eval_L
    Accuracy = sum(BooleanPredictionResults) / len(eval_L)
    Error = 1 - Accuracy
    print("Accuracy: ", Accuracy, "Error: ", Error)
    return SPost #return the predicted labels 



#TODO controllare log perché c'è un imprecisione numerica nell'ordine di e^-15

def MVG_logPredictor(eval_D, eval_L, mu_C, prior):
    logSJoint = np.zeros((len(mu_C), eval_D.shape[1]))
    for label, mean_covmatr in mu_C.items():
        logSJoint[label, :] = compute_LOGjoint_density(
            eval_D, mean_covmatr, prior.ravel()[label])
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    CPosterior = np.exp(logSPost)
    # Computes the LABEL with max Class-Posterior-Probability
    SPost = np.argmax(CPosterior, axis=0)
    BooleanPredictionResults = SPost == eval_L
    Accuracy = sum(BooleanPredictionResults) / len(eval_L)
    Error = 1 - Accuracy
    print("Accuracy: ", Accuracy, "Error: ", Error)
    return SPost #return the predicted labels 
 """
def TiedCovariance_MVG(train_D, train_L, eval_D, eval_L, prior):
    _, Sw = LDA.computeSbSw(train_D, train_L) #computing the empirical within covariance matrix
    class_means = lib.class_mean_vector(train_D, train_L)
    mu_C = { i:(lib.vcol(class_means[:,i:i+1]),Sw) for i in set(list(train_L))} #formatting the right input for the base function
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)
    
def TiedCovariance_logMVG(train_D, train_L, eval_D, eval_L, prior):
    _, Sw = LDA.computeSbSw(train_D, train_L) #computing the empirical within covariance matrix
    class_means = lib.class_mean_vector(train_D, train_L)
    mu_C = { i:(lib.vcol(class_means[:,i:i+1]),Sw) for i in set(list(train_L))} #formatting the right input for the base function
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior, usingLog=True)

def TiedCovariance_NB(train_D, train_L, eval_D, eval_L, prior):
    _, Sw = LDA.computeSbSw(train_D, train_L) #computing the empirical within covariance matrix
    Sw = np.diag(np.diag(Sw))
    class_means = lib.class_mean_vector(train_D, train_L)
    mu_C = { i:(lib.vcol(class_means[:,i:i+1]),Sw) for i in set(list(train_L))} #formatting the right input for the base function
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)

def TiedCovariance_logNB(train_D, train_L, eval_D, eval_L, prior):
    _, Sw = LDA.computeSbSw(train_D, train_L) #computing the empirical within covariance matrix
    Sw = np.diag(np.diag(Sw))
    class_means = lib.class_mean_vector(train_D, train_L)
    mu_C = { i:(lib.vcol(class_means[:,i:i+1]),Sw) for i in set(list(train_L))} #formatting the right input for the base function
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior, usingLog=True)

def MVG(train_D, train_L, eval_D, eval_L, prior):
    mu_C = compute_Mean_CovMatr_perClass(train_D, train_L)
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)    

def logMVG(train_D, train_L, eval_D, eval_L, prior):
    mu_C = compute_Mean_CovMatr_perClass(train_D, train_L)
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior, usingLog=True)

def NaiveBayes(train_D, train_L, eval_D, eval_L, prior):
    mu_C = compute_Mean_CovMatr_perClass(train_D, train_L, diagonal=True)
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior)    

def logNaiveBayes(train_D, train_L, eval_D, eval_L, prior):
    mu_C = compute_Mean_CovMatr_perClass(train_D, train_L, diagonal=True)
    return MVG_basePredictor(eval_D, eval_L, mu_C, prior, usingLog=True)


if __name__ == '__main__':
    D, L = lib.load_iris()
    (train_D, train_L), (eval_D, eval_L) = lib.split_db_2to1(D, L)

    prior = lib.vcol(np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]))
    LeaveOneOutCrossValidation(D, L, prior, MVG)
    LeaveOneOutCrossValidation(D, L, prior, NaiveBayes)
    LeaveOneOutCrossValidation(D, L, prior, TiedCovariance_MVG)
    LeaveOneOutCrossValidation(D, L, prior, TiedCovariance_NB)
    MVG(train_D, train_L, eval_D, eval_L, prior)
    logMVG(train_D, train_L, eval_D, eval_L, prior)
    NaiveBayes(train_D, train_L, eval_D, eval_L, prior)
    logNaiveBayes(train_D, train_L, eval_D, eval_L, prior)
    TiedCovariance_MVG(train_D, train_L, eval_D, eval_L, prior)
    TiedCovariance_logMVG(train_D, train_L, eval_D, eval_L, prior)
    TiedCovariance_NB(train_D, train_L, eval_D, eval_L, prior)
    TiedCovariance_logNB(train_D, train_L, eval_D, eval_L, prior)

#TODO refactor di MVG predictor creando un wrapper
    #test = np.load('5_Generative_Models/Solution/logPosterior_MVG.npy')
    #print(test - logSPost , "\n\n\n" )

    # SJoint = compute_class_posterior_probability(train_D, train_L, MU, C, prior)
    #SJoint_test = np.load('5_Generative_Models/Solution/Posterior_MVG.npy')
    # print("\n\n\n\n\n")
    #print(SJoint_test)
