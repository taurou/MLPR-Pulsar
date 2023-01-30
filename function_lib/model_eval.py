import numpy as np
import scipy.special
import function_lib.lib as lib
import matplotlib.pyplot as plt

#TODO fix scores e llratio è la stessa cosa. llratio è lo score del classificatore.

def generateConfusionMatrix(eval_L, predicted_L):
    numberOfClasses = np.max(eval_L) + 1
    confusionMatr = np.zeros((numberOfClasses, numberOfClasses), dtype=int)
    for i in range(numberOfClasses):
        for j in range(numberOfClasses):
            #i is prediction row, j is class column
            confusionMatr[i,j] = (( predicted_L == i ) * ( eval_L == j )).sum() 
    return confusionMatr

def BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1, C_fn, C_fp, threshold = None): #takes eval_L as labels, llratio is the log likelihood ratio class1/class0, pi_1 is the prior of class 1
    if threshold is None:
        threshold = -np.log( ( pi_1*C_fn ) / ((1-pi_1)*C_fp) )
    predicted_L = np.int32(llratio > threshold) #transforms True / False to 1 / 0
    return generateConfusionMatrix(eval_L, predicted_L)

def BinaryBayesRiskDCFu(confMatr, pi_1, C_fn, C_fp):
    FNR = confMatr[0,1] / confMatr[ :, 1].sum()
    FPR = confMatr[1,0] / confMatr[ :, 0].sum()
    return pi_1 * C_fn * FNR + (1 - pi_1) * C_fp * FPR

def NormalizedBinaryBayesRiskDCF(confMatr, pi_1, C_fn, C_fp):
    return BinaryBayesRiskDCFu(confMatr, pi_1, C_fn, C_fp) / np.min([pi_1*C_fn, (1-pi_1)*C_fp])

#this is a wrapper to return FNR and FPR 
def computeFNR_FPR(eval_L, llratio, pi_1 = 0.5, C_fn = 1, C_fp = 1, threshold = None):
    confMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1, C_fn, C_fp, threshold)
    FNR = confMatr[0,1] / confMatr[ :, 1].sum()
    FPR = confMatr[1,0] / confMatr[ :, 0].sum()

    return (FNR, FPR)

#this is a wrapper for DCFu
def computeBinaryDCFu(eval_L, llratio, pi_1, C_fn, C_fp, threshold = None):
    confMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1, C_fn, C_fp, threshold)
    return BinaryBayesRiskDCFu(confMatr, pi_1, C_fn, C_fp)

#this is a wrapper for DCF
def computeBinaryNormalizedDCF(eval_L, llratio, pi_1, C_fn, C_fp, threshold = None): #AKA actDCF
    confMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1, C_fn, C_fp, threshold)
    return NormalizedBinaryBayesRiskDCF(confMatr, pi_1, C_fn, C_fp)
    
def compute_min_Normalized_DCF(eval_L, scores, pi_1, C_fn, C_fp): #the scores are the same llratio vector!!!
    t = np.array(scores)
    #append -inf and +inf
    t.sort()
    t = np.concatenate([np.array([-np.inf]), t , np.array([np.inf])])
    DCF_list = []
    for th in t:
        DCF_list.append(computeBinaryNormalizedDCF(eval_L, scores, pi_1, C_fn, C_fp, th))
    return np.min(DCF_list)

def plotROC_curves(eval_L, llratio, label):
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    colours = ['r','y','k','g','b']

    for i, score in enumerate(llratio):
        #compute ROC curves
        t = np.array(score)
        #append -inf and +inf
        t.sort()
        t = np.concatenate([np.array([-np.inf]), t , np.array([np.inf])])
        FNR_t = np.zeros(t.size)
        FPR_t = np.zeros(t.size)
        for idx, th in enumerate(t):
            (FNR_t[idx], FPR_t[idx]) = computeFNR_FPR(eval_L, score, threshold = th )
        TPR_t = 1 - FNR_t
        plt.plot(FPR_t, TPR_t, label=label[i], color=colours[i])
    plt.legend(label)

    plt.show()

def BayesErrorPlots(eval_L, llratio, range_start = -3, range_end = 3, num_points = 21, title = ""):
    effPriorLogOdds = np.linspace(range_start, range_end, num_points)
    effPrior = 1 / ( 1 + np.exp(-effPriorLogOdds) )

    DCF = np.zeros(num_points)
    minDCF = np.zeros(num_points)

    for idx, pi in enumerate(effPrior):
        DCF[idx] = computeBinaryNormalizedDCF(eval_L, llratio, pi, 1, 1)
        minDCF[idx] = compute_min_Normalized_DCF(eval_L, llratio, pi, 1, 1)
    plt.figure()
    plt.title(title)
    plt.plot(effPriorLogOdds, DCF, label='actDCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b', linestyle='dashed')
    plt.legend(["actDCF", "minDCF"])
    plt.ylim([0, 1.1])
    plt.xlim([range_start, range_end])
    plt.xlabel("prior log-odds")
    plt.ylabel("min DCF")
    plt.show()
