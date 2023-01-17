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
def computeBinaryNormalizedDCF(eval_L, llratio, pi_1, C_fn, C_fp, threshold = None):
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

def plotROC_curves(eval_L, llratio):
    #compute ROC curves
    t = np.array(llratio)
    #append -inf and +inf
    t.sort()
    t = np.concatenate([np.array([-np.inf]), t , np.array([np.inf])])
    FNR_t = np.zeros(t.size)
    FPR_t = np.zeros(t.size)
    for idx, th in enumerate(t):
        (FNR_t[idx], FPR_t[idx]) = computeFNR_FPR(eval_L, llratio, threshold = th )
    TNR_t = 1 - FPR_t
    TPR_t = 1 - FNR_t
    plt.figure()
    plt.plot(FPR_t, TPR_t)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.figure()
    plt.plot(FNR_t, TNR_t)
    plt.xlabel("FNR")
    plt.ylabel("TNR")

    plt.show()

def BayesErrorPlots(eval_L, llratio, range_start = -3, range_end = 3, num_points = 21):
    effPriorLogOdds = np.linspace(range_start, range_end, num_points)
    effPrior = 1 / ( 1 + np.exp(-effPriorLogOdds) )

    DCF = np.zeros(num_points)
    minDCF = np.zeros(num_points)

    for idx, pi in enumerate(effPrior):
        DCF[idx] = computeBinaryNormalizedDCF(eval_L, llratio, pi, 1, 1)
        minDCF[idx] = compute_min_Normalized_DCF(eval_L, llratio, pi, 1, 1)
    
    plt.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([range_start, range_end])
    plt.xlabel("prior log-odds")
    plt.ylabel("min DCF")
    plt.show()


if __name__ == "__main__":
    '''D, L = lib.load_iris()
    (train_D, train_L), (eval_D, eval_L) = lib.split_db_2to1(D,L)
    prior = lib.vcol(np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]))
    IRISMVGprediction = MVG.logMVG(train_D, train_L, eval_D, eval_L, prior)
    IRISTiedMVGprediction = MVG.TiedCovariance_logMVG(train_D, train_L, eval_D, eval_L, prior)
    IRISconfMatrMVG = generateConfusionMatrix(eval_L, IRISMVGprediction)
    IRISconfMatrTiedMVG = generateConfusionMatrix(eval_L, IRISTiedMVGprediction)
    print("Iris MVG:\n", IRISconfMatrMVG)
    print("Iris Tied MVG:\n", IRISconfMatrTiedMVG )
    
    #Import DATA for commedia
    commedia_logLikelihoods = np.load('./8_Bayes_Decisions_Model_Evaluation/Data/commedia_ll.npy')
    commedia_Labels = np.load('./8_Bayes_Decisions_Model_Evaluation/Data/commedia_labels.npy')
    commedia_Logjoint = commedia_logLikelihoods + np.log(1.0/3.0)
    commedia_llMarginal = scipy.special.logsumexp(commedia_Logjoint, axis = 0)
    commedia_posterior = np.exp(commedia_Logjoint - commedia_llMarginal)
    commedia_prediction = np.argmax(commedia_posterior, axis=0)
    commediaConfMatr = generateConfusionMatrix(commedia_Labels, commedia_prediction)
    print("Commedia:\n", commediaConfMatr)




#BINARY TASK: OPTIMAIL DECISIONS
    llratio = np.load('./8_Bayes_Decisions_Model_Evaluation/Data/commedia_llr_infpar.npy') #llr predictions
    eval_L = np.load('./8_Bayes_Decisions_Model_Evaluation/Data/commedia_labels_infpar.npy') #evaluation labels
    
    #prior = lib.vcol(np.array([1 - prior_1, prior_1]))
    #commedia_Logjoint = commedia_llr_infpar + np.log(prior)
    #commedia_llMarginal = scipy.special.logsumexp(commedia_Logjoint, axis = 0)
    #commedia_posterior = np.exp(commedia_Logjoint - commedia_llMarginal)
    BayesConfMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1 = 0.5, C_fn = 1 , C_fp = 1)
    print("BayesConfMatr_pi0.5_C[1,1]: \n" , BayesConfMatr)
    DCFu = BinaryBayesRiskDCFu(BayesConfMatr, pi_1 = 0.5, C_fn = 1 , C_fp = 1)
    DCF = NormalizedBinaryBayesRiskDCF(BayesConfMatr, pi_1 = 0.5, C_fn = 1 , C_fp = 1)
    print("DCFu: ", DCFu, "DCF: ", DCF)
    BayesConfMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1 = 0.5, C_fn = 10 , C_fp = 1)
    print("BayesConfMatr_pi0.5_C[10,1]: \n" , BayesConfMatr)
    DCFu = BinaryBayesRiskDCFu(BayesConfMatr, pi_1 = 0.5, C_fn = 10 , C_fp = 1)
    DCF = NormalizedBinaryBayesRiskDCF(BayesConfMatr, pi_1 = 0.5, C_fn = 10 , C_fp = 1)
    print("DCFu: ", DCFu, "DCF: ", DCF)
    BayesConfMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1 = 0.8, C_fn = 1 , C_fp = 1)
    print("BayesConfMatr_pi0.8_C[1,1]: \n" , BayesConfMatr)
    DCFu = BinaryBayesRiskDCFu(BayesConfMatr, pi_1 = 0.8, C_fn = 1 , C_fp = 1)
    DCF = NormalizedBinaryBayesRiskDCF(BayesConfMatr, pi_1 = 0.8, C_fn = 1 , C_fp = 1)
    print("DCFu: ", DCFu, "DCF: ", DCF)
    BayesConfMatr = BayesBinaryOptimalDecisionConfMatr(eval_L, llratio, pi_1 = 0.8, C_fn = 1 , C_fp = 10)
    print("BayesConfMatr_pi0.8_C[1,10]: \n" , BayesConfMatr)
    #same but using the wrapper functions
    DCFu = computeBinaryDCFu(eval_L, llratio, pi_1 = 0.8, C_fn = 1 , C_fp = 10)
    DCF = computeBinaryNormalizedDCF(eval_L, llratio, pi_1 = 0.8, C_fn = 1 , C_fp = 10)
    print("DCFu: ", DCFu, "DCF: ", DCF)

    #minimum detection cost 

    llratio = np.load('./8_Bayes_Decisions_Model_Evaluation/Data/commedia_llr_infpar.npy')
    eval_L = np.load('./8_Bayes_Decisions_Model_Evaluation/Data/commedia_labels_infpar.npy') #evaluation labels
    scores = llratio #the scores correspont to the ll ratios returned by the classifier

    print(compute_min_Normalized_DCF(eval_L, scores, 0.5, 1, 1))
    print(compute_min_Normalized_DCF(eval_L, scores, 0.8, 1, 1))
    print(compute_min_Normalized_DCF(eval_L, scores, 0.5, 10, 1))
    print(compute_min_Normalized_DCF(eval_L, scores, 0.8, 1, 10))

    #plot ROC curves for TNR/FNR and TPR/FNR
    plotROC_curves(eval_L, llratio)
    #plot bayes errors
    BayesErrorPlots(eval_L, llratio)
        
'''