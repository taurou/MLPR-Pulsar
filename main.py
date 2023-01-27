import numpy as np
import function_lib.lib as lib
import function_lib.SVM as SVM
import function_lib.plots as plots
from scipy.stats import norm #TODO remove if I move gaussianize func
import math
import function_lib.PCA as PCA
import function_lib.MVG as MVG
import function_lib.LR as LR
import function_lib.GMM as GMM
import function_lib.model_eval as model_eval
import matplotlib.pyplot as plt

######## DATA LOADING AND PREPARATION ########
def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = lib.vcol(np.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("Error while loading file: %s"  % fname)

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)


def gaussianize(DTR, DTE = None):
    #computing rank of DTR
    N = DTR.shape[1] #number of DTR samples
    D = DTR.shape[0] #number of features
    rank = np.zeros(DTR.shape)
    for i in range(D):
        for j in range(N):
            rank[i][j] += (DTR[i] < DTR[i][j] ).sum()
    rank += 1
    rank /= (N+2)
    if(DTE is not None):
        evalRank = np.zeros(DTE.shape)
        for i in range(D):
            for j in range(DTE.shape[1]): #the rank is computed by comparision with the training set DTR
                evalRank[i][j] += (DTR[i] < DTE[i][j] ).sum()
        evalRank += 1
        evalRank /= (N+2)

        return norm.ppf(rank), norm.ppf(evalRank)
    return norm.ppf(rank)
    
def z_normalize(DTR, DTE = None):
    mu = lib.vcol(DTR.mean(axis = 1))
    std_dev = lib.vcol(DTR.std(axis = 1))
    DTRz = (DTR - mu) / std_dev
    if(DTE is not None):
        DTEz = (DTE - mu) / std_dev
        return DTRz, DTEz
    return DTRz

def KfoldsGenerator(D, L, k, seed=0): #passing the classifier function
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) #randomizes the order of the data

    #Preparing single-fold dataset (T-0.66% E-0.33%)
    train_size = int(math.ceil((D.shape[1]*0.66)))
    idxSTrain = idx[0:train_size]
    idxSTest = idx[train_size:]
    DTR, DTE = z_normalize(D[:,idxSTrain], D[:,idxSTest]) 
    LTR = L[idxSTrain]
    LTE = L[idxSTest]

    singleNOPCA = ((DTR, LTR),(DTE, LTE))

    DTR_PCA, DTE_PCA = PCA.compute_PCA(5,DTR,DTE)
    singlePCA5 = ((DTR_PCA,LTR),(DTE_PCA, LTE))
    DTR_PCA, DTE_PCA = PCA.compute_PCA(6,DTR,DTE) 
    singlePCA6 = ((DTR_PCA,LTR),(DTE_PCA, LTE))
    DTR_PCA, DTE_PCA = PCA.compute_PCA(7,DTR,DTE)
    singlePCA7 = ((DTR_PCA,LTR),(DTE_PCA, LTE)) 
    

    #Preparing K-Folds dataset
    group_size = int(math.ceil((D.shape[1]/k)))

    #preparing the k-folds
    NOPCA = [] #Folds are stored in the [(DTR1,LTR1,DTE1,LTE1),...] format
    PCA5 = []
    PCA6 = []
    PCA7 = []

    for i in range(k):
        idxTest = idx[i*group_size:(i*group_size + group_size)]
        idxTrain = [i for i in idx if i not in set(idxTest)]
        DTR, DTE = z_normalize( D[:,idxTrain], D[:,idxTest])
        LTR = L[idxTrain]
        LTE = L[idxTest]
        NOPCA.append(((DTR, LTR), (DTE, LTE)))
        DTR_PCA, DTE_PCA = PCA.compute_PCA(5,DTR,DTE)
        PCA5.append(((DTR_PCA, LTR), (DTE_PCA, LTE))) 
        DTR_PCA, DTE_PCA = PCA.compute_PCA(6,DTR,DTE)
        PCA6.append(((DTR_PCA, LTR), (DTE_PCA, LTE))) 
        DTR_PCA, DTE_PCA = PCA.compute_PCA(7,DTR,DTE)
        PCA7.append(((DTR_PCA, LTR), (DTE_PCA, LTE))) 
    
    return (singleNOPCA, singlePCA5, singlePCA6, singlePCA7) , (NOPCA, PCA5, PCA6, PCA7)

######## MVG ########

def MVGwrapper(data, prior, mode = "k-fold", k = 0):

    if(mode != "k-fold" and mode != "single-fold"):
        print("Error, enter mode")
        return

    MVGmodels = {
        0 : (MVG.logMVG, "fullMVG"),
        1 : (MVG.logNaiveBayes, "diagMVG"),
        2 : (MVG.TiedCovariance_logNB, "TiedDiagMVG"),
        3 : (MVG.TiedCovariance_logMVG, "TiedMVG")
    }

    
    for (model, modelName) in MVGmodels.values():
        if(mode == "single-fold"):
            print("*******Single Fold %s*******" % (modelName))
            singleFoldMVG_minDCF(data, prior, model)
        elif(mode == "k-fold"):
            print("*******%d-Fold %s*******" % (k,modelName))
            kfoldMVG_minDCF(data, prior, model)


def singleFoldMVG_minDCF(single_data, prior_t, MVGmodel):
    labels = {
        0 : "Z-norm Full",
        1 : "Z-norm PCA5",
        2 : "Z-norm PCA6",
        3 : "Z-norm PCA7"
    }

    for i, data in enumerate(single_data):
        (DTR, LTR),(DTE, LTE) = data
        for prior in prior_t:
            pred_L, llr = MVGmodel(DTR, LTR, DTE, LTE, lib.vcol(np.array([1-prior_t, prior_t])))
            minDCF = model_eval.compute_min_Normalized_DCF(LTE, llr, prior, C_fn = 1 , C_fp =  1)
            print("[Single Fold %s] prior[1] = %.1f minDCF = %.3f," % (labels[i], prior, minDCF) )

def kfoldMVG_minDCF(kfold_data, prior_t, MVGmodel):
    labels = {
        0 : "Z-norm Full",
        1 : "Z-norm PCA5",
        2 : "Z-norm PCA6",
        3 : "Z-norm PCA7"
    }
    
    for i, data in enumerate(kfold_data):
        #data is the FULL, PCA5, PCA6... 
        for prior in prior_t:
            kscores_array, kLTE_array = kfoldMVG(data, prior_t, MVGmodel)
            minDCF = model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1)
            print("[K-Fold %s] prior-1 = %.1f minDCF = %.3f," % (labels[i], prior, minDCF) )


def kfoldMVG(kfold_data, prior_t, MVGmodel ):
    K_scores = []
    K_LTE = []
    for fold in kfold_data:
        (DTRk, LTRk),(DTEk, LTEk) = fold
        pred_L, llr = MVGmodel(DTRk, LTRk, DTEk, LTEk, lib.vcol(np.array([1-prior_t, prior_t])))
        K_scores.append(llr)
        K_LTE.append(LTEk)
    kscores_array = np.concatenate(K_scores).ravel() #computing all the scores for all the folds and then putting them in one single array. Same for the labels.
    kLTE_array = np.concatenate(K_LTE).ravel()
    return kscores_array, kLTE_array    


def MVG_actDCF_calibration(kfold_data, prior_t, MVGmodel, calibration = False, showPlot = False): 

    models = {
       "fullMVG" : MVG.logMVG, 
       "diagMVG" : MVG.logNaiveBayes, 
       "tiedDiagMVG" : MVG.TiedCovariance_logNB,
       "tiedMVG" : MVG.TiedCovariance_logMVG
    }

    kscores_array, kLTE_array = kfoldMVG(kfold_data, prior_t, models[MVGmodel])
    if(calibration == True):
        print("Calibrated scores")
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5).ravel()
    else:
        print("Uncalibrated scores")
        calibratedScores = kscores_array
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    if showPlot:
        model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[%s] - prior: %.1f minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (MVGmodel, prior, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[%s] - prior: %.1f minDCF: %.3f actDCF: %.3f" % (MVGmodel, prior, minDCF[idx], actDCF[idx]))

    return minDCF, actDCF



def MVG_actDCF_calibration_evaluation(kfold_data, DTR, LTR, DTE, LTE, prior_t, MVGmodel, calibration = False, showPlot = False): 

    models = {
       "fullMVG" : MVG.logMVG, 
       "diagMVG" : MVG.logNaiveBayes, 
       "tiedDiagMVG" : MVG.TiedCovariance_logNB,
       "tiedMVG" : MVG.TiedCovariance_logMVG
    }

    pred_LTE, scores_LTE = models[MVGmodel](DTR, LTR, DTE, LTE, lib.vcol(np.array([1-prior_t, prior_t])))

    
    if(calibration == True):
        print("Calibrated evaluation scores")
        kscores_array, kLTE_array = kfoldMVG(kfold_data, prior_t, models[MVGmodel])
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5, evaluation_scores=scores_LTE).ravel()
    else:
        print("Uncalibrated evaluation scores")
        calibratedScores = scores_LTE
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    if showPlot:
        model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(LTE, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[%s] - prior: %.1f minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (MVGmodel, prior, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[%s] - prior: %.1f minDCF: %.3f actDCF: %.3f" % (MVGmodel, prior, minDCF[idx], actDCF[idx]))

    return minDCF, actDCF


######## LINEAR REGRESSION ########

def kfoldLR(kfold_data, pi_t, prior_t, l ):
    K_scores = []
    K_LTE = []
    for fold in kfold_data:
        (DTRk, LTRk),(DTEk, LTEk) = fold

        llr, pred_L = LR.computeLR(DTRk, LTRk, DTEk , l, pi_t) 
        K_scores.append(llr)
        K_LTE.append(LTEk)
    kscores_array = np.concatenate(K_scores).ravel() #computing all the scores for all the folds and then putting them in one single array. Same for the labels.
    kLTE_array = np.concatenate(K_LTE).ravel()
    return kscores_array, kLTE_array    


def kfoldLR_minDCF(kfold_data, pi_t, prior_t, l ): #kfold_data takes only one foldtype. e.g. only PCA5. Not the whole array. 
    kscores_array, kLTE_array = kfoldLR(kfold_data, pi_t, prior_t, l )
    minDCF = []
    for prior in prior_t:
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
    return minDCF

def plotLR_minDCF(kfold_data, prior_t, pi_t, filename = "LogReg", save=False, mode = "k-fold"):
    l = np.logspace(-5, 2, num = 20) #limiting the number of points by 50.
    minDCF_array = [ [] for i in range(len(prior_t)) ]
    for lamb in l:
        minDCF = kfoldLR_minDCF(kfold_data, pi_t, prior_t, lamb)
        [ minDCF_array[i].append(minDCF[i]) for i in range(len(minDCF)) ]
     
    print("plotting %d-fold" %(kfold_data[0][0][0].shape[0]))
    plots.plotminDCF(l,minDCF_array, prior_t, "lambda", filename, save)
    
def computeLR_minDCF(kfold_data, prior_t, l, mode = "k-fold"):
    for pi_t in prior_t:
        minDCF = kfoldLR_minDCF(kfold_data, pi_t, prior_t, l)
        for i, prior in enumerate(prior_t):
            print("[linear-LR] - lambda: %f p_T: %.1f prior: %.1f minDCF: %.3f" % (l, pi_t, prior, minDCF[i]))


def LR_actDCF_calibration(kfold_data, pi_t, prior_t, l, calibration = False): #if pi_t = 0 -> no SVM balancing #TODO handle k
    kscores_array, kLTE_array = kfoldLR(kfold_data, pi_t, prior_t, l )
    if(calibration == True):
        print("Calibrated scores")
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5).ravel()
    else:
        print("Uncalibrated scores")
        calibratedScores = kscores_array
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[linear-LR] - lambda: %f p_T: %.1f prior: %.1f minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (l, pi_t, prior, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[linear-LR] - lambda: %f p_T: %.1f prior: %.1f minDCF: %.3f actDCF: %.3f" % (l, pi_t, prior, minDCF[idx], actDCF[idx]))
    return minDCF, actDCF
        
def LR_actDCF_calibration_evaluation(kfold_data, DTR, LTR, DTE, LTE, pi_t, prior_t, l, calibration = False, showPlot = False): 


    scores_LTE, pred_LTE= LR.computeLR(DTR, LTR, DTE, l, pi_t) 

    if(calibration == True):
        print("Calibrated evaluation scores")
        kscores_array, kLTE_array = kfoldLR(kfold_data, pi_t, prior_t, l )
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5, evaluation_scores=scores_LTE).ravel()
    else:
        print("Uncalibrated evaluation scores")
        calibratedScores = scores_LTE
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    if showPlot:
        model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(LTE, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[linear-LR] - lambda: %f p_T: %.1f prior: %.1f minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (l, pi_t, prior, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[linear-LR] - lambda: %f p_T: %.1f prior: %.1f minDCF: %.3f actDCF: %.3f" % (l, pi_t, prior, minDCF[idx], actDCF[idx]))
    return minDCF, actDCF

    
######## SVM ########

def SVM_kfold(kfold_data, C, pi_t, prior_t, kernelType, gamma = 0.0, c = 0.0, selected_prior = None):
    K_scores = []
    K_LTE = []
    for fold in kfold_data:
        (DTRk, LTRk),(DTEk, LTEk) = fold
        if(kernelType == "linear"):
            SVMObj = SVM.SVMClass(DTRk, LTRk, "linear", k = 1.0, pi_t = pi_t) #use default value k=1
        elif(kernelType == "quadratic"):
            SVMObj = SVM.SVMClass(DTRk, LTRk, "poly", k = 1.0, d = 2.0, c = c, pi_t = pi_t)
        elif(kernelType == "RBF"):
            SVMObj = SVM.SVMClass(DTRk, LTRk, "RBF", k = 1.0, gamma = gamma, pi_t = pi_t) 

        alpha = SVMObj.trainSVM(C)
        scores , pred_L = SVMObj.compute_labels(alpha, DTEk)
        K_scores.append(scores)
        K_LTE.append(LTEk)
    
    kscores_array = np.concatenate(K_scores).ravel() #computing all the scores for all the folds and then putting them in one single array. Same for the labels.
    kLTE_array = np.concatenate(K_LTE).ravel()

    return kscores_array, kLTE_array

def SVM_actDCF_calibration(kfold_data, C, pi_t, prior_t, kernelType, gamma = 0.0, c = 0.0, selected_prior = None, calibration = False): #if pi_t = 0 -> no SVM balancing #TODO handle k
    kscores_array, kLTE_array = SVM_kfold(kfold_data, C, pi_t, prior_t, kernelType, gamma, c, selected_prior)
    if(calibration == True):
        print("Calibrated scores")
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5).ravel()
    else:
        print("Uncalibrated scores")
        calibratedScores = kscores_array
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[%s-SVM] - C: %f prior: %.1f, p_T: %.1f minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (kernelType, C, prior, pi_t, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[%s-SVM] - C: %f prior: %.1f, p_T: %.1f minDCF: %.3f actDCF: %.3f" % (kernelType, C, prior, pi_t, minDCF[idx], actDCF[idx]))
    return minDCF, actDCF

def SVM_actDCF_calibration_evaluation(kfold_data, DTR, LTR, DTE, LTE, C, pi_t, prior_t, kernelType, gamma = 0.0, c = 0.0, selected_prior = None, calibration = False, showPlot = False): #if pi_t = 0 -> no SVM balancing #TODO handle k

    if(kernelType == "linear"):
        SVMObj = SVM.SVMClass(DTR, LTR, "linear", k = 1.0, pi_t = pi_t) #use default value k=1
    elif(kernelType == "quadratic"):
        SVMObj = SVM.SVMClass(DTR, LTR, "poly", k = 1.0, d = 2.0, c = c, pi_t = pi_t)
    elif(kernelType == "RBF"):
        SVMObj = SVM.SVMClass(DTR, LTR, "RBF", k = 1.0, gamma = gamma, pi_t = pi_t) 

    alpha = SVMObj.trainSVM(C)
    scores_LTE, pred_LTE = SVMObj.compute_labels(alpha, DTE)

    if(calibration == True):
        print("Calibrated evaluation scores")
        kscores_array, kLTE_array = SVM_kfold(kfold_data, C, pi_t, prior_t, kernelType, gamma, c, selected_prior)
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5, evaluation_scores=scores_LTE).ravel()
    else:
        print("Uncalibrated evaluation scores")
        calibratedScores = scores_LTE
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    if showPlot:
        model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(LTE, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[%s-SVM] - C: %f prior: %.1f, p_T: %.1f minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (kernelType, C, prior, pi_t, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[%s-SVM] - C: %f prior: %.1f, p_T: %.1f minDCF: %.3f actDCF: %.3f" % (kernelType, C, prior, pi_t, minDCF[idx], actDCF[idx]))
    return minDCF, actDCF


def SVM_minDCF(kfold_data, C, pi_t, prior_t, kernelType, gamma = 0.0, c = 0.0, selected_prior = None): #if pi_t = 0 -> no SVM balancing #TODO handle k

    kscores_array, kLTE_array = SVM_kfold(kfold_data, C, pi_t, prior_t, kernelType, gamma, c, selected_prior)
    if(selected_prior is not None):
        return model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, selected_prior, C_fn = 1 , C_fp =  1)
    minDCF = []
    for prior in prior_t:
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
    return minDCF

def computeSVM_minDCF(kfold_data, C, prior_t, kernelType, gamma = 0.0, c = 0.0):
    minDCFunbalanced = SVM_minDCF(kfold_data, C, 0, prior_t, kernelType, gamma, c)
    if kernelType == "quadratic":
        SVM_string = "c : %f" %(c)
    elif kernelType == "RBF":
        SVM_string = "gamma : %f" %(gamma)
    else:
        SVM_string = ""

    for idx, prior in enumerate(prior_t):
        print("[%s-unbalancedSVM] - C: %f prior: %.1f, minDCF: %.3f %s" % (kernelType, C, prior, minDCFunbalanced[idx], SVM_string))
    for pi_t in prior_t:
        minDCF = SVM_minDCF(kfold_data, C, pi_t, prior_t, kernelType, gamma, c)
        for idx, prior in enumerate(prior_t):
            print("[%s-balancedSVM] - C: %f prior: %.1f, p_T: %.1f minDCF: %.3f %s" % (kernelType, C, prior, pi_t, minDCF[idx], SVM_string))


def plotLinSVM_minDCF(kfold_data, prior_t, pi_t = 0, filename="SVM", save=False, mode = "k-fold"): #pi_t = 0 is unbalanced
    C = np.logspace(-5, 0, num = 20) #limiting the number of points by 50.
    minDCF_array = [ [] for i in range(len(prior_t)) ] #create an array with a sub-array for each prior computed.
    for Cx in C:
        minDCF = SVM_minDCF(kfold_data, Cx, pi_t, prior_t, "linear")
        [ minDCF_array[i].append(minDCF[i]) for i in range(len(minDCF)) ]
    print("plotting %d-fold" %(kfold_data[0][0][0].shape[0]))
    plots.plotminDCF(C,minDCF_array, prior_t, "C", filename, save)
    

######## Quad + RBF SVM ########


def plotRBFSVM_minDCF(kfold_data, prior_t, selected_prior, pi_t = 0, filename="SVM", save=False, mode = "k-fold"): #pi_t = 0 is unbalanced
    C = np.logspace(-4, 0, num = 20) #limiting the number of points by 50.
    gamma_range = np.array([ 0.001, 0.01, 0.1 ])
    gamma_minDCF = [ [] for i in range(len(gamma_range)) ]
    for gammaIdx, gamma in enumerate(gamma_range):
        for Cx in C: 
            minDCF = SVM_minDCF(kfold_data, Cx, pi_t, prior_t, "RBF", gamma = gamma, selected_prior= selected_prior)
            gamma_minDCF[gammaIdx].append(minDCF)
    plots.plotminDCF_kernelSVM(C, gamma_minDCF, gamma_range, selected_prior, "RBF", filename, save)



def plotQuadraticSVM_minDCF(kfold_data, prior_t, selected_prior, pi_t = 0, filename="SVM", save=False, mode = "k-fold"): #pi_t = 0 is unbalanced
    C = np.logspace(-4, 0, num = 20) #limiting the number of points by 50.
    c_range = np.array([1, 10, 30, 80, 100])
    c_minDCF = [ [] for i in range(len(c_range)) ]
    for c_idx, c in enumerate(c_range):
        for Cx in C: 
            minDCF = SVM_minDCF(kfold_data, Cx, pi_t, prior_t, "quadratic", c = c, selected_prior= selected_prior)
            c_minDCF[c_idx].append(minDCF)
    plots.plotminDCF_kernelSVM(C, c_minDCF, c_range, selected_prior, "quadratic", filename, save)

######## GMM ########
def GMM_kfold(kfold_data, prior_t, algorithm_iterations=0,  LBG_mode = "full"):
    K_scores = []
    K_LTE = []
    for fold in kfold_data:
        (DTRk, LTRk),(DTEk, LTEk) = fold

        scores, pred_L = GMM.GMMBinaryclassification(DTRk,LTRk,DTEk,algorithm_iterations, LBG_mode ) #TODO add other parameters

        K_scores.append(scores)
        K_LTE.append(LTEk)

    kscores_array = np.concatenate(K_scores).ravel() #computing all the scores for all the folds and then putting them in one single array. Same for the labels.
    kLTE_array = np.concatenate(K_LTE).ravel()
    return kscores_array, kLTE_array


def GMM_minDCF(kfold_data, prior_t, algorithm_iterations=0,  LBG_mode = "full"): #if pi_t = 0 -> no SVM balancing #TODO handle k
    kscores_array, kLTE_array = GMM_kfold(kfold_data, prior_t, algorithm_iterations,  LBG_mode)
    minDCF = []
    for prior in prior_t:
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
    return minDCF

def computeGMM_minDCF(kfold_data, prior_t, algorithm_iterations, LBG_mode):
    minDCF = GMM_minDCF(kfold_data, prior_t, algorithm_iterations,  LBG_mode )

    for idx, prior in enumerate(prior_t):
        print("[%s-GMM] - #iterations: %d prior: %.1f minDCF: %.3f" % (LBG_mode, algorithm_iterations, prior, minDCF[idx]))

def plotGMM_minDCF(kfold_data, prior_t, LBG_mode,  filename="GMM", save = False,  mode = "k-fold"): #pi_t = 0 is unbalanced
    C = np.arange(0, 6, 1) #range of analysis from 1 to 2**7 components of GMM
    minDCF_array = [ [] for i in range(len(prior_t)) ] #create an array with a sub-array for each prior computed.
    for c in C:
        minDCF = GMM_minDCF(kfold_data, prior_t, algorithm_iterations=c,  LBG_mode = LBG_mode )
        [ minDCF_array[i].append(minDCF[i]) for i in range(len(minDCF)) ]
    print("plotting %d-fold" %(kfold_data[0][0][0].shape[0]))
    C = 2**C #number of components
    plots.plotminDCF(C,minDCF_array, prior_t, "Components", filename, save, logScale=False)

def GMM_actDCF_calibration(kfold_data, prior_t, algorithm_iterations, LBG_mode, calibration = False): #if pi_t = 0 -> no SVM balancing #TODO handle k
    kscores_array, kLTE_array = GMM_kfold(kfold_data, prior_t, algorithm_iterations,  LBG_mode)

    if(calibration == True):
        print("Calibrated scores")
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5).ravel()
    else:
        print("Uncalibrated scores")
        calibratedScores = kscores_array
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(kLTE_array, kscores_array, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(LTE, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[%s-GMM] - #iterations: %d prior: %.1f, minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (LBG_mode, algorithm_iterations, prior, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[%s-GMM] - #iterations: %d prior: %.1f, minDCF: %.3f actDCF: %.3f" % (LBG_mode, algorithm_iterations, prior, minDCF[idx], actDCF[idx]))
    return minDCF, actDCF

def GMM_actDCF_calibration_evaluation(kfold_data, DTR, LTR, DTE, LTE, prior_t, algorithm_iterations, LBG_mode, calibration = False, showPlot = False): #if pi_t = 0 -> no SVM balancing #TODO handle k
    scores_LTE, pred_LTE = GMM.GMMBinaryclassification(DTR,LTR,DTE,algorithm_iterations, LBG_mode )

    if(calibration == True):
        print("Calibrated evaluation scores")
        kscores_array, kLTE_array = GMM_kfold(kfold_data, prior_t, algorithm_iterations,  LBG_mode)
        calibratedScores = LR.calibrateScores(kscores_array, kLTE_array, l = 1e-4, pi_t = 0.5, evaluation_scores=scores_LTE).ravel()
    else:
        print("Uncalibrated evaluation scores")
        calibratedScores = scores_LTE
    minDCF = []
    actDCF = []
    actDCF_calibrated = []
    if showPlot:
        model_eval.BayesErrorPlots(kLTE_array, calibratedScores)
    for idx, prior in enumerate(prior_t):
        minDCF.append(model_eval.compute_min_Normalized_DCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        actDCF.append(model_eval.computeBinaryNormalizedDCF(LTE, scores_LTE, prior, C_fn = 1 , C_fp =  1))
        if(calibration == True):
            actDCF_calibrated.append(model_eval.computeBinaryNormalizedDCF(LTE, calibratedScores, prior, C_fn = 1 , C_fp =  1))
            print("[%s-GMM] - #iterations: %d prior: %.1f, minDCF: %.3f actDCF: %.3f actDCF_calibrated: %.3f" % (LBG_mode, algorithm_iterations, prior, minDCF[idx], actDCF[idx], actDCF_calibrated[idx]))
        else:
            print("[%s-GMM] - #iterations: %d prior: %.1f, minDCF: %.3f actDCF: %.3f" % (LBG_mode, algorithm_iterations, prior, minDCF[idx], actDCF[idx]))
    return minDCF, actDCF




if __name__ == "__main__":
    DTR, LTR = load("dataset/Train.txt")
    #plots.plot_hist(DTR,LTR, "raw dataset", "plots/1_RawDataset", True)
    # DTR_gau = gaussianize(DTR)
    #plots.plot_hist(DTR_gau,LTR, "gaussianised dataset" , "plots/2_GaussianizedDataset", True)
    DTR_z = z_normalize(DTR) #TODO maybe move it to kfold function and perform it on the DTR of the fold.
    #TODO maybe also perform PCS 
    #plots.plot_hist(DTR_z, LTR, "Z-Norm", "plots/3_zNormDataset", True)

    #plots.heatmap(DTR_z, LTR,  "Heatmap, Z-norm dataset", "heatmap", True)

    #plots.plot_correlations(DTR_z, LTR)

    #Create the K-folds 
    k = 3 #num of folds
    singleFoldData, kFoldData = (singleNOPCA, singlePCA5, singlePCA6, singlePCA7) , (NOPCA, PCA5, PCA6, PCA7) = KfoldsGenerator(DTR,LTR, k)

    prior_t = np.array([0.5, 0.1, 0.9])
    



    #MVGwrapper(singleFoldData, prior_t, mode = "single-fold")
    #MVGwrapper(kFoldData, prior_t, mode = "k-fold", k = k)


    '''
    plotLR_minDCF(NOPCA, prior_t, 0.5, "LR-Full-πT=0.5", True )
    plotLR_minDCF(PCA5, prior_t, 0.5,  "LR-PCA5-πT=0.5", True)
    plotLR_minDCF(PCA6, prior_t, 0.5,  "LR-PCA6-πT=0.5", True)
    plotLR_minDCF(PCA7, prior_t, 0.5,  "LR-PCA7-πT=0.5", True)

    plotLR_minDCF(NOPCA, prior_t, 0.1, "LR-Full-πT=0.1", True )
    plotLR_minDCF(PCA5, prior_t, 0.1,  "LR-PCA5-πT=0.1", True)
    plotLR_minDCF(PCA6, prior_t, 0.1,  "LR-PCA6-πT=0.1", True)
    plotLR_minDCF(PCA7, prior_t, 0.1,  "LR-PCA7-πT=0.1", True)

    plotLR_minDCF(NOPCA, prior_t, 0.9, "LR-Full-πT=0.9", True )
    plotLR_minDCF(PCA5, prior_t, 0.9,  "LR-PCA5-πT=0.9", True)
    plotLR_minDCF(PCA6, prior_t, 0.9,  "LR-PCA6-πT=0.9", True)
    plotLR_minDCF(PCA7, prior_t, 0.9,  "LR-PCA7-πT=0.9", True)
    
    print("SVM-Full-unbalanced")
    plotLinSVM_minDCF(NOPCA, prior_t, 0.0, "SVM-Full-unbalanced", True)
    print("SVM-PCA5-unbalanced")
    plotLinSVM_minDCF(PCA5, prior_t, 0.0, "SVM-PCA5-unbalanced", True)
    print("SVM-PCA6-unbalanced")
    plotLinSVM_minDCF(PCA6, prior_t, 0.0, "SVM-PCA6-unbalanced", True)
    print("SVM-PCA7-unbalanced")
    plotLinSVM_minDCF(PCA7, prior_t, 0.0, "SVM-PCA7-unbalanced", True)

    
    print("SVM-Full-πT=0.5")
    plotLinSVM_minDCF(NOPCA, prior_t, 0.5, "SVM-Full-πT=0.5", True)
    print("SVM-PCA5-πT=0.5")
    plotLinSVM_minDCF(PCA5, prior_t, 0.5, "SVM-PCA5-πT=0.5", True)
    print("SVM-PCA6-πT=0.5")
    plotLinSVM_minDCF(PCA6, prior_t, 0.5, "SVM-PCA6-πT=0.5", True)
    print("SVM-PCA7-πT=0.5")
    plotLinSVM_minDCF(PCA7, prior_t, 0.5, "SVM-PCA7-πT=0.5", True)
    

    print("SVM-Full-πT=0.1")
    plotLinSVM_minDCF(NOPCA, prior_t, 0.1, "SVM-Full-πT=0.1", True)
    print("SVM-PCA5-πT=0.1")
    plotLinSVM_minDCF(PCA5, prior_t, 0.1, "SVM-PCA5-πT=0.1", True)
    print("SVM-PCA6-πT=0.1")
    plotLinSVM_minDCF(PCA6, prior_t, 0.1, "SVM-PCA6-πT=0.1", True)
    print("SVM-PCA7-πT=0.1")
    plotLinSVM_minDCF(PCA7, prior_t, 0.1, "SVM-PCA7-πT=0.1", True)

    print("SVM-Full-πT=0.9")
    plotLinSVM_minDCF(NOPCA, prior_t, 0.9, "SVM-Full-πT=0.9", True)
    print("SVM-PCA5-πT=0.9")
    plotLinSVM_minDCF(PCA5, prior_t, 0.9, "SVM-PCA5-πT=0.9", True)
    print("SVM-PCA6-πT=0.9")
    plotLinSVM_minDCF(PCA6, prior_t, 0.9, "SVM-PCA6-πT=0.9", True)
    print("SVM-PCA7-πT=0.9")
    plotLinSVM_minDCF(PCA7, prior_t, 0.9, "SVM-PCA7-πT=0.9", True)
    
    
    print("RBF-SVM-Full")
    plotRBFSVM_minDCF(NOPCA, prior_t, selected_prior=0.5, filename="RBF-SVM-Full", save = True )
    print("RBF-SVM-PCA7")
    plotRBFSVM_minDCF(PCA7, prior_t, selected_prior=0.5, filename="RBF-SVM-PCA7", save = True )
    
    print("RBF-SVM-PCA5")
    plotRBFSVM_minDCF(PCA5, prior_t, selected_prior=0.5, filename="RBF-SVM-PCA5", save = True )
    print("RBF-SVM-PCA6")
    plotRBFSVM_minDCF(PCA6, prior_t, selected_prior=0.5, filename="RBF-SVM-PCA6", save = True )
    
    
    print("QuadSVM-NOPCA")
    plotQuadraticSVM_minDCF(NOPCA, prior_t, selected_prior=0.5, filename="QuadSVM-Full", save = True )
    print("QuadSVM-PCA7")
    plotQuadraticSVM_minDCF(PCA7, prior_t, selected_prior=0.5, filename="QuadSVM-PCA7", save = True )
    
    print("QuadSVM-PCA5")
    plotQuadraticSVM_minDCF(PCA5, prior_t, selected_prior=0.5, filename="QuadSVM-PCA5", save = True )
    print("QuadSVM-PCA6")
    plotQuadraticSVM_minDCF(PCA6, prior_t, selected_prior=0.5, filename="QuadSVM-PCA6", save = True )
    
    
    print("GMM-noPCA-full")
    plotGMM_minDCF(NOPCA, prior_t, LBG_mode="full", filename="GMM-noPCA-full", save=True)
    print("GMM-noPCA-tied")
    plotGMM_minDCF(NOPCA, prior_t, LBG_mode="tied", filename="GMM-noPCA-tied", save=True)
    print("GMM-noPCA-diag")
    plotGMM_minDCF(NOPCA, prior_t, LBG_mode="diag", filename="GMM-noPCA-diag", save=True)

    print("GMM-PCA7-full")
    plotGMM_minDCF(PCA7, prior_t, LBG_mode="full", filename="GMM-PCA7-full", save=True)
    print("GMM-PCA7-tied")
    plotGMM_minDCF(PCA7, prior_t, LBG_mode="tied", filename="GMM-PCA7-tied", save=True)
    print("GMM-PCA7-diag")
    plotGMM_minDCF(PCA7, prior_t, LBG_mode="diag", filename="GMM-PCA7-diag", save=True)

    
    print("GMM-PCA6-full")
    plotGMM_minDCF(PCA6, prior_t, LBG_mode="full", filename="GMM-PCA6-full", save=True)
    print("GMM-PCA6-tied")
    plotGMM_minDCF(PCA6, prior_t, LBG_mode="tied", filename="GMM-PCA6-tied", save=True)
    print("GMM-PCA6-diag")
    plotGMM_minDCF(PCA6, prior_t, LBG_mode="diag", filename="GMM-PCA6-diag", save=True)

    ### Linear LR ####
   
    print("LinearLR NO PCA")
    computeLR_minDCF(NOPCA, prior_t, 1e-5)
    print("LinearLR PCA7")
    computeLR_minDCF(PCA7, prior_t, 1e-5)
    print("LinearLR NO PCA")
    computeLR_minDCF(NOPCA, prior_t, 1e-4)
    print("LinearLR PCA7")
    computeLR_minDCF(PCA7, prior_t, 1e-4)
    

    ### LINEAR SVM ###
    print("LinSVM noPCA")
    computeSVM_minDCF(NOPCA, 5*1e-1, prior_t, "linear")
    computeSVM_minDCF(PCA7, 5*1e-1, prior_t, "linear")
    '''

    ### QUAD SVM ###
    #computeSVM_minDCF(NOPCA, 1e-3, prior_t, "quadratic", c = 10.0 )
    #computeSVM_minDCF(PCA7, 1e-3, prior_t, "quadratic", c = 10.0 )


    ### RBF SVM ###
    #computeSVM_minDCF(NOPCA, 1e-1, prior_t, "RBF", gamma = 1e-1 )
    #computeSVM_minDCF(PCA7, 1e-1, prior_t, "RBF", gamma = 1e-1 )

    ### GMM NO PCA ###
    #print("NO PCA")
    #computeGMM_minDCF(NOPCA, prior_t, 4, "full")
    #computeGMM_minDCF(NOPCA, prior_t, 4, "diag")
    #computeGMM_minDCF(NOPCA, prior_t, 3, "tied")
    ### GMM PCA = 7 ###
    #print("PCA = 7")
    #computeGMM_minDCF(PCA7, prior_t, 4, "full")
    #computeGMM_minDCF(PCA7, prior_t, 2, "diag")
    #computeGMM_minDCF(PCA7, prior_t, 3, "tied")


    #### SCORE CALIBRATION ####pyt


    #SVM_actDCF_calibration(NOPCA, 1e-1, 0, prior_t, "RBF", gamma = 1e-1, calibration = False )
    #LR_actDCF_calibration(NOPCA, 0.5, prior_t, 1e-5, calibration = True)
    #MVG_actDCF_calibration(NOPCA, prior_t, "tiedMVG", calibration = True)



    #### EVALUATION ####


    DTE, LTE = load("dataset/Test.txt")
    DTRz, DTEz = z_normalize(DTR, DTE)
    DTRz_PCA7, DTEz_PCA7 = PCA.compute_PCA(7, DTRz, DTEz)

    '''
    print("NO PCA Evaluation")
    MVG_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, "fullMVG", calibration = True) 
    MVG_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, "diagMVG", calibration = True) 
    MVG_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, "tiedDiagMVG", calibration = True) 
    MVG_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, "tiedMVG", calibration = True) 
    

    for prior in prior_t:
        LR_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, 0, prior_t, 1e-5, calibration = True)

    print("unbalanced SVM")
    SVM_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, 1e-1, prior, prior_t, "RBF", gamma = 1e-1, calibration = True )
    print("balanced SVM")
    for prior in prior_t:
        SVM_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, 1e-1, prior, prior_t, "RBF", gamma = 1e-1, calibration = True )
    '''
    GMM_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, 4, "full", calibration = False)
    GMM_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, 4, "diag", calibration = False)
    GMM_actDCF_calibration_evaluation(NOPCA,DTRz, LTR, DTEz, LTE, prior_t, 3, "tied", calibration = False)


    '''
    print("PCA7 Evaluation")
    MVG_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, "fullMVG", calibration = True) 
    MVG_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, "diagMVG", calibration = True) 
    MVG_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, "tiedDiagMVG", calibration = True) 
    MVG_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, "tiedMVG", calibration = True) 
    

    for prior in prior_t:
        LR_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, 0, prior_t, 1e-5, calibration = True)

    print("unbalanced SVM")
    SVM_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, 1e-1, prior, prior_t, "RBF", gamma = 1e-1, calibration = True )
    print("balanced SVM")
    for prior in prior_t:
        SVM_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, 1e-1, prior, prior_t, "RBF", gamma = 1e-1, calibration = True )
    
    GMM_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, 4, "full", calibration = False)
    GMM_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, 2, "diag", calibration = False)
    GMM_actDCF_calibration_evaluation(PCA7,DTRz, LTR, DTEz, LTE, prior_t, 3, "tied", calibration = False)
    '''


    print("ciao")
