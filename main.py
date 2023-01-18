import numpy as np
import function_lib.lib as lib
import function_lib.Lab4 as LAB4
import function_lib.plots as plots
from scipy.stats import norm #TODO remove if I move gaussianize func
import math
import function_lib.PCA as PCA
import function_lib.MVG as MVG
import function_lib.LR as LR
import function_lib.model_eval as model_eval
import matplotlib.pyplot as plt


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
    #TODO check if I have to apply PCA only on Training fold.
    #applying PCA
    D_PCA5 = PCA.compute_PCA(5,D) #TODO check se applicare PCA a TRAINING e a EVAL separatamente
    D_PCA6 = PCA.compute_PCA(6,D)
    D_PCA7 = PCA.compute_PCA(7,D)

    #Preparing single-fold dataset (T-0.66% E-0.33%)
    train_size = int(math.ceil((D.shape[1]*0.66)))
    idxSTrain = idx[0:train_size]
    idxSTest = idx[train_size:]
    DTR = D[:,idxSTrain]
    LTR = L[idxSTrain]
    DTE = D[:,idxSTest]
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
        DTR = D[:,idxTrain]
        LTR = L[idxTrain]
        DTE = D[:,idxTest]
        LTE = L[idxTest]
        NOPCA.append(((DTR, LTR), (DTE, LTE)))
        DTR_PCA, DTE_PCA = PCA.compute_PCA(5,DTR,DTE)
        PCA5.append(((DTR_PCA, LTR), (DTE_PCA, LTE))) 
        DTR_PCA, DTE_PCA = PCA.compute_PCA(6,DTR,DTE)
        PCA6.append(((DTR_PCA, LTR), (DTE_PCA, LTE))) 
        DTR_PCA, DTE_PCA = PCA.compute_PCA(7,DTR,DTE)
        PCA7.append(((DTR_PCA, LTR), (DTE_PCA, LTE))) 
    
    return (singleNOPCA, singlePCA5, singlePCA6, singlePCA7) , (NOPCA, PCA5, PCA6, PCA7)

def MVGwrapper(data, prior, mode, k = 0):

    if(mode != "k-fold" and mode != "single-fold"):
        print("Error, enter mode")
        return

    MVGmodels = {
        0 : (MVG.logMVG, "fullMVG"),
        1 : (MVG.logNaiveBayes, "naiveBayesMVG"),
        2 : (MVG.TiedCovariance_logNB, "TiedCovarianceNaiveBayesMVG"),
        3 : (MVG.TiedCovariance_logMVG, "TiedCovarianceMVG")
    }

    
    for (model, modelName) in MVGmodels.values():
        if(mode == "single-fold"):
            print("*******Single Fold %s*******" % (modelName))
            singleFoldMVG(data, prior, model)
        elif(mode == "k-fold"):
            print("*******%d-Fold %s*******" % (k,modelName))
            kfoldMVG(data, prior, model)


def singleFoldMVG(single_data, prior, MVGmodel):
    labels = {
        0 : "Z-norm Full",
        1 : "Z-norm PCA5",
        2 : "Z-norm PCA6",
        3 : "Z-norm PCA7"
    }

    for i, data in enumerate(single_data):
        (DTR, LTR),(DTE, LTE) = data
        for j in range(prior.shape[1]):
            pred_L, llr = MVGmodel(DTR, LTR, DTE, LTE, prior[:,j:j+1])
            prior_class1 = prior[1][j]
            minDCF = model_eval.compute_min_Normalized_DCF(LTE, llr, prior_class1, C_fn = 1 , C_fp =  1)
            print("[Single Fold %s] prior[1] = %.1f minDCF = %.3f," % (labels[i], prior_class1, minDCF) )

def kfoldMVG(kfold_data, prior, MVGmodel):
    labels = {
        0 : "Z-norm Full",
        1 : "Z-norm PCA5",
        2 : "Z-norm PCA6",
        3 : "Z-norm PCA7"
    }
    

    for i, data in enumerate(kfold_data):
        #data is the FULL, PCA5, PCA6... 
        for j in range(prior.shape[1]):
            prior_class1 = prior[1][j]
            K_scores = []
            K_LTE = []
            for fold in data:
                (DTRk, LTRk),(DTEk, LTEk) = fold
                pred_L, llr = MVGmodel(DTRk, LTRk, DTEk, LTEk, prior[:,j:j+1])
                K_scores.append(llr)
                K_LTE.append(LTEk)
            kscores_array = np.concatenate(K_scores).ravel() #computing all the scores for all the folds and then putting them in one single array. Same for the labels.
            kLTE_array = np.concatenate(K_LTE).ravel()
            minDCF = model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, prior_class1, C_fn = 1 , C_fp =  1)
            print("[K-Fold %s] prior-1 = %.1f minDCF = %.3f," % (labels[i], prior_class1, minDCF) )


def kfoldLR_minDCF(kfold_data, pi_t, l ): #kfold_data takes only one foldtype. e.g. only PCA5. Not the whole array. 


    K_scores = []
    K_LTE = []
    for fold in kfold_data:
        (DTRk, LTRk),(DTEk, LTEk) = fold

        llr, pred_L = LR.computeLR(DTRk, LTRk, DTEk , l, pi_t) #TODO lambda handling
        K_scores.append(llr)
        K_LTE.append(LTEk)
    kscores_array = np.concatenate(K_scores).ravel() #computing all the scores for all the folds and then putting them in one single array. Same for the labels.
    kLTE_array = np.concatenate(K_LTE).ravel()
    minDCF = model_eval.compute_min_Normalized_DCF(kLTE_array, kscores_array, pi_t, C_fn = 1 , C_fp =  1)
    #print("[lin-LR] lambda = prior_T = %.1f minDCF = %.3f," % (l, pi_t, minDCF) )
    #TODO FIX PRINT
    return minDCF


def plotLRminDCF(kfold_data, prior_t, mode = "k-fold"):
    print("a)")
    l = np.logspace(-5, 2, num = 20) #limiting the number of points by 50.
    minDCF_array = []
    for lamb in l:
        minDCF_prior = []
        for pi_t in prior_t:
            minDCF = kfoldLR_minDCF(kfold_data, pi_t, lamb)
            minDCF_prior.append(minDCF)
        minDCF_array.append(minDCF_prior) #should result an array of type [ [minDCF-pi_T0, minDCF-pi_T1, minDCF-pi_T2] ...   ]
    
    print("plotting %d-fold" %(kfold_data[0][0][0].shape[0]))
    plots.plotminDCF(l,minDCF_array, prior_t, "lambda")

    print("prova")
    





if __name__ == "__main__":
    DTR, LTR = load("dataset/Train.txt")
    # plots.plot_hist(DTR,LTR, "raw dataset", "plots/1_RawDataset", True)
    # DTR_gau = gaussianize(DTR)
    # plots.plot_hist(DTR_gau,LTR, "gaussianised dataset" , "plots/2_GaussianizedDataset", True)
    DTR_z = z_normalize(DTR)
    #plots.plot_hist(DTR_z, LTR, "Z-Norm", "plots/3_zNormDataset", True)

    #plots.heatmap(DTR_z, LTR,  "Heatmap, Z-norm dataset")

    #plots.plot_correlations(DTR_z, LTR)

    #Create the K-folds 
    k = 5 #num of folds
    singleFoldData, kFoldData = (singleNOPCA, singlePCA5, singlePCA6, singlePCA7) , (NOPCA, PCA5, PCA6, PCA7) = KfoldsGenerator(DTR_z,LTR, k)

    prior_f = np.array([0.5, 0.9, 0.1])
    prior_t = np.array([0.5, 0.1, 0.9])
    prior = np.vstack([prior_f, prior_t])
    



    #MVGwrapper(singleFoldData, prior, mode = "single-fold")
    #MVGwrapper(kFoldData, prior, mode = "k-fold", k = k)
    

    plotLRminDCF(PCA7, prior_t)
    plotLRminDCF(NOPCA, prior_t)







    print("ciao")
