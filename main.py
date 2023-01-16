import numpy as np
import function_lib.lib as lib
import function_lib.Lab4 as LAB4
import function_lib.plots as plots
from scipy.stats import norm #TODO remove if I move gaussianize func
import math
import function_lib.PCA as PCA

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
    D_PCA5 = PCA.compute_PCA(5,D)
    D_PCA6 = PCA.compute_PCA(6,D)
    D_PCA7 = PCA.compute_PCA(7,D)

    #Preparing single-fold dataset (T-0.66% E-0.33%)
    train_size = int(math.ceil((D.shape[1]*0.66)))
    idxSTrain = idx[0:train_size]
    idxSTest = idx[train_size:]

    singleNOPCA = ((D[:,idxSTrain], L[idxSTrain]),(D[:,idxSTest], L[idxSTest]))
    singlePCA5 = ((D_PCA5[:,idxSTrain], L[idxSTrain]),(D_PCA5[:,idxSTest], L[idxSTest]))
    singlePCA6 = ((D_PCA6[:,idxSTrain], L[idxSTrain]),(D_PCA6[:,idxSTest], L[idxSTest])) 
    singlePCA7 = ((D_PCA7[:,idxSTrain], L[idxSTrain]),(D_PCA7[:,idxSTest], L[idxSTest])) 
    

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
        NOPCA.append(((D[:,idxTrain], L[idxTrain]), (D[:,idxTest], L[idxTest])))
        PCA5.append(((D_PCA5[:,idxTrain], L[idxTrain]), (D_PCA5[:,idxTest], L[idxTest]))) 
        PCA6.append(((D_PCA6[:,idxTrain], L[idxTrain]), (D_PCA6[:,idxTest], L[idxTest])))
        PCA7.append(((D_PCA7[:,idxTrain], L[idxTrain]), (D_PCA7[:,idxTest], L[idxTest])))
    
    return singleNOPCA, singlePCA5, singlePCA6, singlePCA7, NOPCA, PCA5, PCA6, PCA7 

if __name__ == "__main__":
    DTR, LTR = load("dataset/Train.txt")
    print("hello")
    # plots.plot_hist(DTR,LTR, "raw dataset", "plots/1_RawDataset", True)
    # DTR_gau = gaussianize(DTR)
    # plots.plot_hist(DTR_gau,LTR, "gaussianised dataset" , "plots/2_GaussianizedDataset", True)
    DTR_z = z_normalize(DTR)
    #plots.plot_hist(DTR_z, LTR, "Z-Norm", "plots/3_zNormDataset", True)

    #plots.heatmap(DTR_z, LTR,  "Heatmap, Z-norm dataset")

    #plots.plot_correlations(DTR_z, LTR)

    #Create the K-folds 
    k = 5 #num of folds
    singleNOPCA, singlePCA5, singlePCA6, singlePCA7, kNOPCA, kPCA5, kPCA6, kPCA7 = KfoldsGenerator(DTR_z,LTR, k)
    print("ciao")