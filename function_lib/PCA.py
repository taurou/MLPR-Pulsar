import numpy as np

def compute_Mean_CovMatr(D):
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    D_centered = D - mu
    #computing the covariance matrix
    C = np.dot(D_centered, D_centered.T)/D_centered.shape[1]
    return mu, C

#compute PCA method #2 - with singular value decomposition
# m = number of dimensions we want , returns the reduced D.
def compute_PCA(m, DTR, DTE = None):    #computing the covariance matrix
    _, C = compute_Mean_CovMatr(DTR)
    #computing the single value decomposition
    U, _, _ = np.linalg.svd(C)
    P = U[:, 0:m] #take the first m columns of the sorted eigenvector
    DTR_PCA = np.dot(P.T, DTR) #apply the projection on the samples D on the base P
    if(DTE is None):
        return DTR_PCA
    else:
        DTE_PCA = np.dot(P.T, DTE)
        return DTE_PCA #if we're considering the evaluation set, return only the eval data on which PCA was performed
