import numpy as np
import scipy.special
import lib
import matplotlib.pyplot as plt
import Lab4 as LAB4
import GMM_load 

def GAU_logpdf(X, mu, var):
    # Computes the log-density of the dataset
    X =X.flatten()
    return (-0.5*np.log(2*np.pi))-0.5*np.log(var)-(((X-mu)**2)/(2*var))

def computeStartGMM(X): #returns a list [(1.0,mu,C)]
    mu, C = LAB4.compute_Mean_CovMatr(X)
    return [(1.0, mu, C)]

def logpdf_GMM(X, gmm): #gmm is a list of type [(w,mu,C), ...] Return a ll for each sample.
    logpdf_array = []
    priors_array = []
    for (w, mu, C) in gmm:
        X_logpdf = LAB4.logpdf_GAU_ND(X, mu, C)
        logpdf_array.append(X_logpdf)
        priors_array.append(w)
    S = np.vstack(logpdf_array)
    priors = lib.vcol(np.array(priors_array)) #create a column vector with the M priors.
    S += np.log(priors) #adding the priors to the llmatrix S
    logdens = scipy.special.logsumexp(S, axis=0)
    return (logdens, S) #returns the marginal log densities and the joint

#mode "full"; "tied" : tied-covariance; "diag" : diagonal

def eig_constraint(sigma,psi):
    U, s, _ = np.linalg.svd(sigma)
    s[s < psi] = psi
    return np.dot(U, lib.vcol(s) * U.T)


def GMM_EM_full(X, gmm, psi = 0.1, stop = 1e-6 ):
    new_ll = None 
    old_ll = None
    G = len(gmm)
    N = X.shape[1]
    D = X.shape[0] #number of features

    while old_ll is None or new_ll - old_ll > stop:
        old_ll = new_ll
        (logSMarg, logSJoint ) = logpdf_GMM(X,gmm)
        new_ll = np.sum(logSMarg)/N #compute the log likelihood for the whole dataset  
        
        #E step
        P = np.exp(logSJoint - logSMarg) #P contains all the components responsabilities

        #M step
        gamma = P #gamma is the posterior probabilty
        Z = gamma.sum(axis = 1)
        F = np.zeros((D, G))
        S = []
        newGMM = []
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (lib.vrow(gamma)*X).sum(axis = 1)
            S = np.dot(X, (lib.vrow(gamma)*X).T)
            w = Z/N
            mu = lib.vcol(F/Z)
            sigma = S/Z - np.dot(mu,mu.T)
            #to avoid degenerate solutions
            sigma = eig_constraint(sigma,psi)
            newGMM.append((w,mu,sigma))

        gmm=newGMM
        print(new_ll)
    print(new_ll-old_ll)
    return gmm

def GMM_EM_diag(X, gmm, psi = 0.1, stop = 1e-6 ):
    new_ll = None 
    old_ll = None
    G = len(gmm)
    N = X.shape[1]
    D = X.shape[0] #number of features

    while old_ll is None or new_ll - old_ll > stop:
        old_ll = new_ll
        (logSMarg, logSJoint ) = logpdf_GMM(X,gmm)
        new_ll = np.sum(logSMarg)/N #compute the log likelihood for the whole dataset  
        
        #E step
        P = np.exp(logSJoint - logSMarg) #P contains all the components responsabilities

        #M step
        gamma = P #gamma is the posterior probabilty
        Z = gamma.sum(axis = 1)
        F = np.zeros((D, G))
        S = []
        newGMM = []
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (lib.vrow(gamma)*X).sum(axis = 1)
            S = np.dot(X, (lib.vrow(gamma)*X).T)
            w = Z/N
            mu = lib.vcol(F/Z)
            sigma = ( S/Z - np.dot(mu,mu.T) ) * np.eye(D) #only the diagonal in the Diagonal model
            #to avoid degenerate solutions
            sigma = eig_constraint(sigma,psi)
            newGMM.append((w,mu,sigma))

        gmm=newGMM
        print(new_ll)
    print(new_ll-old_ll)
    return gmm

def GMM_EM_tied(X, gmm, psi = 0.1, stop = 1e-6 ):
    new_ll = None 
    old_ll = None
    G = len(gmm)
    N = X.shape[1]
    D = X.shape[0] #number of features

    while old_ll is None or new_ll - old_ll > stop:
        old_ll = new_ll
        (logSMarg, logSJoint ) = logpdf_GMM(X,gmm)
        new_ll = np.sum(logSMarg)/N #compute the log likelihood for the whole dataset  
        
        #E step
        P = np.exp(logSJoint - logSMarg) #P contains all the components responsabilities

        #M step
        gamma = P #gamma is the posterior probabilty
        Z = gamma.sum(axis = 1)
        F = np.zeros((D, G))
        S = []
        newGMM = []
        sigmaSum = np.zeros((D,D))
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (lib.vrow(gamma)*X).sum(axis = 1)
            S = np.dot(X, (lib.vrow(gamma)*X).T)
            w = Z/N
            mu = lib.vcol(F/Z)
            sigma = S/Z - np.dot(mu,mu.T)
            sigmaSum += Z*sigma #Z*sigma
            #to avoid degenerate solutions
            newGMM.append((w,mu,0))

        sigma_tied = eig_constraint(sigmaSum/N,psi)
        for i,g in enumerate(newGMM):
            newGMM[i] = (g[0],g[1],sigma_tied)
        gmm=newGMM
        print(new_ll)
    print(new_ll-old_ll)
    return gmm

def LBGsplit(gmm, alpha):
    gmm2G = []
    for g in gmm:
        U, s, _ = np.linalg.svd(g[2])
        d = U[:, 0:1] * s[0] **0.5 * alpha
        w = g[0]
        mu = g[1]
        sigma = g[2]
        gmm2G.append((w/2, mu + d, sigma))
        gmm2G.append((w/2, mu - d, sigma))
    return gmm2G


def LBG(X, gmm, numIterations, mode, alpha = 0.1, psi = 0.1, stop = 1e-6):
    if (mode == "tied"):
        EM = GMM_EM_tied
    elif(mode == "full"):
        EM = GMM_EM_full
    elif(mode == "diag"):
        EM = GMM_EM_diag
    else:
        print("error in mode field")
        return None
    gmm = EM(X, gmm, psi, stop)

    for i in range(numIterations):
        splittedGMM = LBGsplit(gmm, alpha)
        gmm = EM(X, splittedGMM, psi, stop)
    
    return gmm

def GMMclassification(DTR, LTR, DTE, LTE, LBG_mode, algorithm_iterations, alpha = 0.1, psi = 0.1, stop = 1e-6):
    
    numClass = np.max(LTR) + 1 #suppose N classes from 0 to N-1
    DTE_marginals = np.zeros((numClass, DTE.shape[1])) #marginals for the evaluation set
    X = [ DTR[:, LTR == i] for i in range(numClass)]
    for i, Xc in enumerate(X):
        gmm = LBG(Xc, computeStartGMM(Xc), algorithm_iterations, LBG_mode, alpha, psi, stop)
        (marginalLL, _) = logpdf_GMM(DTE, gmm)
        DTE_marginals[ i:i+1, :] += (marginalLL)
    predicted_L = np.argmax(DTE_marginals, axis = 0)

    return predicted_L


def plotNormalDensityOverNormalizedHistogram(dataset, gmm):
    # Function used to plot the computed normal density over the normalized histogram
    dataset = dataset.flatten()
    plt.figure()
    plt.hist(dataset, bins=30, edgecolor='black', linewidth=0.5, density=True)
    # Define an array of equidistant 1000 elements between -10 and 5
    XPlot = np.linspace(-10, 5, 1000)
    # We should plot the density, not the log-density, so we need to use np.exp
    y = np.zeros(1000)
    for i in range(len(gmm)):
        y += gmm[i][0]*np.exp(GAU_logpdf(XPlot, gmm[i][1], gmm[i][2])).flatten()
    plt.plot(XPlot, y, color="red", linewidth=3)
    plt.show()
    return  
    


if __name__ == "__main__":
    gmm = GMM_load.load_gmm("10_Gaussian_Mixture_Models\Data\GMM_4D_3G_init.json")
    
    X = np.load("10_Gaussian_Mixture_Models/Data/GMM_data_4D.npy")
    #(logdens, S) = logpdf_GMM(X,gmm)
    #logdens_solution = np.load("10_Gaussian_Mixture_Models/Data/GMM_4D_3G_init_ll.npy")
    #print(logdens_solution - logdens)

    print("GMM_EM")
    #gmm4D = GMM_EM_full(X,gmm)
    #gmm4D = GMM_EM_tied(X,gmm)
    
    X1D = np.load("10_Gaussian_Mixture_Models/Data/GMM_data_1D.npy")
    #gmm1D = GMM_EM_tied(X1D,gmm)
    '''
    plotNormalDensityOverNormalizedHistogram(X1D,gmm1D)
    gmm1D = GMM_EM_full(X1D,gmm)
    plotNormalDensityOverNormalizedHistogram(X1D,gmm1D)
    gmm1D = GMM_EM_diag(X1D,gmm)
    plotNormalDensityOverNormalizedHistogram(X1D,gmm1D)
    '''

    #mu, C = LAB4.compute_Mean_CovMatr(X)
    #LBG(X, computeStartGMM(X), 2, "full")
    (D,L) = lib.load_iris()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D,L)
    k=0
    pred = GMMclassification(DTR, LTR, DTE, LTE, "diag", k, psi=0.01)
    print(lib.compute_accuracy_error(LTE, pred)[1])
    
    #comparewith = GMM_load.load_gmm("10_Gaussian_Mixture_Models\Data\GMM_4D_3G_EM.json")
    #comparewith_1 = np.array([[i[0],i[1],i[2]] for i in comparewith])
    #print(gmm_1 - comparewith_1)
'''
    
    gmm = GMM_load.load_gmm("10_Gaussian_Mixture_Models\Data\GMM_1D_3G_init.json")
    X = np.load("10_Gaussian_Mixture_Models/Data/GMM_data_1D.npy")
    (logdens, S) = logpdf_GMM(X,gmm)
    logdens_solution = np.load("10_Gaussian_Mixture_Models/Data/GMM_1D_3G_init_ll.npy")
    print(logdens_solution - logdens)

'''    
  