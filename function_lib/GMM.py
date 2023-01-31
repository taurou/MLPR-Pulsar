import numpy as np
import scipy.special
import function_lib.lib as lib
import matplotlib.pyplot as plt

######UTILITIES######

def compute_Mean_CovMatr(D):
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    D_centered = D - mu
    #computing the covariance matrix
    C = np.dot(D_centered, D_centered.T)/D_centered.shape[1]
    return mu, C

def logpdf_GAU_ND(x, mu, c):
    M = x.shape[0]
    logdet_sign, logdet = np.linalg.slogdet(c) #returns the sign and the log-determinant
    const = -M*0.5*np.log(2*np.pi) -0.5*logdet
    c_inv = np.linalg.inv(c) #inversion of the covariance matrix
    return_val = [ const - 0.5*np.dot( np.dot((x[:,i:i+1]-mu).T, c_inv), x[:,i:i+1]-mu) for i in range (x.shape[1]) ]
    return np.array(return_val).ravel()


def GAU_logpdf(X, mu, var):
    # Computes the log-density of the dataset
    X =X.flatten()
    return (-0.5*np.log(2*np.pi))-0.5*np.log(var)-(((X-mu)**2)/(2*var))

######END OF UTILITIES######

def computeStartGMM(X): #returns a list [(1.0,mu,C)]
    mu, C = compute_Mean_CovMatr(X)
    return [(1.0, mu, C)]

def logpdf_GMM(X, gmm): #gmm is a list of type [(w,mu,C), ...] Return a ll for each sample.
    logpdf_array = []
    priors_array = []
    for (w, mu, C) in gmm:
        X_logpdf = logpdf_GAU_ND(X, mu, C)
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
        #print(new_ll)
    #print(new_ll-old_ll)
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
        #print(new_ll)
    #print(new_ll-old_ll)
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
        #print(new_ll)
    #print(new_ll-old_ll)
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


######CLASSIFIER FUNCTION#######

#LGB_mode is tied, diag, full
def GMMBinaryclassification(DTR, LTR, DTE, algorithm_iterations, LBG_mode, alpha = 0.1, psi = 0.1, stop = 1e-6):
    
    numClass = 2 #suppose N classes from 0 to N-1
    DTE_marginals = np.zeros((numClass, DTE.shape[1])) #marginals for the evaluation set
    X = [ DTR[:, LTR == i] for i in range(numClass)]
    for i, Xc in enumerate(X):
        gmm = LBG(Xc, computeStartGMM(Xc), algorithm_iterations, LBG_mode, alpha, psi, stop)
        (marginalLL, _) = logpdf_GMM(DTE, gmm)
        DTE_marginals[ i:i+1, :] += (marginalLL)
    predicted_L = np.argmax(DTE_marginals, axis = 0)

    llr = DTE_marginals[1,:] - DTE_marginals[0,:] #computing the score for the binary case

    return llr.ravel(), predicted_L.ravel() #return scores and labels

def plotGMMNormalDensityOverNormalizedHistogram(dataset, gmm):
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
    
  