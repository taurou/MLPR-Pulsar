import numpy as np
import matplotlib.pyplot as plt
import function_lib.lib as lib
import scipy.optimize
class SVMClass:
    def __init__(self, DTR, LTR, kernelType, k, pi_t = 0.0, c = 0.0, d = 2.0, gamma = 0.0): 
        self.kernelType = kernelType
        self.c = c
        self.d = d
        self.pi_t = pi_t
        self.gamma = gamma
        self.k = k
        if(self.kernelType != "poly" and self.kernelType != "RBF" and self.kernelType != "linear" ):
            print("error, specify kernel type")
            return None
        self.LTR = LTR
        self.X = DTR
        self.Z = 2*LTR - 1
        if(kernelType == "linear"):
            self.X_EXT = np.vstack([self.X, np.ones((1,self.X.shape[1]))*self.k ])
            kernel = G = np.dot(self.X_EXT.T, self.X_EXT)
        elif(kernelType == "poly"):
            kernel = self.compute_PolyKernel(self.X, self.X, c, d, k)
        elif(kernelType == "RBF"): #Radial Basis Function kernel
            kernel = self.compute_RBFKernel(self.X, self.X, gamma, k)
        self.H = kernel * lib.vrow(self.Z) * lib.vcol(self.Z)


    def compute_PolyKernel(self, X1, X2, c, d, k):
        return (np.dot(X1.T, X2) + c)**d + k**2

    def compute_RBFKernel(self, X1, X2, gamma, k):
        distances = lib.vcol((X1 ** 2).sum(axis = 0)) + lib.vrow((X2 ** 2).sum(axis = 0)) - 2 * np.dot(X1.T, X2)
        kernel = np.exp(-gamma * distances)
        return kernel + k**2

    def JDual(self, alpha):
        Ha = np.dot(self.H,lib.vcol(alpha))
        aHa = np.dot(lib.vrow(alpha),Ha)
        J = -0.5*aHa.ravel() + alpha.sum() #.sum because a.T DOT 1 will return the sum.  
        J_grad = - Ha + 1
        return J, J_grad.ravel()
    
    def LDual(self, alpha):
        J, J_grad = self.JDual(alpha)
        return -J, -J_grad

    #Functions for the primal of the linear model

    def generate_C_intervals(self, C): #generates the list with intervals (0,C)
        C_intervals = np.array([(0,C) for i in range(self.LTR.size)])
        if(self.pi_t > 0): #balanced SVM
            emp_pi_t = self.LTR[self.LTR==1].shape[0] / self.LTR.shape[0] #empirical pi_t
            C_intervals[self.LTR == 1] = (0, (C*self.pi_t)/emp_pi_t)
            C_intervals[self.LTR == 0] = (0, (C*(1-self.pi_t)/(1-emp_pi_t)))
        return C_intervals

    def compute_scores(self, alpha, DTE):
        if(self.kernelType == "linear"):
            X_EVALEXT = np.vstack([DTE, np.ones((1,DTE.shape[1]))*self.k ])
            w_hat = np.dot(self.X_EXT, lib.vcol(alpha)*lib.vcol(self.Z)).sum(axis = 1)
            return np.dot(w_hat.T,X_EVALEXT)
        elif(self.kernelType == "poly"):
            kernel = self.compute_PolyKernel(self.X, DTE, self.c, self.d, self.k)
        elif(self.kernelType == "RBF"): #Radial Basis Function kernel
            kernel = self.compute_RBFKernel(self.X, DTE, self.gamma, self.k)
        return np.dot(alpha*self.Z, kernel)

    def compute_labels(self, alpha, DTE):
        scores = self.compute_scores(alpha, DTE)
        return scores.ravel(), np.int32((scores > 0).ravel())

    def trainSVM(self, C):
        (alpha, _, _) = scipy.optimize.fmin_l_bfgs_b(self.LDual, np.zeros((self.X.shape[1],1)), bounds = self.generate_C_intervals(C) ) #passing iprint = 1 makes it print the computing steps 
        return alpha
