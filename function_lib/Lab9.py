import numpy as np
import matplotlib.pyplot as plt
import lib
import scipy.optimize

class SVMClass:
    def __init__(self, DTR, LTR, k = 1.0):
        self.k = k
        self.DTR = DTR
        self.LTR = LTR
        self.X = np.vstack([DTR, np.ones((1,DTR.shape[1]))*self.k ])
        self.Z = 2*LTR - 1
        G = np.dot(self.X.T, self.X)
        self.H = G * lib.vrow(self.Z) * lib.vcol(self.Z)


    def JDual(self, alpha):
        Ha = np.dot(self.H,lib.vcol(alpha))
        aHa = np.dot(lib.vrow(alpha),Ha)
        J = -0.5*aHa.ravel() + alpha.sum() #.sum because a.T DOT 1 will return the sum.  
        J_grad = - Ha + 1
        return J, J_grad.ravel()
    
    def LDual(self, alpha):
        J, J_grad = self.JDual(alpha)
        return -J, -J_grad

    def generate_C_intervals(self, C): #generates the list with intervals (0,C)
        list = [(0,C) for i in range(self.LTR.size)]
        return list

    def w_hat_primal(self, alpha):
        return lib.vcol((lib.vrow(alpha) * lib.vrow(self.Z) * self.X).sum(axis = 1))

    def J_primal_obj(self, C, alpha):
        w_hat = self.w_hat_primal(alpha) 
        w_hat_norm = np.linalg.norm(w_hat)**2
        S = 1 - self.Z*np.dot(w_hat.T, self.X) #w.T*Xi
        scores_0 = np.vstack((S, np.zeros(S.shape[1])))
        max_sum = scores_0.max(axis = 0).sum()
        return 0.5*w_hat_norm + C*max_sum

    def J_duality_gap(self, C, alpha):
        J_primal = self.J_primal_obj(C, alpha)
        J_dual, _ = self.JDual(alpha)
        return J_primal - J_dual

    def compute_scores(self, alpha, DTE):
        Xt_hat = np.vstack([DTE, np.ones((1,DTE.shape[1]))*self.k ])
        w_hat = self.w_hat_primal(alpha)
        return np.dot(w_hat.T,Xt_hat)
    
    def compute_labels(self, alpha, DTE):
        scores = self.compute_scores(alpha, DTE)
        return np.int32((scores > 0).ravel())
        
    def trainSVM(self, C):
        return scipy.optimize.fmin_l_bfgs_b(self.LDual, start, bounds = self.generate_C_intervals(C), factr = 1.0, maxiter = 100000, maxfun = 100000) #passing iprint = 1 makes it print the computing steps 



class SVM_KernelClass:
    def __init__(self, DTR, LTR, kernelType, k = 0.0, c = 0.0, d = 0.0, gamma = 0.0): #K is basically the same as before but we add power of 2 of it at the kernel.
        self.kernelType = kernelType
        self.c = c
        self.d = d
        self.gamma = gamma
        self.k = k
        if(self.kernelType != "Poly" and self.kernelType != "RBF" ):
            print("error, specify kernel type")
        self.LTR = LTR
        self.X = DTR
        self.Z = 2*LTR - 1
        if(kernelType == "Poly"):
            kernel = self.compute_PolyKernel(self.X, self.X, c, d, k)
        elif(kernelType == "RBF"): #Radial Basis Function kernel
            kernel = self.compute_RBFKernel(self.X, self.X, gamma, k)
        self.H = kernel * lib.vrow(self.Z) * lib.vcol(self.Z)

    def compute_PolyKernel(self, X1, X2, c, d, k):
        return (np.dot(X1.T, X2) + c)**d + k**2

    def compute_RBFKernel(self, X1, X2, gamma, k):
        kernel = np.zeros((X1.shape[1], X2.shape[1]))
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                xi = X1[:, i:i+1]
                xj = X2[:, j:j+1]
                diff_sqnorm = np.linalg.norm( xi - xj )**2
                kernel[i,j] = np.exp(-gamma * diff_sqnorm)
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

    def generate_C_intervals(self, C): #generates the list with intervals (0,C)
        list = [(0,C) for i in range(self.LTR.size)]
        return list

    def compute_scores(self, alpha, DTE):
        if(self.kernelType == "Poly"):
            kernel = self.compute_PolyKernel(self.X, DTE, self.c, self.d, self.k)
        elif(self.kernelType == "RBF"): #Radial Basis Function kernel
            kernel = self.compute_RBFKernel(self.X, DTE, self.gamma, self.k)
        return (lib.vcol(alpha) * lib.vcol(self.Z) * kernel).sum(axis = 0)

    def compute_labels(self, alpha, DTE):
        scores = self.compute_scores(alpha, DTE)
        return np.int32((scores > 0).ravel())


    def trainSVM(self, C):
        return scipy.optimize.fmin_l_bfgs_b(self.LDual, start, bounds = self.generate_C_intervals(C), factr = 1.0, maxiter = 100000, maxfun = 100000) #passing iprint = 1 makes it print the computing steps 


if __name__ == '__main__':

    # FIRST PART
    D, L = lib.load_iris_binary()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D, L)
    # logRegObj.logreg_obj(np.array([1,1,1,1,1]))

    SVMObj = SVMClass(DTR, LTR) #use default value k=1
    start = np.zeros((DTR.shape[1],1))
    for C in [0.1, 1.0, 10.0]:
        
        (alpha, v_alpha, d) = SVMObj.trainSVM(C)
        pred_L = SVMObj.compute_labels(alpha, DTE)
        print(C, "k=1", lib.compute_accuracy_error(pred_L, LTE))
        print("primal obj", SVMObj.J_primal_obj(C, alpha))
        print("dual obj", SVMObj.JDual(alpha)[0])
        print("duality gap", SVMObj.J_duality_gap(C,alpha), SVMObj.J_primal_obj(C, alpha) - SVMObj.JDual(alpha)[0])
        print("\n")

    SVMObj10 = SVMClass(DTR, LTR, 10.0) 
    start = np.zeros((DTR.shape[1],1))
    for C in [0.1, 1.0, 10.0]:
        
        (alpha, v_alpha, d) = SVMObj10.trainSVM(C)
        pred_L = SVMObj10.compute_labels(alpha, DTE)
        print(C,  "k=10", lib.compute_accuracy_error(pred_L, LTE))
        print("primal obj", SVMObj10.J_primal_obj(C, alpha))
        print("dual obj", SVMObj10.JDual(alpha)[0], v_alpha)
        print("duality gap", SVMObj10.J_duality_gap(C,alpha), SVMObj10.J_primal_obj(C, alpha) - SVMObj10.JDual(alpha)[0])
        print("\n")


#Poly Kernel
SVMPoly_d2_c0 = SVM_KernelClass(DTR,LTR,kernelType="Poly", k = 0, d = 2.0, c = 0)
(alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
print("Poly k=0, d = 2.0, c = 0, C = 1.0")
print(SVMPoly_d2_c0.JDual(alpha)[0])
print(lib.compute_accuracy_error(pred_L, LTE))

SVMPoly_d2_c0 = SVM_KernelClass(DTR,LTR,kernelType="Poly", k = 0, d = 2.0, c = 1)
(alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
print("Poly k=0, d = 2.0, c = 1, C = 1.0")
print(SVMPoly_d2_c0.JDual(alpha)[0])
print(lib.compute_accuracy_error(pred_L, LTE))

SVMPoly_d2_c0 = SVM_KernelClass(DTR,LTR,kernelType="Poly", k = 1, d = 2.0, c = 0)
(alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
print("Poly k=1, d = 2.0, c = 0, C = 1.0")
print(SVMPoly_d2_c0.JDual(alpha)[0])
print(lib.compute_accuracy_error(pred_L, LTE))

SVMPoly_d2_c0 = SVM_KernelClass(DTR,LTR,kernelType="Poly", k = 1, d = 2.0, c = 1)
(alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
print("Poly k=1, d = 2.0, c = 1, C = 1.0")
print(SVMPoly_d2_c0.JDual(alpha)[0])
print(lib.compute_accuracy_error(pred_L, LTE))


#RBF kernel
SVM_RBF_l1 = SVM_KernelClass(DTR,LTR,kernelType="RBF", k = 0, gamma = 1.0)
(alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
print("RBF k = 0, gamma = 1.0, C = 1.0")
print(SVM_RBF_l1.JDual(alpha)[0])
pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
print(lib.compute_accuracy_error(pred_L, LTE))

SVM_RBF_l1 = SVM_KernelClass(DTR,LTR,kernelType="RBF", k = 0, gamma = 10.0)
(alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
print("RBF k = 0, gamma = 10.0, C = 1.0")
print(SVM_RBF_l1.JDual(alpha)[0])
pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
print(lib.compute_accuracy_error(pred_L, LTE))

SVM_RBF_l1 = SVM_KernelClass(DTR,LTR,kernelType="RBF", k = 1, gamma = 1.0)
(alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
print("RBF k=1, gamma = 1.0, C = 1.0")
print(SVM_RBF_l1.JDual(alpha)[0])
pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
print(lib.compute_accuracy_error(pred_L, LTE))

SVM_RBF_l1 = SVM_KernelClass(DTR,LTR,kernelType="RBF", k = 1, gamma = 10.0)
(alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
print("RBF k=1, gamma = 10.0, C = 1.0")
print(SVM_RBF_l1.JDual(alpha)[0])
pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
print(lib.compute_accuracy_error(pred_L, LTE))

