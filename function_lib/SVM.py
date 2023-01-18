import numpy as np
import matplotlib.pyplot as plt
import lib
import scipy.optimize
class SVMClass:
    def __init__(self, DTR, LTR, kernelType, k, pi_t = 0.0, c = 0.0, d = 0.0, gamma = 0.0): #K is basically the same as before but we add power of 2 of it at the kernel.
        self.kernelType = kernelType
        self.c = c
        self.d = d
        self.pi_t = pi_t
        self.gamma = gamma
        self.k = k
        self.X_linear = -1
        if(self.kernelType != "poly" and self.kernelType != "RBF" and self.kernelType != "linear" ):
            print("error, specify kernel type")
            return None
        self.LTR = LTR
        self.X = DTR
        self.Z = 2*LTR - 1
        if(kernelType == "linear"):
            self.X_linear, kernel = self.compute_LinearG()
        elif(kernelType == "poly"):
            kernel = self.compute_PolyKernel(self.X, self.X, c, d, k)
        elif(kernelType == "RBF"): #Radial Basis Function kernel
            kernel = self.compute_RBFKernel(self.X, self.X, gamma, k)
        self.H = kernel * lib.vrow(self.Z) * lib.vcol(self.Z)

    def compute_LinearG(self):
        X = np.vstack([self.X, np.ones((1,self.X.shape[1]))*self.k ]) 
        G = np.dot(X.T, X)
        return X, G                             

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

    #Functions for the primal of the linear model

    def w_hat_primal(self, alpha):
        return lib.vcol((lib.vrow(alpha) * lib.vrow(self.Z) * self.X_linear).sum(axis = 1))

    def J_primal_obj(self, C, alpha):
        w_hat = self.w_hat_primal(alpha) 
        w_hat_norm = np.linalg.norm(w_hat)**2
        S = 1 - self.Z*np.dot(w_hat.T, self.X_linear) #w.T*Xi
        scores_0 = np.vstack((S, np.zeros(S.shape[1])))
        max_sum = scores_0.max(axis = 0).sum()
        return 0.5*w_hat_norm + C*max_sum

    def J_duality_gap(self, C, alpha):
        J_primal = self.J_primal_obj(C, alpha)
        J_dual, _ = self.JDual(alpha)
        return J_primal - J_dual

    def generate_C_intervals(self, C): #generates the list with intervals (0,C)
        list = np.array([(0,C) for i in range(self.LTR.size)])
        if(self.pi_t > 0): #balanced SVM
            emp_pi_t = LTR[LTR==1].shape[0]
            list[self.LTR == 1] = (0, (C*self.pi_t)/emp_pi_t)
            list[self.LTR == 0] = (0, (C*(1-self.pi_t)/(1-emp_pi_t)))
        return list

    def compute_scores(self, alpha, DTE):
        if(self.kernelType == "linear"):
            Xt_hat = np.vstack([DTE, np.ones((1,DTE.shape[1]))*self.k ])
            w_hat = self.w_hat_primal(alpha)
            return np.dot(w_hat.T,Xt_hat)
        elif(self.kernelType == "poly"):
            kernel = self.compute_PolyKernel(self.X, DTE, self.c, self.d, self.k)
        elif(self.kernelType == "RBF"): #Radial Basis Function kernel
            kernel = self.compute_RBFKernel(self.X, DTE, self.gamma, self.k)
        return (lib.vcol(alpha) * lib.vcol(self.Z) * kernel).sum(axis = 0)

    def compute_labels(self, alpha, DTE):
        scores = self.compute_scores(alpha, DTE)
        return scores, np.int32((scores > 0).ravel())


    def trainSVM(self, C):
        return scipy.optimize.fmin_l_bfgs_b(self.LDual, start, bounds = self.generate_C_intervals(C), factr = 1.0, maxiter = 100000, maxfun = 100000) #passing iprint = 1 makes it print the computing steps 


if __name__ == '__main__':

    # FIRST PART
    D, L = lib.load_iris_binary()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D, L)
    # logRegObj.logreg_obj(np.array([1,1,1,1,1]))

    SVMObj = SVMClass(DTR, LTR, kernelType= "linear", k = 1.0) #use default value k=1
    start = np.zeros((DTR.shape[1],1))
    for C in [0.1, 1.0, 10.0]:
        
        (alpha, v_alpha, d) = SVMObj.trainSVM(C)
        _, pred_L = SVMObj.compute_labels(alpha, DTE)
        print(C, "k=1", lib.compute_accuracy_error(pred_L, LTE))
        print("primal obj", SVMObj.J_primal_obj(C, alpha))
        print("dual obj", SVMObj.JDual(alpha)[0])
        print("duality gap", SVMObj.J_duality_gap(C,alpha), SVMObj.J_primal_obj(C, alpha) - SVMObj.JDual(alpha)[0])
        print("\n")

    SVMObj10 = SVMClass(DTR, LTR, kernelType= "linear", k = 10.0) 
    start = np.zeros((DTR.shape[1],1))
    for C in [0.1, 1.0, 10.0]:
        
        (alpha, v_alpha, d) = SVMObj10.trainSVM(C)
        _, pred_L = SVMObj10.compute_labels(alpha, DTE)
        print(C,  "k=10", lib.compute_accuracy_error(pred_L, LTE))
        print("primal obj", SVMObj10.J_primal_obj(C, alpha))
        print("dual obj", SVMObj10.JDual(alpha)[0], v_alpha)
        print("duality gap", SVMObj10.J_duality_gap(C,alpha), SVMObj10.J_primal_obj(C, alpha) - SVMObj10.JDual(alpha)[0])
        print("\n")


    #Poly Kernel
    SVMPoly_d2_c0 = SVMClass(DTR,LTR,kernelType="poly", k = 0, d = 2.0, c = 0)
    (alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
    _, pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
    print("Poly k=0, d = 2.0, c = 0, C = 1.0")
    print(SVMPoly_d2_c0.JDual(alpha)[0])
    print(lib.compute_accuracy_error(pred_L, LTE))

    SVMPoly_d2_c0 = SVMClass(DTR,LTR,kernelType="poly", k = 0, d = 2.0, c = 1)
    (alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
    _, pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
    print("Poly k=0, d = 2.0, c = 1, C = 1.0")
    print(SVMPoly_d2_c0.JDual(alpha)[0])
    print(lib.compute_accuracy_error(pred_L, LTE))

    SVMPoly_d2_c0 = SVMClass(DTR,LTR,kernelType="poly", k = 1, d = 2.0, c = 0)
    (alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
    _, pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
    print("Poly k=1, d = 2.0, c = 0, C = 1.0")
    print(SVMPoly_d2_c0.JDual(alpha)[0])
    print(lib.compute_accuracy_error(pred_L, LTE))

    SVMPoly_d2_c0 = SVMClass(DTR,LTR,kernelType="poly", k = 1, d = 2.0, c = 1)
    (alpha, v_alpha, _) = SVMPoly_d2_c0.trainSVM(1.0)
    _, pred_L = SVMPoly_d2_c0.compute_labels(alpha, DTE)
    print("Poly k=1, d = 2.0, c = 1, C = 1.0")
    print(SVMPoly_d2_c0.JDual(alpha)[0])
    print(lib.compute_accuracy_error(pred_L, LTE))


    #RBF kernel
    SVM_RBF_l1 = SVMClass(DTR,LTR,kernelType="RBF", k = 0, gamma = 1.0)
    (alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
    print("RBF k = 0, gamma = 1.0, C = 1.0")
    print(SVM_RBF_l1.JDual(alpha)[0])
    _, pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
    print(lib.compute_accuracy_error(pred_L, LTE))

    SVM_RBF_l1 = SVMClass(DTR,LTR,kernelType="RBF", k = 0, gamma = 10.0)
    (alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
    print("RBF k = 0, gamma = 10.0, C = 1.0")
    print(SVM_RBF_l1.JDual(alpha)[0])
    _, pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
    print(lib.compute_accuracy_error(pred_L, LTE))

    SVM_RBF_l1 = SVMClass(DTR,LTR,kernelType="RBF", k = 1, gamma = 1.0)
    (alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
    print("RBF k=1, gamma = 1.0, C = 1.0")
    print(SVM_RBF_l1.JDual(alpha)[0])
    _, pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
    print(lib.compute_accuracy_error(pred_L, LTE))

    SVM_RBF_l1 = SVMClass(DTR,LTR,kernelType="RBF", k = 1, gamma = 10.0)
    (alpha, v_alpha, _) = SVM_RBF_l1.trainSVM(1.0)
    print("RBF k=1, gamma = 10.0, C = 1.0")
    print(SVM_RBF_l1.JDual(alpha)[0])
    _, pred_L = SVM_RBF_l1.compute_labels(alpha, DTE)
    print(lib.compute_accuracy_error(pred_L, LTE))

