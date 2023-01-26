import numpy as np
import matplotlib.pyplot as plt
import function_lib.lib as lib
import scipy.optimize

#Numerical optimization


class logRegClass:
    def __init__(self, DTR, LTR, l, pi_t):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pi_t = pi_t

    def logreg_obj(self,v): #todo
        w = lib.vcol(v[0:-1]) #column vector
        b = v[-1]
        w_norm = np.linalg.norm(w)**2
        Z = 2*self.LTR - 1
        S1 = np.dot(w.T, self.DTR[:,self.LTR == 1]) + b    #for z = 1
        S_1 = np.dot(w.T, self.DTR[:,self.LTR == 0]) + b   #for z = -1
        logexpr1 = np.logaddexp(0, -Z[self.LTR == 1] * S1)          #for z = 1
        logexpr_1 = np.logaddexp(0, -Z[self.LTR == 0] * S_1)        #for z = -1
        n_t =  (self.LTR == 1).sum()
        n_f = (self.LTR == 0).sum()
        
        
        return 0.5*self.l*w_norm + (self.pi_t*(logexpr1.sum()))/n_t + ((1 - self.pi_t)*(logexpr_1.sum()))/n_f


def computeLR(DTR, LTR, DTE ,l, pi_t):
    logRegObj = logRegClass(DTR, LTR, l, pi_t)
    (v, _, _) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True) #passing iprint = 1 makes it print the computing steps 
    Wmin = lib.vcol(v[0:-1]) #this is the value of W found 
    Bmin = v[-1] #this is the value of b found
    S = (np.dot(Wmin.T, DTE) + Bmin).ravel() #computing the scores array for the found W and b.
    LP = np.int32((S > 0).ravel())
    return S, LP  #label predicted using as threshold 0. 

def calibrateScores(scores, labels, l, pi_t = 0.5):
    scores = lib.vrow(scores)
    logRegObj = logRegClass(scores, labels, l, pi_t)
    (v, _, _) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, np.zeros(len(scores) + 1), approx_grad = True) #passing iprint = 1 makes it print the computing steps 
    alpha = lib.vcol(v[0:-1]) #this is the value of W found 
    beta_star = v[-1] #this is the value of b found
    beta = beta_star - np.log(pi_t/(1-pi_t))
    return scores*alpha + beta

