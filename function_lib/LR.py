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
    x0 = np.zeros(DTR.shape[0] + 1)
    logRegObj = logRegClass(DTR, LTR, l, pi_t)
    (v, J_v, _) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad = True) #passing iprint = 1 makes it print the computing steps 
    Wmin = lib.vcol(v[0:-1]) #this is the value of W found 
    Bmin = v[-1] #this is the value of b found
    #print(J_v)
    S = np.dot(Wmin.T, DTE) + Bmin #computing the scores array for the found W and b.
    LP = np.int32((S > 0).ravel())
    return S, LP  #label predicted using as threshold 0. 

# Compute and return the objective function value. You can 
# retrieve all required information from self.DTR, self.LTR, self.l
# in the main portion, after loading the data, instantiate a new object
# logRegObj = logRegClass(DTR, LTR, l)
# You can now use logRegObj.logreg_obj as objective function:
# scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, ...)


if __name__ == '__main__':


    #BinaryLogisticRegression
    D, L = lib.load_iris_binary()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D, L)
    # logRegObj.logreg_obj(np.array([1,1,1,1,1]))

    x0 = np.zeros(DTR.shape[0] + 1)
    for l in [1e-6, 1e-3, 0.1, 1.0]:
        logRegObj = logRegClass(DTR, LTR, l)
        (v, J_v, _) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad = True) #passing iprint = 1 makes it print the computing steps 
        Wmin = lib.vcol(v[0:-1]) #this is the value of W found 
        Bmin = v[-1] #this is the value of b found
        #print(J_v)
        S = np.dot(Wmin.T, DTE) + Bmin #computing the scores array for the found W and b.
        LP = np.int32((S > 0).ravel())  #label predicted using as threshold 0. 
        acc, err = lib.compute_accuracy_error(LP, LTE)
        print("lambda: ", l, "accuracy: ", acc, "error: ", err, "value of Jfunction: ", J_v)
        #J_v is the value of the objective function given the found w and b.
"""
#Multiclass Logistic Regression
    D, L = lib.load_iris()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D, L)
    #logRegObj = logRegClass(DTR, LTR, l)
    x0 = np.zeros(((DTR.shape[0]+1)*len(set(list(LTR))),)) #creating one dimensional array with W and b (b located at the last K(#dimensions) elements of the array)

    #logRegObj.Multiclass_logreg_obj(np.random.rand( (DTR.shape[0] + 1),   len(set(list(LTR)))    ))

    for l in [1e-6, 1e-3, 0.1, 1.0]:
        logRegObj = logRegClass(DTR, LTR, l)
        (v, J_v, d) = scipy.optimize.fmin_l_bfgs_b(logRegObj.Multiclass_logreg_obj, x0, approx_grad = True) #passing iprint = 1 makes it print the computing steps 
        v = v.reshape((DTR.shape[0] + 1, len(set(list(LTR))))) # TODO questo approccio può essere sbagliato perché potrebero non esserci elementi con label 2 (esempio). elementi con label 0, 1, 3. darebbe 3 come totale, ma in quel caso servirebbero 4!!!!
        Wmin = v[0:-1, :] #column vector
        Bmin = lib.vcol(v[-1, :])
        print(J_v)
        S = np.dot(Wmin.T, DTE) + Bmin #computing the scores array for the found W and b.
        #LP = (S > 0).ravel()  #label predicted using as threshold 0.
        #acc, err = lib.compute_accuracy_error(LP, LTE)
        #print("lambda: ", l, "accuracy: ", acc, "error: ", err, "value of Jfunction: ", J_v)
        SPost = np.argmax(CPosterior, axis=0) #TODO finire calcolando max posterior 
        print("A")
"""