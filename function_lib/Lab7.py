import numpy as np
import matplotlib.pyplot as plt
import lib
import scipy.optimize

#Numerical optimization

def f_approxGrad(array): #with numerical approximation
    y = array[0]
    z = array[1]
    val = (y+3)**2 + np.sin(y) + (z+1)**2
    return (y+3)**2 + np.sin(y) + (z+1)**2

def f_explicitGrad(array):
    y = array[0]
    z = array[1]
    df_dy = 2*(y+3)+np.cos(y)
    df_dz = 2*(z+1)
    return (y+3)**2 + np.sin(y) + (z+1)**2, np.array([df_dy, df_dz])


class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l

    def logreg_obj(self,v): #todo
        w = lib.vcol(v[0:-1]) #column vector
        b = v[-1]
        w_norm = np.linalg.norm(w)**2
        S = np.dot(w.T, self.DTR) + b
        Z = 2*self.LTR - 1
        logexpr = np.logaddexp(0, -Z * S)
        return 0.5*self.l*w_norm + logexpr.mean()
"""
    def Multiclass_logreg_obj(self,v): #todo
        numClasses = len(set(list(self.LTR)))
        v = v.reshape(((self.DTR.shape[0] + 1 ), numClasses)) # TODO questo approccio può essere sbagliato perché potrebero non esserci elementi con label 2 (esempio). elementi con label 0, 1, 3. darebbe 3 come totale, ma in quel caso servirebbero 4!!!!
        w = v[0:-1, :] #column vector
        b = lib.vcol(v[-1, :])
        w_norm = (w*w).sum()
        S = np.dot(w.T, self.DTR) + b #computing the matrix of scores
        S_exp = np.exp(S)
        S_sumexp = S_exp.sum(0)
        S_logsumexp = np.log(S_sumexp)
        Ylog = S - S_logsumexp
        
        T = np.zeros(Ylog.shape, dtype=int)
        for i in set(list(self.LTR)): #every column corresponds to a sample, the i-th row is 1 if belongs to the i-th class
            T[i:i+1, :] = np.multiply(self.LTR == i, 1)
        
        return 0.5*self.l*w_norm - ((T*Ylog).sum())/self.DTR.shape[1]
"""

"""     sum = 0    
        for i in range(self.DTR.shape[0]):
            Ci = self.LTR[i] #Label
            Zi = 2*Ci - 1
            Xi = lib.vcol(self.DTR[:, i:i+1])
            sum += np.logaddexp(0, -Zi*(np.dot(w.T, Xi) + b))
        final_value = 0.5*self.l*w_norm + sum/self.DTR.shape[1]
        return final_value.ravel() #i used ravel in order to make it unidimensional
 """

# Compute and return the objective function value. You can 
# retrieve all required information from self.DTR, self.LTR, self.l
# in the main portion, after loading the data, instantiate a new object
# logRegObj = logRegClass(DTR, LTR, l)
# You can now use logRegObj.logreg_obj as objective function:
# scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, ...)


if __name__ == '__main__':
    # FIRST PART
    x0 = np.array([0,0]) #starting point of the algorith
    #using f without eplicit gradient return (f_approxGrad) aka numerical approximation
    (minpos, minval, d) = scipy.optimize.fmin_l_bfgs_b(f_approxGrad, x0, iprint = 1, approx_grad = True) #passing iprint = 1 makes it print the computing steps 
    print("###1 estimated min: ", minpos, "objective value at the min: ", minval)
    #using f with eplicit gradient return (f_explicitGrad) 
    (minpos, minval, d) = scipy.optimize.fmin_l_bfgs_b(f_explicitGrad, x0, iprint = 1) #passing iprint = 1 makes it print the computing steps 
    print("###2 estimated min: ", minpos, "objective value at the min: ", minval)



    #BinaryLogisticRegression
    D, L = lib.load_iris_binary()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D, L)
    # logRegObj.logreg_obj(np.array([1,1,1,1,1]))

    x0 = np.zeros(DTR.shape[0] + 1)
    for l in [1e-6, 1e-3, 0.1, 1.0]:
        logRegObj = logRegClass(DTR, LTR, l)
        (v, J_v, d) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad = True) #passing iprint = 1 makes it print the computing steps 
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