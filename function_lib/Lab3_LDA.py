import Lab3_PCA as PCA
import numpy


def computeSbSw(D,L):  # returns Sb,Sw
    #computing means for the whole dataset and for each class
    mean_vector = []
    # ATTENTION! I used range(3) because i know there are 3 classes, but it's not the case all the time
    for i in set(list(L)):
        mean = D[:, L == i].mean(1).reshape(D.shape[0], 1)
        mean_vector.append(mean)
    #print(mean_vector)

    class_mean = numpy.hstack(mean_vector)
    overall_mean = D.mean(1).reshape(D.shape[0], 1)
    #print(class_mean, overall_mean)
    N = D.shape[1]  # number of total elements N
    # list containing the number of elements per class
    Nc = [D[:, L == i].shape[1] for i in set(list(L))]

    #between class covariance
    Sb = 0
    for i in set(list(L)):
        Sb = Sb + numpy.dot((class_mean[:, i].reshape(D.shape[0], 1)-overall_mean),
                            (class_mean[:, i].reshape(D.shape[0], 1)-overall_mean).T)*Nc[i]
    Sb = Sb / N
    #within class covariance
    Sw = 0
    for i in set(list(L)):  # numero di classi è 3
        for j in range(D[:, L == i].shape[1]):  # number of elements for this class
            x = D[:, L == i][:, j].reshape(D.shape[0], 1)
            Sw = Sw + numpy.dot((x-class_mean[:, i].reshape(D.shape[0], 1)),
                                (x-class_mean[:, i].reshape(D.shape[0], 1)).T)

    Sw = Sw / N
    return Sb, Sw

def computeLDA_generalizedEigProb(Sb,Sw):
    import scipy.linalg
    s, U = scipy.linalg.eigh(Sb,Sw)
    W = U[:, ::-1]
    return W

def computeSol1Genaralized(D,Sb,Sw):
    W = computeLDA_generalizedEigProb(Sb,Sw)[:,0:2]
    print(W)
    Y = numpy.dot(W.T,D)
    return Y

if __name__ == '__main__':
    D, L = PCA.load2()
    Sb, Sw = computeSbSw(D, L)
    Y = computeSol1Genaralized(D,Sb,Sw)
    PCA.plot(Y,L)

