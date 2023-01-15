import numpy as np
import matplotlib.pyplot as plt

def load2():
    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

#compute PCA method #1
def compute_PCA_projection(m,D):
    #centering the dataset
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    D_centered = D - mu
    #computing the covariance matrix
    C = np.dot(D_centered, D_centered.T)/D_centered.shape[1]
    #computing the eigenvectors and eigenvalues
    s, U = np.linalg.eigh(C) #because the matrix is symmetric, otherwise I should have used linalg.eig 
    #the used function returns the eigenvalues and eigenvector sorted in ascending order (.eig doesn't)
    P = U [:, ::-1][:, 0:m] #1st: invert the order to descendant, 2nd: take the first m columns
    print(P)

    x = D
    y = np.dot(P.T, x) #apply the projection on the samples D on the base P
    return y

#compute PCA method #2 - with singular value decomposition
def compute_PCA_projection2(m,D):
    #centering the dataset
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    D_centered = D - mu
    #computing the covariance matrix
    C = np.dot(D_centered, D_centered.T)/D_centered.shape[1]
    #computing the single value decomposition
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m] #take the first m columns
    print(P)
    x = D
    y = np.dot(P.T, x) #apply the projection on the samples D on the base P
    return y


def plot(y,L):
    plt.figure()
    plt.scatter(y[0,L==0], y[1,L==0], label="Iris-setosa")
    plt.scatter(y[0,L==1], y[1,L==1], label="Iris-versicolor")
    plt.scatter(y[0,L==2], y[1,L==2], label="Iris-virginica")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    D, L = load2()
    y = compute_PCA_projection2(2,D)
    plot(y,L)
    A = np.load("3_Dimensionality_Reduction/Solution/IRIS_PCA_matrix_m4.npy")

    print(A)
