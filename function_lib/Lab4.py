import numpy as np
import matplotlib.pyplot as plt


def logpdf_GAU_ND(x, mu, c):
    M = x.shape[0]


    logdet_sign, logdet = np.linalg.slogdet(c) #returns the sign and the log-determinant
    const = -M*0.5*np.log(2*np.pi) -0.5*logdet
    c_inv = np.linalg.inv(c) #inversion of the covariance matrix
    return_val = [ const - 0.5*np.dot( np.dot((x[:,i:i+1]-mu).T, c_inv), x[:,i:i+1]-mu) for i in range (x.shape[1]) ]
    return np.array(return_val).ravel()



def compute_Mean_CovMatr(D):
    mu = D.mean(1)
    mu = mu.reshape((mu.size,1))
    D_centered = D - mu
    #computing the covariance matrix
    C = np.dot(D_centered, D_centered.T)/D_centered.shape[1]
    return mu, C

def ll(x, mu, c):
    x_gaussian_density = logpdf_GAU_ND(x,mu,c)
    return x_gaussian_density.sum()

if __name__ == '__main__':

    #ES1
    plt.figure()
    XPlot=np.linspace(-8,12,1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape((1,XPlot.size)), m, C)))
    plt.show()

    #comparison with uni-dimensional solution
    pdfSol = np.load('4_Multivariate_Normal_Densities/Solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(XPlot.reshape(1,XPlot.size), m, C)
    print(np.abs(pdfSol - pdfGau).max())      

    #comparison with provided N-dimensional solution
    XND = np.load('4_Multivariate_Normal_Densities/Solution/XND.npy')
    mu = np.load('4_Multivariate_Normal_Densities/Solution/muND.npy')
    C = np.load('4_Multivariate_Normal_Densities/Solution/CND.npy')
    pdfSol = np.load('4_Multivariate_Normal_Densities/Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())

    #ES2

    #test on first dataset
    x = np.load('4_Multivariate_Normal_Densities/Solution/XND.npy')
    mu, C = compute_Mean_CovMatr(x)
    print(mu,C)
    log_likelihood= ll(x, mu, C)
    print(log_likelihood)



    #test on second dataset
    x = np.load('4_Multivariate_Normal_Densities/Solution/X1D.npy')
    mu, C = compute_Mean_CovMatr(x)
    print(mu,C)
    #visualize how well the estimated density fits the samples
    plt.figure()
    plt.hist(x.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1,XPlot.size), mu, C)))
    plt.show()

    log_likelihood = ll(x, mu, C)
    print(log_likelihood)
