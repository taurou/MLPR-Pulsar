import matplotlib.pyplot as plt
import numpy as np

dataset_features = {
    0: "Mean of the integrated profile",
    1: "Standard deviation of the integrated profile",
    2: "Excess kurtosis of the integrated profile",
    3: "Skewness of the integrated profile",
    4: "Mean of the DM-SNR curve",
    5: "Standard deviation of the DM-SNR curve",
    6: "Excess kurtosis of the DM-SNR curve",
    7: "Skewness of the DM-SNR curve",
    }

def plot_hist(D, L, title = " ", filename = "defaultname", save = False ):

    D0 = D[:, L==0]
    D1 = D[:, L==1]



    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.title(title)
        plt.xlabel(dataset_features[dIdx])
        plt.hist(D0[dIdx, :], density = True, bins=90, alpha = 0.4, label = 'Non-pulsar')
        plt.hist(D1[dIdx, :], density = True, bins=90, alpha = 0.4, label = 'Pulsar')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        if(save):        
            plt.savefig('%s_hist_%d-stepfilled.png' % (filename, dIdx), dpi=250 )
    plt.show()

def heatmap(D, L, title = " ", filename = "defaultname", save = False ):
    labels = { 0: "Whole-dataset", 1 : "Non-pulsar", 2: "Pulsar" }
    corrcoeffs = { 0: np.abs(np.corrcoef(D)), 1 : np.abs(np.corrcoef(D[:,L==0])), 2 : np.abs(np.corrcoef(D[:,L==1]))}
    map_colours = { 0: "Greys", 1 : "Reds", 2: "Blues" } 
    for i in range(len(labels)):
        plt.figure()
        plt.title(title + " - " + labels[i])
        plt.imshow(corrcoeffs[i], cmap=map_colours[i], interpolation='nearest')
        if(save):        
            plt.savefig('%s_heatmap_%s.png' % (filename, labels[i]), dpi=250 )
    plt.show()

def plotminDCF(x,minDCF_array,prior_t, x_label, filename = "minDCF", save = False, logScale = True):
    plt.figure()
    plt.title(filename)
    colours=['r','y','b']
    if(logScale):
        plt.xscale("log")
    else:
        plt.xticks(x) #in case of GMM shows only 1, 2, 4, 8...
    plt.xlabel(x_label)
    plt.ylabel("minDCF")
    plt.xlim([x[0], x[len(x)-1]])
    for idx, pi_t in enumerate(prior_t):
        labelDCF = "minDCF ??=%.1f" % (pi_t)    
        plt.plot(x, minDCF_array[idx], label=labelDCF, color=colours[idx])
        
    plt.legend([ "minDCF ??=%.1f" % (prior) for prior in prior_t ])
    if(save):        
        plt.savefig('%s.png' % (filename), dpi=250 )
        plt.close()
    else:
        plt.show()

def plotminDCF_kernelSVM(x,minDCF_array,parameter_array, prior, kernelType = "quadratic", filename = "kSVM-minDCF", save = False):
    plt.figure()
    plt.title(filename)
    colours=['r','y','b', 'k', 'g']
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xlim([x[0], x[len(x)-1]])
    parameterName = "c"
    if(kernelType=="RBF"):
        parameterName = "??"

    for idx, param in enumerate(parameter_array):
        labelDCF = "minDCF ??=%.1f, %s=%f" % (prior, parameterName, param)    
        plt.plot(x, minDCF_array[idx], label=labelDCF, color=colours[idx])
        
    plt.legend([ "minDCF ??=%.1f, %s=%f" % (prior, parameterName, param) for param in parameter_array ])
    if(save):        
        plt.savefig('%s.png' % (filename), dpi=250 )
        plt.close()
    else:
        plt.show()
