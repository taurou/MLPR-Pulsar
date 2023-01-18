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
            plt.savefig('%s_hist_%d-stepfilled.png' % (filename, dIdx) )
    plt.show()

def heatmap(D, L, title = " ", filename = "defaultname", save = False ):
    labels = { 0: "Dataset", 1 : "Non-pulsar", 2: "Pulsar" }
    corrcoeffs = { 0: np.abs(np.corrcoef(D)), 1 : np.abs(np.corrcoef(D[:,L==0])), 2 : np.abs(np.corrcoef(D[:,L==1]))}
    map_colours = { 0: "Greys", 1 : "Reds", 2: "Blues" } 
    for i in range(len(labels)):
        plt.figure()
        plt.title(title + " - " + labels[i])
        plt.imshow(corrcoeffs[i], cmap=map_colours[i], interpolation='nearest')
        if(save):        
            plt.savefig('%s_heatmap_%s.png' % (filename, labels[i]) )
    plt.show()
    

def plotminDCF(x,minDCF_array,prior_t, x_label):
    plt.figure()
    colours=['r','y','b']
    plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("minDCF")
    #plt.xlim([l[0], l[len(l)]])
    for idx, pi_t in enumerate(prior_t):
        labelDCF = "minDCF pi = %.1f" % (pi_t)    
        plt.plot(x, np.vstack(minDCF_array)[:,idx], label=labelDCF, color=colours[idx])
        
    plt.legend([ "minDCF pi=%.1f" % (prior) for prior in prior_t ])
    plt.show()
