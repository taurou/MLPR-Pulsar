import numpy
import matplotlib.pyplot as plt

dict = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}

def plot(data, labels):
    features=["Sepal lenght", "Sepal width", "Petal lenght", "Petal width"]


    #sepal length
    for i in range(4):
        plt.figure()
        plt.hist(data[i,labels==0], density=True,alpha = 0.5, label="Iris-setosa")
        plt.hist(data[i,labels==1], density=True,alpha = 0.5, label="Iris-versicolor")
        plt.hist(data[i,labels==2], density=True,alpha = 0.5, label="Iris-virginica")
        plt.xlabel(features[i])
        plt.legend()
    plt.show()

    #scatterplots
    for i in range(4):
        for j in range(4):
            if i == j:
                plt.figure()
                plt.hist(data[i,labels==0], density=True,alpha = 0.5, label="Iris-setosa")
                plt.hist(data[i,labels==1], density=True,alpha = 0.5, label="Iris-versicolor")
                plt.hist(data[i,labels==2], density=True,alpha = 0.5, label="Iris-virginica")
                plt.xlabel(features[i])
                plt.legend()
            else:
                plt.figure()
                plt.scatter(data[i,labels==0], data[j,labels==0], label="Iris-setosa")
                plt.scatter(data[i,labels==1], data[j,labels==1], label="Iris-versicolor")
                plt.scatter(data[i,labels==2], data[j,labels==2], label="Iris-virginica")
                plt.xlabel(features[i])
                plt.ylabel(features[j])
                plt.legend()

        plt.show()
def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))


def load():
    i = 0
    f = open('iris.csv','r')
    data = []
    label = []
    for line in f:
        str = line.strip().split(",")
        datastring = [float(i) for i in str[0:4]]
        data.append(numpy.array(datastring).reshape((4,1)))
        label.append(dict[str[4]])
    #print(data)
    #print(label)
    f.close()

    #Transform and return numpy arrays
    d = numpy.hstack(data)
    l = numpy.array(label)
    print(d.shape, l.shape)
    
    return d, l


if __name__ == '__main__':

    data, labels = load()
    #D0 = data[:,labels==0] #successivamente non li ho più usati e ho integrato labels direttamente nella funzione il plot
    #D1 = data[:,labels==1]
    #D2 = data[:,labels==2]
    
    plot(data,labels)

    #centro i dati

    mu = mcol(data.mean(1))
    data_centered = data - mu
    print(data_centered, data)
    plot(data_centered,labels)
