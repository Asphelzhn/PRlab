import numpy as np
from matplotlib import pyplot as plt

def loadData():
    dataset = np.array([[1,1],[2,2],[2,0],
                        [0,0],[1,0],[0,1]])
    tagset = np.array([1,1,1,-1,-1,-1])
    return dataset, tagset

def initPara(data):
    dim = np.shape(data)[1]
    eta = 0.01
    w = np.zeros(dim)
    # w.reshape(dim,1)
    b = 0.0
    return eta, w, b

def perceptron(dataset, tagset):
    num, dim = np.shape(dataset)
    eta, w, b = initPara(dataset)
    flag = 1
    while flag != 0:
        flag = 0
        for i in range(num):
            data = dataset[i]
            tag = tagset[i]
            result = (np.dot(w, data) + b) * tag
            if result <= 0.0:
                w = w + eta * tag * data
                b = b + eta * tag
                flag = 1
                print (w, b)
    return w, b

def showPlt(dataset, tagset, w, b):
    num = np.shape(dataset)[0]
    marker = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(num):
        marker_type = int(tagset[i,] + 1)
        plt.plot(dataset[i, 0], dataset[i, 1], marker[marker_type], markersize=7)
    plt.xlim(xmax=3, xmin=-1)
    plt.ylim(ymax=3, ymin=-1)
    k = -1.0 * w[1] / w[0]
    b = -1.0 * b / w[0]
    print ('Percrptron')
    print (w, b)
    x = np.array(range(-5, 5))
    plt.plot(x, k * x + b, color='b')
    plt.title('Percrptron')
    plt.savefig('percrptron.jpg')
    plt.show()

if __name__ == '__main__':
    dataset, tagset = loadData()
    w, b = perceptron(dataset, tagset)
    showPlt(dataset,tagset,w,b)
