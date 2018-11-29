import numpy as np
from matplotlib import pyplot as plt
import  time
def loadData():
    '''
    dataset = np.array([[1,1],[2,2],[2,0],
                         [0,0],[1,0],[0,1]])
    tagset = np.array([1, 1, 1, -1, -1, -1])
    '''
    dataset = np.loadtxt("TrainSamples.csv", dtype=np.float, delimiter=",")
    tagset = np.loadtxt("TrainLabels.csv", dtype=np.float, delimiter=",")
    return dataset, tagset

def lmse(dataset,tagset):
    num, dim = np.shape(dataset)
    b = np.ones((num,1))* 1
    eta = 0.000001
    data_tag = dataset.copy()
    for i in range(num):
        data_tag[i] = tagset[i] * data_tag[i]
    tagset.reshape((num,1))
    data_tag = np.column_stack((tagset,data_tag))
    # print data_tag
    a = np.ones(((dim+1),1)) / (dim+1)
    if (np.linalg.det(np.dot(data_tag.T, data_tag)) < 0.01):
        a = np.dot(np.dot(np.linalg.inv(np.dot(data_tag.T, data_tag) + 0.01*np.eye((dim+1))), data_tag.T), b)
    else:
        a = np.dot(np.dot(np.linalg.inv(np.dot(data_tag.T, data_tag)), data_tag.T),b)
    # print a
    return a / a[0]

def showPlt(dataset, tagset, w):
    num = np.shape(dataset)[0]
    marker = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(num):
        marker_type = int(tagset[i,] + 1)
        plt.plot(dataset[i, 0], dataset[i, 1], marker[marker_type], markersize=7)
    plt.xlim(xmax=3, xmin=-1)
    plt.ylim(ymax=3, ymin=-1)
    k = -1.0 * w[1] / w[2]
    b = -1.0 * w[0] / w[2]
    print ('LMSE')
    print (w, b)
    print (k)
    x = np.array(range(-5, 5))
    plt.plot(x, k * x + b, color='b')
    plt.title('LMSE')#('L:'+str(w[0]) + ' * x1 + ' + str(w[1]) + ' * x2 + '+str(b))
    plt.savefig('lmse.jpg')
    plt.show()

if __name__ == '__main__':

    dataset, tagset = loadData()
    np.random.seed(2)
    for data in dataset:
        data+=np.random.rand(1)*2
    num,dim = np.shape(dataset)
    '''
    a = lmse(dataset,tagset)
    a = a / a[0]
    showPlt(dataset, tagset, a)
    print (a)
    '''
    a = []
    testresult = []

    for id in range(10):
        tag = tagset.copy()
        for n in range(num):
            if (tag[n] == id ):
                tag[n] = 1
            else:
                tag[n] = -1
        result = lmse(dataset, tag)
        a.append(result)
    # print a;
    liketag = np.ones((num, 1))
    example = np.column_stack((liketag, dataset))

    testdata = np.loadtxt("TestSamples.csv", dtype=np.float, delimiter=",")
    testtag = np.loadtxt("TestLabels.csv", dtype=np.float, delimiter=",")
    num, dim = np.shape(testdata)
    liketag = np.ones((num, 1))
    dataset = np.column_stack((liketag, testdata))
    for i in range(num):
        normaldata = dataset[i,:]
        flag = 0
        category = 10
        for id in range(10):
            judge = np.dot(a[id].T, normaldata)
            # print id, judge
            if (judge < 0):
                flag += 1
                category = id
        if (flag == 1):
            # print category
            testresult.append(category)
        else:
            testresult.append(10)
    print (testresult)

    num = [0]*10
    for i in range(np.shape(testdata)[0]):
            num[int(testtag[i])] += 1
    print (num)
    num = [0] * 10
    for i in range(np.shape(testdata)[0]):
        if (testtag[i] == testresult[i]):
            num[testresult[i]] += 1
    print (num)
