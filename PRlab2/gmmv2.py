import numpy as np

# np.seterr(divide='ignore', invalid='ignore')

def loadDataSet():
    dataset1 = np.loadtxt("Train1.csv", dtype=np.float, delimiter=",")
    dataset2 = np.loadtxt("Train2.csv", dtype=np.float, delimiter=",")
    return dataset1, dataset2

def scaleDataSet(dataset):
    traindata = dataset
    max = traindata.max()
    min = traindata.min()
    traindata = (traindata - min) / (max-min)
    return traindata

def initPara(data, k):
    dim = np.shape(data)[1]
    max = data.max()
    min = data.min()
    alpha = np.ones(k)
    mu = np.random.uniform(min, max, (k, dim))
    sigma = np.ones([k, dim, dim])
    for i in range(k):
        alpha[i] = 1.0 / k
        # mu = np.random.uniform(min,max,(k,dim))
        sigma[i,:,:] = np.identity(dim)
    return alpha, mu, sigma

def gaussPDF(data, mu, sigma):
    dim = np.shape(sigma)[1]
    coefficient = 1.0 / (np.power(2*np.pi, dim / 2.0) * np.linalg.det(sigma+np.eye(dim)*0.01))
    pdf = coefficient * np.exp(-0.5 * np.sum(np.dot((data-mu), np.linalg.inv(sigma+np.eye(dim)*0.01)) * (data-mu), axis=1))

    return pdf

def expectation(data, alpha, mu, sigma):
    num = np.shape(data)[0]
    k = np.shape(alpha)[0]
    gamma = np.zeros((num, k))
    for i in range(k):
        gamma[:, i] = gaussPDF(data, mu[i], sigma[i])
    gamma = gamma * alpha
    for i in range(num):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

def maximization(data, gamma):
    num, dim = np.shape(data)
    k = np.shape(gamma)[1]
    alpha = np.zeros(k)
    mu = np.zeros((k, dim))
    sigma = np.zeros([k, dim, dim])
    for i in range(k):
        alpha[i] = np.sum(gamma[:,i]) / num
        mu[i,:] = np.sum(gamma[:,i].reshape(num,1)*data, axis=0) / np.sum(gamma[:,i])
        cov = 0
        for j in range(num):
            cov += gamma[j, i] * np.dot((data[j] - mu[i]).reshape(dim,1),  (data[j] - mu[i]).reshape(1,dim)) / np.sum(gamma[:,i])
        sigma[i,:,:] = cov

    return alpha, mu, sigma

def gmmParaEstimation(dataset, k):
    traindata = dataset
    alpha, mu, sigma = initPara(traindata, k)
    while 1:
        gamma = expectation(traindata, alpha, mu, sigma)
        new_alpha, new_mu, new_sigma = maximization(traindata, gamma)
        if (abs(sigma-new_sigma)<0.0000001).all():
            break
        else:
            # print '******************'
            # print sigma
            # print new_sigma
            alpha = new_alpha.copy()
            mu = new_mu.copy()
            sigma = new_sigma.copy()
    # print '*************final***************'
    # print new_alpha
    # print new_mu #* (max - min)
    # print new_sigma
    return new_alpha, new_mu, new_sigma

def gmmClassifier(test, alpha_list, mu_list, sigma_list):
    num, dim = np.shape(test)
    k = len(alpha_list[0])
    prob = []
    for j in range(len(alpha_list)):
        alpha, mu, sigma = alpha_list[j], mu_list[j], sigma_list[j]
        gamma = np.zeros((num, k))
        for i in range(k):
            gamma[:, i] = gaussPDF(test, mu[i], sigma[i])
        gamma = gamma * alpha
        prob.append(np.sum(gamma,axis=1))
    parray = np.array(prob)
    category = parray.argmax(axis=0).flatten().tolist()
    summary = [0]*len(alpha_list)
    for i in range(len(category)):
        for kind in range(len(alpha_list)):
            if category[i] == kind:
                summary[kind] += 1
    print(summary)



if __name__ == '__main__':
    traindata = np.loadtxt("TrainSamples.csv", dtype=np.float, delimiter=",")
    trainlabel = np.loadtxt("TrainLabels.csv", dtype=np.float, delimiter=",")
    group = []
    for id in range(10):
        group_index = np.where(trainlabel == id)
        group.append(list(group_index[0]))
    testdata = np.loadtxt("TestSamples.csv", dtype=np.float, delimiter=",")
    testlabel = np.loadtxt("TestLabels.csv", dtype=np.float, delimiter=",")
    testgroup = []
    for id in range(10):
        group_index = np.where(testlabel == id)
        testgroup.append(list(group_index[0]))

    for k in range(4,5):# k is gauss number
        print ('Gauss num is' + str(k))
        alpha_list = []
        mu_list = []
        sigma_list = []
        for i in range(10):
            print ('Group '+str(i))
            traingroup = traindata[group[i],]
            traingroup = scaleDataSet(traingroup)
            alpha, mu, sigma = gmmParaEstimation(traingroup, k)
            alpha_list.append(alpha)
            mu_list.append(mu)
            sigma_list.append(sigma)


        for i in range(10):
            print (len(testgroup[i]))
            testdatagroup = testdata[testgroup[i],]
            print ('test ' +str(i))
            testdatagroup = scaleDataSet(testdatagroup)
            print (len(testdatagroup))
            gmmClassifier(testdatagroup, alpha_list, mu_list, sigma_list)



    # dataset1, dataset2 = loadDataSet()
    #
    # alpha_list = []
    # mu_list = []
    # sigma_list = []
    #
    # alpha1, mu1, sigma1 = gmmParaEstimation(dataset1, 1)
    # alpha2, mu2, sigma2 = gmmParaEstimation(dataset2, 1)
    # alpha_list.append(alpha1)
    # alpha_list.append(alpha2)
    # mu_list.append(mu1)
    # mu_list.append(mu2)
    # sigma_list.append(sigma1)
    # sigma_list.append(sigma2)
    #
    # test1 = np.loadtxt("Test1.csv", dtype=np.float, delimiter=",")
    # test2 = np.loadtxt("Test2.csv", dtype=np.float, delimiter=",")
    # gmmClassifier(test2, alpha_list, mu_list, sigma_list)