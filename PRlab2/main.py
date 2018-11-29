import matplotlib.pyplot as plt
from gmm import *
import csv
# 设置调试模式
DEBUG = True

def test1():
    # 载入数据1
    Y=[]
    csv_reader=csv.reader((open("Train1.csv")))
    for row in csv_reader:
        Y.append(row)
    matY = np.matrix(Y, dtype=float,copy=True)
    Y=np.array(Y,dtype=float)

    # 模型个数，即聚类的类别个数
    K = 2

    # 计算 GMM1 模型参数
    mu, cov, alpha = GMM_EM(matY, K, 500)



    # 载入数据2

    Y2=[]
    csv_reader=csv.reader((open("Train2.csv")))
    for row2 in csv_reader:
        Y2.append(row2)
    matY2 = np.matrix(Y2, dtype=float,copy=True)
    Y2=np.array(Y2,dtype=float)

    # 模型个数，即聚类的类别个数
    K = 2

    # 计算 GMM2 模型参数
    mu2, cov2, alpha2 = GMM_EM(matY2, K, 500)



    # 模拟测试1
    test=[]
    csv_reader=csv.reader((open("Test2.csv")))
    for row in csv_reader:
        test.append(row)
    mattest = np.matrix(test, dtype=float,copy=True)
    test=np.array(test,dtype=float)

    #对数据进行归一化
    mattest = scale_data(mattest)
    # 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
    N = test.shape[0]
    # 求当前模型参数下，各模型对样本的响应度矩阵
    gamma = getExpectation2(mattest, mu, cov, alpha)
    print('\n','GMM1:','*'*40,'\n')
    print(gamma)
    # 对每个样本，求响应度最大的模型下标，作为其类别标识
    category = gamma.argmax(axis=1).flatten().tolist()[0]


    #print(category)


    # 求GMM2各模型对样本的响应度矩阵
    gamma2 = getExpectation2(mattest, mu2, cov2, alpha2)
    print('\n','GMM2:','*'*40,'\n')
    print(gamma2)
    # 对每个样本，求响应度最大的模型下标，作为其类别标识
    category2 = gamma2.argmax(axis=1).flatten().tolist()[0]
    count1=0
    count2=0
    print("#"*40,'模型概率对比',"#"*40)
    for i in range(N):
        print('(',gamma[i,category[i]],',',gamma2[i,category2[i]],')')
        if(gamma[i,category[i]]>gamma2[i,category2[i]]):
            count1=count1+1
        else:
            count2=count2+1
    print('\n','两个GMM模型分别有：','*'*40,'\n')
    print(count1)
    print(count2)


def test2():
    # 载入数据1
    Y = []
    csv_reader = csv.reader((open("TrainSamples.csv")))
    for row in csv_reader:
        Y.append(row)
    labels=[]
    csv_reader_labels=csv.reader((open("TrainLabels.csv")))
    for label in csv_reader_labels:
        labels.append(label)
    data=list(zip(Y,labels))
    data_arry=[[] for i in range(10)]
    print(data_arry)
    for feature,label in data:
        #print(feature, ',', label)
        data_arry[int(label[0])].append(feature)
    print(data_arry[0][1])
    mu=[0 for i in range(10)];cov=[0 for i in range(10)];alpha=[0 for i in range(10)]
    for i in range(10):
        matY = np.matrix(data_arry[i][:], dtype=float, copy=True)

        # 模型个数，即聚类的类别个数
        K = 2

        # 计算 GMM1 模型参数
        mu[i], cov[i], alpha[i] = GMM_EM(matY, K, 100)


    testY = []
    csv_reader = csv.reader((open("TestSamples.csv")))
    for row in csv_reader:
        testY.append(row)
    testlabels = []
    csv_reader_labels = csv.reader((open("TestLabels.csv")))
    for label in csv_reader_labels:
        testlabels.append(label)
    testdata = list(zip(testY, testlabels))

    gamma = [[] for i in range(10)]
    category = [[] for i in range(10)]
    count = [0 for i in range(10)]

    testmatY=np.matrix(testY,dtype=float)

    for i in range(10):
        gamma[i] = getExpectation2(testmatY, mu[i], cov[i], alpha[i])
        # 对每个样本，求响应度最大的模型下标，作为其类别标识
        category[i] = gamma[i].argmax(axis=1).flatten().tolist()[0]
    print(mu[0],cov[0],alpha[0])
    print('*'*80,'gamma:\n',np.array(gamma[0]).shape)
        #if(np.argmax(gamma[i,category[i]])==testlabels[sample]):count[i]=count[i]+1
    print('*'*100)
    for i in range(10):
        print('第{}类正确分类数:'.format(i),count[i],'\n')

test2()


