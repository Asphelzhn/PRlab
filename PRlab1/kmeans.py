import matplotlib.pyplot as plt
import csv
from numpy import *
set_printoptions(threshold=inf)
# 加载数据
def loadDataSet(fileName):
    DataSet=[]
    csv_file=csv.reader(open(fileName+'.csv','r'))
    for stu in csv_file:
        DataSet.append(stu)
    return array(DataSet,dtype=float)

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 构建聚簇中心，取前k个作为质心
def randCent2(dataSet, k):
    # 每个质心有n个坐标值，总共要k个质心
    centroids = array(dataSet[0:k,:])
    print(centroids.shape)
    return centroids

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    # 每个质心有n个坐标值，总共要k个质心
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent2):
    m = shape(dataSet)[0]
    # 用于存放该样本属于哪类及质心距离
    clusterAssment = mat(zeros((m,2)))
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    # 用来判断聚类是否已经收敛
    clusterChanged = True
    print('中心点迭代\n')
    while clusterChanged:
        clusterChanged = False;
        # 把每一个数据点划分到离它最近的中心点
        for i in range(m):
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    # 如果第i个数据点到第j个中心点更近，则将i归属为j
                    minDist = distJI; minIndex = j
                    # 如果聚类结果发生变化，则需要继续迭代
            if clusterAssment[i,0] != minIndex: clusterChanged = True;
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i,:] = minIndex,minDist**2
        #print(centroids,'\n')
        for cent in range(k):   # 重新计算中心点
            # 去第一列等于cent的所有列
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent,:] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment



# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
def ClusterTest():
    x=array([[0,0],[1,0],[0,1],[1,1],[2,1],[1,2],[2,2],[3,2]
       ,[6,6],[7,6],[8,6],[7,7],[8,7],[9,7],[7,8],[8,8]
       ,[9,8],[8,9],[9,9]])
    myCentroids,clusAssing=kMeans(x,2)
    print('中心点：\n')
    print(myCentroids)
    print('\n聚类情况:\n')
    print(clusAssing)

    plt.figure()
    x_cent=(myCentroids[0,0])
    y_cent =(myCentroids[0,1])
    plt.scatter(x_cent, y_cent,c='b',s=100,marker='^')

    x_cent2 = array(myCentroids[1, 0])
    y_cent2 = array(myCentroids[1, 1])
    plt.scatter(x_cent2, y_cent2, c='r', s=100, marker='^')

    x_dot = [i[0] for i in x ][0:8]
    y_dot = [i[1] for i in x][0:8]
    plt.scatter(x_dot, y_dot, c='b',)

    x_dot2 = [i[0] for i in x][9:]
    y_dot2 = [i[1] for i in x][9:]
    plt.scatter(x_dot2, y_dot2, c='r', )

    plt.scatter(x_cent,y_cent)
    plt.xticks(linspace(0,10,10))
    plt.yticks(linspace(0,10,10))

    plt.show()



# --------------------测试2----------------------------------------------------
# MNIST数据集测试
def MinistTest():
    DataSet=loadDataSet('ClusterSamples')
    #print(DataSet[0])
    #print(DataSet.shape)
    myCentroids, clusAssing = kMeans(DataSet, 10)
    print('中心点：\n')
    print(myCentroids)
    print('\n聚类情况:\n')
    print(clusAssing)
    Label=loadDataSet('SampleLabels')
    count=zeros(100).reshape(10,10)
    for i in range(10000):
        #print(int(clusAssing[i,0]),'    +   ',int(Label[i]))
        count[int(clusAssing[i,0])][int(Label[i])]+=1
    count=array(count,dtype=int32)
    print(count)


MinistTest()