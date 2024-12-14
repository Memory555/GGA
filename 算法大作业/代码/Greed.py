import pandas as pd
import numpy as np
import math
# import time


def Inital_pop(train_v):
    # dataframe = pd.read_csv("TSP.csv", sep=",", header=None)
    # v = dataframe.iloc[:, 1:3]#去除第一列12345678910,只保留x,y
    # train_v = 100 * np.random.rand(20, 2)#随机产生20个城市

    # print(v)

    # train_v= np.array(v)
    train_d=train_v
    #初始化距离 为10*10的全0矩阵
    dist = np.zeros((train_v.shape[0],train_d.shape[0]))
    #print(dist.shape)#(10,10)

    #计算距离矩阵
    for i in range(train_v.shape[0]):
        for j in range(train_d.shape[0]):
            dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
    print(dist)

    # train_v= np.array(v)
    train_d=train_v
    #初始化距离 为10*10的全0矩阵
    dist = np.zeros((train_v.shape[0],train_d.shape[0]))
    #print(dist.shape)#(10,10)

    #计算距离矩阵
    for i in range(train_v.shape[0]):
        for j in range(train_d.shape[0]):
            dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
    print(dist)


    """
    s:已经遍历过的城市
    dist：城市间距离矩阵
    sumpath:目前的最小路径总长度
    Dtemp：当前最小距离
    flag：访问标记
    """

    i=1
    n=train_v.shape[0]#城市个数
    j=0
    sumpath=0#目前的最小路径总长度
    s=[]#已经遍历过的城市
    s.append(0)#从城市0开始
    # start = time.clock()
    while True:
        k=1#从1开始,因为人在城市0，所以我们设定先从城市1开始选择
        Detemp=float('inf')#当前最小距离
        while True:
            flag=0#访问标记，否0
            if k in s:#是否访问，如果访问过，flag设为1
                flag = 1
            if (flag==0) and (dist[k][s[i-1]] < Detemp):#如果未访问过，并且距离小于最小距离
                j = k;
                Detemp=dist[k][s[i - 1]];#当前两做城市相邻距离
            k+=1#遍历下一城市
            if k>=n:
                break;
        s.append(j)
        i+=1;
        sumpath+=Detemp
        if i>=n:
            break;

    sumpath+=dist[0][j]#加上dist[0][j] 表示最后又回到起点
    # end = time.clock()
    # print("结果：")
    # print(sumpath)
    # for m in range(n):
    #     print("%s-> "%(s[m]),end='')
    # print()
    # print("程序的运行时间是：%s"%(end-start))
    return s
