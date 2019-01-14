import numpy as np
import matplotlib.pyplot as plt
import random
import csv

random.seed(5)
def dataset():
    f=open('PATH TO YOUR DATASET')
    lines=csv.reader(f)
    lines=list(lines)
    data=[];labels=[]
    for i in lines[0:len(lines)-1]:
        labels.append(i.pop())
        data.append([float(n) for n in i])
    return data,labels
def main():
    data,labels=dataset()
    clusters=[]

    sse=[]
    k=range(1,6)
    for i in  range(1,6):

        centroids,new=kmeans(data,i)
        clusters.append(new)
        sse.append(sqer(new,centroids))
    print(sse)
    plt.plot(k,sse)

def kmeans(data,i):
    old={1:[],2:[],3:[]};new={}
    centroids=centroids1(data,i)

    while(old!=new):
        old=new
        centroids,new=kmeansclustering(data,centroids)

    return centroids,new
def centroids1(data,i):
    c=[]
    for j in range(i):
        x=random.randint(0,len(data)-1)

        if data[x] not in c:
            c.append(data[x])

    return c
def kmeansclustering(data,centroids):
    new={}

    k=0
    while k<len(data):

        distance=[]
        for j in range(len(centroids)):
            distance.append(euclidean(data[k],centroids[j]))

        index=distance.index(min(distance))

        if index not in new.keys():
            new[index]=[]
        new[index].append(data[k])
        k+=1
    mean=[]
    for k in new:
        sorted(new[k])
        mean.append(mean1(new[k]))

    return mean,new
def sqer(new,centroid):
    sum1=0
    for j in range(len(new)):
        for i in range(len(new[j])):
            sum1+=add(new[j][i],centroid[j])
    return sum1
def mean1(k):
    m=[]
    if k==[]:
        k=[0,0,0,0]
    for j in range(len(k[0])):

        m.append(np.mean(np.mat(k)[:,j]))

    return m
def euclidean(x,y):
    sum1=0

    for i in range(len(x)):
        sum1+=(x[i]-y[i])**2
    return np.sqrt(sum1)
def add(x,y):
    sum1=0
    for j in range(len(x)):
        sum1+=(x[j]-y[j])**2
    return sum1


main()