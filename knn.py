import random
import math
import operator

def getdataset():
    with open(PASTE PATH TO YOUR DATASET) as f:
        data=f.read()
        data=data.split()
        data1=[]
        data2=[]
        for i in  range(len(data)):
            if i%2==0:
                data[i]=float(data[i])
                data1.append(float(data[i]))

    trainset=[]
    testset=[]
    for i in range(len(data1)):
        if random.random()<0.9:
            trainset.append(data1[i])
        else:
            testset.append(data1[i])

    return testset,trainset,data
def getneighbours(trainset,testinstance):
    distance=[]
    for i in range(len(trainset)):
        dist=euclideandistance(trainset[i],testinstance)
        distance.append((trainset[i],dist))
    distance.sort(key=operator.itemgetter(1))

    neighbours=[]
    for i in range(5):
        neighbours.append(distance[i])

    return neighbours
def getmyclass(neighbours,data):

    for j in range(len(data)):
        if data[j]==neighbours[0][0]:
                print(data[j],data[j+1])

def euclideandistance(x,y):
    distance=x-y
    if distance<0:
        distance=distance*-1
    return distance
def main():
    testset,trainset,data=getdataset()
    for i in range(len(testset)):
        neighbours=getneighbours(trainset,testset[i])
        getmyclass(neighbours,data)
        print('\nTEST INSTANCE IS '+ str(testset[i])+'\n\n')
        for j in range(len(data)):
            if data[j]==testset[i]:
                print('CLASS IS '+data[j+1]+'\n\n' )
main()




