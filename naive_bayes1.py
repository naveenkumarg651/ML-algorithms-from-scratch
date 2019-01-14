#email spam -naive bayes classifier
import re
import numpy as np
def vocablist(dataset):
    vocablist=set([])
    for i in dataset:
        vocablist=vocablist|set(i)
    return vocablist
def loaddataset():
    import re
    f=open('play.txt')
    mainlist=[]
    classvec=[]
    x=f.readlines()
    x=len(x)-1
    f.close()
    f=open('play.txt')

    for i in range(x):
         x=f.readline()
         data=re.split(r'\W*',x)
         if data[len(data)-2]=='yes':
             classvec.append(1)
         elif data[len(data)-2]=='no':
             classvec.append(0)
         data=data[0:len(data)-2]
         mainlist.append(data)
    print(mainlist)
    return mainlist,classvec
def main():

    dataset,classvec=loaddataset()
    vocablist1=vocablist(dataset)
    vocablist1=list(vocablist1)
    trainmatrix=[]
    for i in dataset:
        x=linevect(vocablist1,i)
        trainmatrix.append(x)
    p_of_bad,pb1,pb0=train(trainmatrix,classvec)
    x=input('enter your test condition')
    x=re.split(r'\W*',x)
    print(x)
    test=linevect(vocablist1,x)
    classify(test,pb1,pb0,p_of_bad)

def linevect(vocablist,line):
    vector=[0]*len(vocablist)
    for i in line:
        if i in vocablist:
            vector[vocablist.index(i)]+=1
    return vector
def train(trainmatrix,classvec):
    numofdocs=len(trainmatrix)
    numofwords=len(trainmatrix[0])
    p_of_bad=sum(classvec)/len(classvec)
    p1num=np.ones(numofwords);p1denom=2
    p0num=np.ones(numofwords);p0denom=2
    for i in range(numofdocs):
        if classvec[i]==1:
            p1num+=trainmatrix[i]
            p1denom+=sum(trainmatrix[i])
        else:
            p0num+=trainmatrix[i]
            p0denom+=sum(trainmatrix[i])
    pb1=np.log(p1num/p1denom);pb0=np.log(p0num/p0denom)
    return p_of_bad,pb1,pb0
def classify(test,p1,p0,total_pb):
    pb=sum(test*p1)+np.log(total_pb)
    pg=sum(test*p0)+np.log(1-total_pb)
    if pb>pg:
        print('they can play')
    else:
        print('they cannot play')
main()

