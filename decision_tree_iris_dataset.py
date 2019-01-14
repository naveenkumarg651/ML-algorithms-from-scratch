import numpy as np
import csv,random
random.seed(0.9)
def dataset():
    with open('G:/mypython/datasets/iris.txt') as f:
        lines=list(csv.reader(f))
        data=[]
        for i in lines:
            if not i:
                continue
            r=i
            k=[float(j) for j in i[:-1]]
            k.append(r[-1])
            data.append(k)
        data.pop()
        train=[]
        test=[]
        for i in data:
            if random.random()>0.3:
                train.append(i)
            else:
                test.append(i)
        return np.array(train),np.array(test)
def main():
    train,test=dataset()
    prune=test[:20,:]
    test=test[20:,:]
    tree=create(train,1)
    print(tree)
    tree=pruning(tree,test)
    print(tree)
    test_func(tree,prune)
def create(data,toln):
    feat,val=bestsplit(data,toln)
    if feat==None:return val
    retree={}
    retree['index']=feat
    retree['value']=val
    mat0,mat1=binsplit(data,feat,val)
    retree['left']=create(mat0,toln)
    retree['right']=create(mat1,toln)
    return retree
def bestsplit(data,toln):
    classes=set([i[-1] for i in data])
    s=gini([data[:int(len(data)/2),:],data[int((len(data)/2))+1:,:]],classes)
    bests=9999;bestindex=-1;bestvalue=-1
    for i in  range(len(data[0])-1):
        for j in data[:,i]:
            mat0,mat1=binsplit(data,i,j)
            if np.shape(mat0)[0]<toln or np.shape(mat1)[0]<toln:
                continue
            error=gini([mat0,mat1],classes)
            if error<bests:
                bestindex=i
                bestvalue=j
                bests=error
    '''if s-bests<0.5:
        return None,term(data)'''
    if np.shape(mat0)[0]<toln or np.shape(mat1)[0]<toln:
        return None,term(data)
    return bestindex,bestvalue
def gini(groups,classes):
    n=sum([len(i) for i in groups])
    gini=0
    for g in groups:
        size=len(g)
        score=0
        if size==0:continue
        for c in classes:
            p=[i[-1] for i in g].count(c)
            score+=p*p
        gini+=(1-score)*(size/n)
    return gini
def term(data):
    i=[i[-1] for i in data]
    return max(set(i),key=i.count)
def binsplit(data,feat,val):
    mat0=data[np.nonzero(data[:,feat]>val)[0],:]
    mat1=data[np.nonzero(data[:,feat]<=val)[0],:]
    return mat0,mat1
def test_func(tree,test):
    yhat=np.zeros((len(test),1))
    count=0
    for i in test:
        yhat=forecast(tree,i)
        if yhat==i[-1]:
            count+=1
    print((count/len(test))*100)
def istree(obj):
    return type(obj).__name__=='dict'
def forecast(tree,instance):


    if instance[tree['index']]>tree['value']:
        if istree(tree['left']):
            return forecast(tree['left'],instance)
        else:
            return tree['left']
    else:
        if istree(tree['right']):
            return forecast(tree['right'],instance)
        else:
            return tree['right']
def pruning(tree,test):
    if np.shape(test)[0]==0:return getmean(tree)
    if istree(tree['left']) or istree(tree['right']):
        mat0,mat1=binsplit(test,tree['index'],tree['value'])
    if istree(tree['left']):tree['left']=pruning(tree['left'],test)
    if istree(tree['right']):tree['right']=pruning(tree['right'],test)
    if not istree(tree['right']) and not istree(tree['left']):
        if tree['right']==term(test) and tree['left']==term(test):
            print('merging1')
            return tree['right']
        elif tree['right']==term(test) and tree['left']!=term(test):
            print('merging2')
            return tree['right']
        elif tree['left']!=term(test) and tree['left']==term(test):
            print('merging3')
            return tree['left']
    else:
        return tree
    return tree
def getmean(tree):
    print('in get mean')
    d=[]
    if istree(tree['right']):d.append(getmean(tree['right']))
    if istree(tree['left']):d.append(getmean(tree['left']))
    return max(set(d),d.count)
main()





