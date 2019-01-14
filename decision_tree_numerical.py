import numpy as np
import csv
#pruning is not woking for this  model
def dataset():
    f=open('G:/mypython/cancer.csv')
    lines=(list(csv.reader(f)))
    p=[]
    k=0
    for i in lines:
        if k==0:
            k+=1
            continue
        for j in i:
            j=[float(k) for k in i]
        p.append(j)
    p=np.array(p)
    train=p[:300,:]
    test=p[300:400,:]
    cross=p[406:500,:]
    return train,cross,test
def main():
    train,cross,test=dataset()
    tree=create(train,1)
    forecast(tree,test)
    tree=pruning(tree,cross)
    forecast(tree,test)
    return
def create(data,ops):
    feat,val=bestsplit(data,ops)
    if feat==None:return val
    retree={}
    retree['index']=feat
    retree['value']=val
    mat0,mat1=binsplit(data,feat,val)
    retree['left']=create(mat0,ops)
    retree['right']=create(mat1,ops)
    return retree
def bestsplit(data,ops):
    classes=set([i[-1] for i in  data])
    bests=99999;bestindex=-1;bestvalue=-1
    for i in range(len(data[0])-1):
        for j in data[:,i]:

            mat0,mat1=binsplit(data,i,j)

            if np.shape(mat0)[0]<ops or np.shape(mat1)[1]<ops:
                continue
            error=gini([mat0,mat1],classes)
            if error<bests:
                bests=error
                bestindex=i
                bestvalue=j
    mat0,mat1=binsplit(data,bestindex,bestvalue)
    if np.shape(mat0)[0]<ops or np.shape(mat1)[0]<ops:
        return None,term(data)
    return bestindex,bestvalue
def term(data):
    i=[i[-1] for i in data]
    return max(set(i),key=i.count)
def gini(groups,classes):
    n=sum([len(s) for s in groups])
    gini=0
    for g in groups:
        size=len(g)
        if size==0:
            continue
        score=0
        for c in classes:
            p=[i[-1] for i in g].count(c)
            score+=p*p
        gini+=(1-score)*size/n
    return gini
def binsplit(data,feat,val):
    mat0=data[np.nonzero(data[:,feat]>val)[0],:]
    mat1=data[np.nonzero(data[:,feat]<=val)[0],:]
    return mat0,mat1
def forecast(tree,test):
    count=0
    yhat=[0]*len(test)
    for i in range(len(test)):
        yhat[i]=testing(tree,test[i])
        if yhat[i]==test[i][-1]:
            count+=1
    print((count/len(test))*100)
def istree(data):
    return type(data).__name__=='dict'
def testing(tree,test):

    if test[tree['index']]>tree['value']:
        if istree(tree['left']):
            return testing(tree['left'],test)
        else:
            return tree['left']
    else:
        if istree(tree['right']):
            return testing(tree['right'],test)
        else:
            return tree['right']
def pruning(tree,test):
    if len(test)==0: return
    if istree(tree['left']) or istree(tree['right']):
        mat0,mat1=binsplit(test,tree['index'],tree['value'])
    if istree(tree['left']):tree['left']=pruning(tree['left'],mat0)
    if istree(tree['right']):tree['right']=pruning(tree['right'],mat1)
    if not istree(tree['left']) and not istree(tree['right']):
        obj=term(test)
        if tree['left']==obj and tree['right']!=obj:
            tree['right']=obj

        elif tree['right']==obj and tree['left']!=obj:
            tree['left']=obj

        elif tree['right']!=obj and tree['left']!=obj:
            tree['left']=tree['right']=obj

    else:
        return tree
    return tree
main()

dataset()