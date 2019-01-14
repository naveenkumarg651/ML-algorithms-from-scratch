#output tree would be in the form of dictionary

import numpy as np
import csv
import re
def dataset():
    f=open('G:/mypython/datasets/reg_fat.txt')
    lines=csv.reader(f)
    lines=list(lines)
    data=[]
    for i in lines:
        data.append(re.split(r'\W*',i[0]))
    dataset1=[]
    for i in data:
        i=[float(k) for k in i[2:]]
        dataset1.append(i)
    dataset1=np.array(dataset1)
    x=dataset1[0:18,:]
    cross=dataset1[18:20,:]
    test=dataset1[20:,:]
    return x,cross,test
def main():
    train,test,inputs=dataset()
    tree=create_tree(train,(1,1))
    print(tree,'\n')
    tree=pruning(tree,test)
    print(tree)
    treeforecast(tree,inputs)

def create_tree(data,ops):
    feat,val=choosebestsplit(data,(2,2))
    if feat==None:return val
    retree={}
    retree['spind']=feat
    retree['spval']=val
    lset,rset=binsplit(data,feat,val)
    retree['left']=create_tree(lset,(1,4))
    retree['right']=create_tree(rset,(1,4))
    return retree
def leaf(data):
    return np.mean(data[:,-1])
def error(data):
    return np.var(data[:,-1])*np.shape(data)[0]
def binsplit(data,feat,val):
    mat0=data[np.nonzero(data[:,feat]>val)[0],:]
    mat1=data[np.nonzero(data[:,feat]<=val)[0],:]
    return mat0,mat1
def choosebestsplit(data,ops):
    tols=ops[0];toln=ops[1]
    if len(set(data[:,-1]))==1:
        return None,leaf(data)
    m,n=np.shape(data)
    s=error(data)
    bests=9999999;bestindex=-1;bestvalue=0;
    for i in range(n-1):
        for j in data[:,i]:
            mat0,mat1=binsplit(data,i,j)
            if np.shape(mat0)[0]<toln or np.shape(mat1)[0]<toln:continue
            news=error(mat0)+error(mat1)
            if news<bests:
                bests=news
                bestindex=i
                bestvalue=j
    mat0,mat1=binsplit(data,bestindex,bestvalue)
    if np.shape(mat0)[0]<toln or np.shape(mat1)[0]<toln:
        return None,leaf(data)
    if (s-bests)<tols:
        return None,leaf(data)
    return bestindex,bestvalue
def pruning(tree,test):
    if np.shape(test)[0]==0: return getmean(tree)
    if istree(tree['left']) or istree(tree['right']):
        mat0,mat1=binsplit(test,tree['spind'],tree['spval'])
    if istree(tree['left']):tree['left']=pruning(tree['left'],mat0)
    if istree(tree['right']):tree['right']=pruning(tree['right'],mat1)
    if not istree(tree['right']) and not istree(tree['left']):
        mat0,mat1=binsplit(test,tree['spind'],tree['spval'])
        errornomerge=np.sum(np.power((mat0[:-1]-tree['left']),2))+np.sum(np.power((mat1[:-1]-tree['right']),2))
        treemean=(tree['left']+tree['right'])/2
        errormerge=np.sum(np.power((test[-1]-treemean),2))
        if errormerge<errornomerge:
            print('merging')
            treemean
            return treemean
        else:
            print('in else')
            return tree
    else:
        return tree
    return tree
def getmean(tree):
    if istree(tree['left']):tree['left']=getmean(tree['left'])
    if istree(tree['right']):tree['right']=getmean(tree['right'])
    return (tree['left']+tree['right'])/2
def istree(tree):
    if type(tree).__name__=='dict':
        return True
    else:
        return False
def regtree(tree):
    return float(tree)
def treeforecast(tree,inputs):
    yhat=np.zeros((len(inputs),1))
    x=0
    for i in range(len(inputs)):
        yhat[i,0]=getoutput(tree,list(inputs[i]))
        x+=np.abs(yhat[i,0]-inputs[i,-1])
    print(x)
def getoutput(tree,inputs):
    if not istree(tree):return float(tree)
    if inputs[tree['spind']]>tree['spval']:
        if istree(tree['left']):
            return getoutput(tree['left'],inputs)
        else:
            return regtree(tree['left'])
    else:
        if istree(tree['right']):
            return getoutput(tree['right'],inputs)
        else:
            return regtree(tree['right'])

main()





