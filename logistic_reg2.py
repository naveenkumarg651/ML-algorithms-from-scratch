import numpy as np,re
def logistic():
    f=open('G:/mypython/datasets/play.txt')
    length=len(f.readlines())
    f.close()
    f=open('G:/mypython/datasets/play.txt')
    x=[];y=[]
    for i in range(length):
        line=re.split(r'\W*',f.readline())
        line.pop()
        if line.pop()=='yes':
            y.append(1)
        else:
            y.append(0)
        x.append(line)
    x.pop()
    values=[]
   
    for i in x:
        values.extend(i)
    values=set(values)
    values=list(values)
    trainmat=[]
    for i in x:
        i=wordvec(i,values)
        i.insert(0,1)
        trainmat.append(i)
    y.pop()
    print(trainmat)        
    y=np.array(y)
    trainmat=np.array(trainmat)
    weights=np.array(np.zeros(len(trainmat[0])))
    print(weights.shape,trainmat.shape,y.shape)
    intialcost=cost(trainmat,y,weights)
    weights=grad(trainmat,y,weights,intialcost)
    yplot=[]
    for i in  range(len(y)):
        if hyp(trainmat[i],weights)>0.5:
            print(y[i],'  ',1)
            yplot.append(1)
        else:
            print(y[i],'  ',0)
            yplot.append(0)
            sum1=0
    for i in range(len(y)):
        sum1+=(y[i]-yplot[i])**2
    print(np.sqrt(sum1/len(y)))
    print('enter the test case')
    x=input()
    x=re.split(r'\W*',x)
    x=wordvec(x,values)
    x.insert(0,1)
    x=np.array(x)
    print(hyp(x,weights))
    
        
    
def hyp(x,weights):
    return 1/(1+np.power(np.e,-1*(x.dot(weights))))
def cost(x,y,b):
    return np.sum((-1*y)*np.log(hyp(x,b))-((1-y)*np.log(1-hyp(x,b))))/len(y)
def grad(x,y,b,ic):
    old=ic
    costchange=2
    while(costchange>0.001):
        hy=hyp(x,b)
        error=hy-y
        gradient=x.T.dot(error)/len(y)
        b=b-(1*gradient)
        new=cost(x,y,b)
        costchange=old-new
        old=new
    return b
        
        
        

def wordvec(line,values):
    zeros=[0]*len(values)
    
    for i in line:
        zeros[values.index(i)]+=1
    return zeros
logistic()
        
        
    
        