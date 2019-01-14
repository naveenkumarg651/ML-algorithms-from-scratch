import numpy as np,csv,matplotlib.pyplot as plt
def svm():
    f=open('G:/mypython/cancer.csv')
    lines=csv.reader(f)
    lines=list(lines)
    del(lines[0])
    x=[];y=[];g1=[];g2=[]
    for i in lines:
        x1=[]
        for j in range(len(i)):
            x1.append(float(i[j]))


        if x1.pop()==1:
            y.append(1)
            g1.append(x1)
        else:
            y.append(-1)
            g2.append(x1)
        x1.insert(0,1)
        x.append(x1)

    trainx=[];trainy=[]

    for i in range(400):
        trainx.append(x[i])

        trainy.append(y[i])
    testx=[];testy=[]
    for i in range(400,len(x)):
        testx.append(x[i])
        testy.append(y[i])

    trainx=np.array(trainx)
    trainy=np.array(trainy)
    testx=np.array(testx)
    testy=np.array(testy)

    g1=np.array(g1)
    g2=np.array(g2)
    b=np.zeros(len(x[0]))
    ic,alpha,r=cost(trainx,trainy,b)
    b=grad(trainx,trainy,b,ic,alpha,r)
    plot=[]
    for i in range(len(testx)):
        if testx[i].dot(b)>=1:
            print(1,testy[i])
            plot.append(1)
        elif testx[i].dot(b)<=1:
            print(-1,testy[i])
            plot.append(-1)
    sum1=0
    for i in range(len(testx)):
        sum1+=(testy[i]-plot[i])**2
    print((np.sqrt(sum1/len(testy)))*100)
    plt.scatter(g1[:,3],g1[:,2],marker='_')
    plt.scatter(g2[:,3],g2[:,2],marker='*')
    x1=[b[3],b[4],-b[4],b[3]]
    x2=[b[3],b[4],b[4],-b[3]]
    x2x3=np.array([x1,x2])

    x,y,u,v=zip(*x2x3)
    ax=plt.gca()
    ax.quiver(x,y,u,v,scale=1,color='red')

def cost(x,y,b):
    alpha=100000000
    lr=0.0001
    p=((b.dot(b)))+alpha*np.sum(1-np.abs((y*(x.dot(b)-1))))/len(x)
    return p,alpha,lr

def grad(x,y,b,ic,alpha,lr):
    costchange=2
    old=ic

    while(costchange>0.0001):
        for i in range(len(x)):
            if y[i]*(x[i].dot(b)-1)>=1:
                b=b-lr*(b)
            else:
                b=b-lr*((b)-(alpha*y[i]*x[i]))

        new,alpa,lr=cost(x,y,b)

        costchange=old-new
        print(costchange)
        old=new
    return b

svm()