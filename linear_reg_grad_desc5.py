def linear():
    import numpy as np,re
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    f=open('G:/mypython/datasets/reg_fishlen.txt')
    length=len(f.readlines())
    f.close()
    f=open('G:/mypython/datasets/reg_fishlen.txt')
    x=[];x1=[];x2=[];y=[]
    for i in range(length):
        line=f.readline()
        if i>=4:
            line=[float(i) for i in re.findall(r'[\d\.\d]+',line)]
            line[0]=1
            x.append(line[:-1])
            x1.append(line[1])
            x2.append(line[2])
            y.append(line[-1])
    '''fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(x1,x2,y)'''
    x=np.array(x)
    y=np.array(y)
    weights=np.array([0,0,0])
    intialcost=cost(x,y,weights)
    weights=grad(x,y,weights,intialcost)
    yplot=[]
    for i in range(len(y)):
        print(y[i],'   ',x[i].dot(weights))
        yplot.append(x[i].dot(weights))
    sum1=0
    for i in range(len(y)):
        sum1+=(y[i]-yplot[i])**2
    print(np.sqrt(sum1/len(y)))
def cost(x,y,b):
    import numpy as np
    return np.sum((x.dot(b)-y)**2)/(2*len(y))
def grad(x,y,b,ic):
    import numpy as np
    oldcost=ic
    costchange=2
    for i in range(1000000):
        hyp=x.dot(b)
        error=hyp-y
        print('error')
        gradient=x.T.dot(error)/len(y)
        b=b-(0.0002*gradient)
        new=cost(x,y,b)
        costchange=oldcost-new
        oldcost=new
    return b
        
linear()
            
    