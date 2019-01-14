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
    
    x=np.mat(x)
    y=np.mat(y).T
    y1=np.array(y)
    xtx=x.T.dot(x)
    fig=plt.figure()
    ax=Axes3D(fig)
    xty=x.T.dot(y)
    q=xtx.I.dot(xty)
    yplot=[]
    sum1=0
    for i in range(len(y)):
        print(y[i],'   ',x[i].dot(q))
        yplot.append(x[i].dot(q))
        sum1+=(y[i]-(x[i].dot(q)))**2
    print(np.sqrt(sum1/len(y)))
    ax.scatter(x1,x2,yplot)
    ax.scatter(x1,x2,y1)
        
linear()