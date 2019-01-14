def linear():
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    f=open('G:/mypython/datasets/reg_agebp.txt')
    length=len(f.readlines())
    f.close()
    x=[]
    y=[]
    f=open('G:/mypython/datasets/reg_agebp.txt')
    for i in range(length):
        line=re.split(r'\W*',f.readline())
        if i >=4:
            
            
            x.append(float(line[2]))
            y.append(float(line[3]))
    plt.scatter(x,y)
    plt.xlabel('age')
    plt.ylabel('blood pressure')
    x_mean,y_mean=np.mean(x),np.mean(y)
    num=0;denom=0
    for i in range(len(x)):
        num+=(x[i]-x_mean)*(y[i]-y_mean)
        denom+=(x[i]-x_mean)**2
    b1=num/denom
    b0=y_mean-b1*x_mean
    predict=[]
    for i in range(len(x)):
        print('predicted=  ',b0+b1*x[i],'  actual =',y[i])
        predict.append(b0+b1*x[i])
    plt.plot(x,predict,color='black')
    plt.scatter(x_mean,y_mean,color='blue')
    sum1=0
    for i in range(len(y)):
        sum1+=(y[i]-predict[i])**2
    print(np.sqrt(sum1/len(y)))
        
    
    
    
linear()
