def linear():
    import numpy as np,re
    f=open('G:/mypython/datasets/reg_fat.txt')
    length=len(f.readlines())
    f.close()
    f=open('G:/mypython/datasets/reg_fat.txt')
    dataset=[]
    target=[]
    for i in range(length):
        line=f.readline()
        line=re.split(r'\W*',line)
        line.pop()
        line=[float(example) for example in line]
        target.append(line.pop(len(line)-1))
        dataset.append(line[1:])
    dataset=np.mat(dataset)
    target1=np.mat(target)
    
    xtx=dataset.T*dataset
    xty=dataset.T*target1.T
    weights=xtx.I*xty
   
    y=dataset[0]*weights
    sum1=0
    for i in range(len(dataset)):
        x=dataset[i]*weights
        sum1+=np.abs(x-target[i])
    import matplotlib.pyplot as plt
    print(dataset[:5,2],target[:5])
    plt.scatter(dataset[:,2].flatten().A[0],target[:])
    plt.plot(dataset[:,2].T.flatten().A[0],(dataset*weights).T.flatten().A[0])
    
        
    
    

linear()
        