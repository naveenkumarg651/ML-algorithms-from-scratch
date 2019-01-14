import numpy as np
import re
np.random.seed(0)
def dataset():
    f=open('C:/Users/Naveen Kumar/Desktop/movie.txt')
    data=f.readlines()
    data=data[:len(data)-5]
    trainx=[]
    testx=[]
    word_list=[]
    trainy=[]
    testy=[]
    for i in data:
        p=re.split(r'\W*',i)[:-1]
        if np.random.random()>0.4:
            trainy.append(float(p.pop()))
            trainx.append(p)
            word_list.extend(p)
        else:
            testy.append(float(p.pop()))
            testx.append(p)
            word_list.extend(p)
    return trainx,trainy,testx,testy,word_list

def main():
    x,y,testx,testy,word_list=dataset()
    word_list=list(set(word_list))
    x.extend(testx)
    y.extend(testy)

    trainx_vector=np.array(word_vector(x,word_list))
    y=np.mat(y).T
    syn1,syn2,b1,b2=neural_net(trainx_vector,y)
    testx_vector=np.array(word_vector(testx,word_list))
    count=0
    for i in range(len(testx_vector)):
        x=nonlin(nonlin(testx_vector[i].dot(syn1)+b1[0]).dot(syn2)+b2[0])
        if x>0.5:
            if y[i]==1:
                count+=1
        elif x<0.5:
            if y[i]==0:
                count+=1
    print((count/len(testx_vector))*100)
    return
def word_vector(train,word_list):
    vector=[]
    for i in train:
        v=[0]*len(word_list)
        for k in i :
            try:
                index=word_list.index(k)
                v[index]=1
            except Exception as e:
                ' '
        vector.append(v)
    return vector
def neural_net(x,y):

    y=np.array(y)
    x=np.array(x)

    syn1=np.random.randn(184,5)
    syn2=np.random.randn(5,1)
    b1=np.ones((84,5))
    b2=np.ones((84,1))
    alpha=0.1
    for i in range(60000):
        #forward propogation
        l0=x
        l1=nonlin(np.dot(l0,syn1)+b1)
        l2=nonlin(np.dot(l1,syn2)+b2)
        l2_error=l2-y
        if i%10000==0:
            print(np.abs(np.mean(l2_error)))
        l2_delta=l2_error*nonlin(l2,True)
        dw2=l1.T.dot(l2_delta)
        db2=np.sum(l2_delta,axis=0)

        l1_error=l2_delta.dot(syn2.T)
        l1_delta=l1_error*nonlin(l1,True)
        dw1=l0.T.dot(l1_delta)
        db1=np.sum(l1_delta,axis=0)

        syn1-=alpha*dw1
        b1-=alpha*db1
        syn2-=alpha*dw2
        b2-=alpha*db2
    return syn1,syn2,b1,b2
def nonlin(x,deriv=False):
    if deriv==True:
        return x*(1-x)
    return (1/(1+np.power(np.e,(-x))))




main()