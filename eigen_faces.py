from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import operator
def dataset():
    img_data=[]
    for i in range(0,41):
        img=imread('G:/mypython/eigen_faces_data/%d.jpg'%i)
        img=rgb2gray(img)
        img_data.append(img)
    return img_data
def main():
    data=dataset()
    vectors=[]
    for i in data:
        vectors.append(np.reshape(i,(1,-1))[0])
    vectors=np.array(vectors)
    mean=np.mean(vectors,axis=0)
    std_data=vectors-mean

    cov_mat=np.cov(std_data)
    eigenvalues,eigenvectors=np.linalg.eig(cov_mat)
    idx=np.argsort(-eigenvalues)
    eigenvalues=eigenvalues[idx]
    eigenvectors=eigenvectors[idx]
    eigenvectors[:,20]
    eigen_faces=project(eigenvectors.T,std_data)
    eigen_weights=[]
    for i in std_data:
        eigen_weights.append(project1(eigen_faces,i.T))
    test_input=input('enter the test input image along with its path\n')
    test=imread(test_input)
    test=rgb2gray(test)
    test=np.reshape(test,(1,-1))
    test=test-(mean)
    test_weights=project1(eigen_faces,test.T)
    name,images=knn(eigen_weights,test_weights)
    print('\n***************************************\n      given image is similar to  ',name)
    for i in images:
        img=vectors[i,:]
        img=np.reshape(img,(250,250))
        plt.imshow(img)
        plt.figure()
    print('\n\nBelow shown images are predicted images')
    return
def project(eigenvectors,vectors):
    return np.dot(eigenvectors,vectors)

def project1(eigenfaces,vector):
    return np.dot(eigenfaces,vector)
def knn(train,test):
    distance=[]
    for i in range(len(train)):
        d=euclidean(train[i],test)
        distance.append((i,d))
    distance=sorted(distance,key=operator.itemgetter(1))

    classcount={'arnold schwarzenegger':0,'britney spears':0,'jackie chan':0,'jennifer lopez':0}
    name_count={'arnold schwarzenegger':[],'britney spears':[],'jackie chan':[],'jennifer lopez':[]}
    for i in range(7):
        p=distance[i][0]

        if distance[i][0]<=40 and distance[i][0]>=30:
            classcount['arnold schwarzenegger']+=1
            name_count['arnold schwarzenegger'].append(p)
        elif distance[i][0]<=29 and distance[i][0]>=20:
            classcount['britney spears']+=1
            name_count['britney spears'].append(p)
        elif distance[i][0]<=19 and distance[i][0]>=10:
            classcount['jackie chan']+=1
            name_count['jackie chan'].append(p)
        elif distance[i][0]<=9 and distance[i][0]>=0:
            classcount['jennifer lopez']+=1
            name_count['jennifer lopez'].append(p)

    print(classcount)
    max1=-999
    for i in classcount.keys():
        if classcount[i]>max1:
            max1=classcount[i]
            name=i
            images=name_count[i]
    return name,images

def euclidean(test,train):
    sum1=0
    for i in range(len(test)):
        sum1+=(test[i]-train[i])**2
    return np.sqrt(sum1)
main()

