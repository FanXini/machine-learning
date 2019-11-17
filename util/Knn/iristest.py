from sklearn.datasets import load_iris
import numpy as np

iris=load_iris()
print(iris)
data=np.array(iris.data)
target=np.array(iris.target)
target_names=np.array(['Iris-setosa','Iris-versicolor','Iris-virginica'])
f=open('E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/testIris.txt','w')
for i in range(len(iris.data)):
    strTemp=""
    for j in range(iris.data.shape[1]):
        strTemp=strTemp+str(iris.data[i][j])+','
    strTemp+=str(target[i])+'\n'
    f.write(strTemp)


