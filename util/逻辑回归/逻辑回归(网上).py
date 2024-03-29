from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import numpy
import math

#load the dataset
data = loadtxt('E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]
print(X)

pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail', 'Pass'])
show()


