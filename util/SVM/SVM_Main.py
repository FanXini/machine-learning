from numpy import *

from util.SVM import SVM_me

#test1:load data
print("step1:loat the data")
dataSet=loadtxt('testSet.txt',usecols=(0,1),dtype=float)
labels=loadtxt('testSet.txt',usecols=(2),dtype=int)
dataSet=array(dataSet)
labels=array(labels)
print(len(dataSet))
train_x=dataSet[0:81,:] #0-80行是训练数据，81-100是测试数据
train_y=labels[0:81]
test_x=dataSet[81:101,:]
test_y=labels[81:101]

print("step2:train SVM")
C=0.6
toler = 0.001
maxIter = 50
svm= SVM_me.train_SVM(train_x, train_y, C, toler, maxIter)

## step 3: testing
print ("step 3: testing...")
accuracy = SVM_me.testSVM(svm, test_x, test_y)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
#SVM_me.showSVM(svm)



