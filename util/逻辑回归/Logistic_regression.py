import numpy
import math
import pylab as plb

def CreateDataSet():
    attributeSet=numpy.loadtxt('Iris.txt',delimiter=',',usecols=(0,1,2,3),dtype=float)
    typeSet=numpy.loadtxt('Iris.txt',dtype=str,delimiter=',',usecols=(4))
    type=[]
    for i in range(len(typeSet)):
        if (typeSet[i] == 'Iris-setosa'):
            type.append(0)
        elif(typeSet[i]=='Iris-versicolor'):
            type.append(1)
    dataSet = numpy.insert(attributeSet, 4, values=type, axis=1)
    return dataSet,attributeSet,type

def logistic_regression(attributeSet,type):
    learn_rate=0.001    #学习速率
    iter_count=1000     #迭代次数
    Wt = numpy.ones(attributeSet.shape[1], dtype=float)  #初始化权值全部为1
    Wt = Wt.reshape(len(Wt), 1)
    for i in range(iter_count):
        Wt=findBestWeight(learn_rate,attributeSet,type,Wt)
    return Wt

def findBestWeight(learn_rate,attributeSet,type,Wt):
    current_Wt=Wt
    gradient={}   #用来存储每一个权值对应的偏导
    result=numpy.dot(attributeSet,Wt)
    for i in range(current_Wt.shape[0]):
        gradient[i]=0
    for i in range(len(attributeSet)):
        for key in gradient:
            gradient[key]+=(Sigmoid(result[i])-type[i])*attributeSet[i][key]  #求出第key个权值的偏导
    diff=[]
    for key in gradient:
        diff.append(gradient[key]*learn_rate)
    diff=numpy.array(diff).reshape(len(diff),1)
    new_Wt=current_Wt-diff
    return new_Wt

#激活函数，将结果缩放到(0,1)
def Sigmoid(num):
    return 1/(1+math.exp(-num))

def plot(dataSet,Wt):
    att0=dataSet[:,0]
    att1 = dataSet[:, 1]
    att2 = dataSet[:, 2]
    att3 = dataSet[:, 3]
    y_predict=Wt[0]*att0+Wt[1]*att1+Wt[2]*att2+Wt[3]*att3
    print(y_predict)
    type=dataSet[:,2]
    plb.plot(y_predict,'r-')
    plb.show()



def predict(attribute,Wt):
    result=numpy.dot(attribute,Wt)
    sigResult=Sigmoid(result)
    if(sigResult<0.5):
        print("该花种类是:山鸢尾（Iris-setosa）")
        return 0
    else:
        print("该花种类是:杂色鸢尾（Iris-versicolor）")
        return 1

if __name__=="__main__":
    dataSet,attributeSet,type=CreateDataSet()
    Wt=logistic_regression(attributeSet,type)
    print("通过梯度下降得到的权值:{0}".format(Wt))
    plot(dataSet,Wt)
    inX=input("请输入想要预测花的属性值(分别输入：花萼长度、花萼宽度、花瓣长度、花瓣宽度)用空格分开:")
    attribute=inX.split(' ')
    attribute=numpy.array(attribute,dtype=float)
    predict(attribute,Wt)

    #验证是否正确
    # result=[]
    # for i in numpy.arange(len(attributeSet)):
    #     result.append(predict(attributeSet[i],Wt))
    # result=numpy.array(result)
    # bool=result==0
    # print(len(result[bool]))