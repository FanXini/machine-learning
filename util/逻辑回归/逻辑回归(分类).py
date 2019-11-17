import  numpy
import  math

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
    Wt=numpy.ones(attributeSet.shape[1],dtype=float)
    Wt=Wt.reshape(len(Wt),1)
    learn_rate=0.001
    iter_count=1000
    for i in numpy.arange(iter_count):
        Wt=compute_gradient(Wt,attributeSet,type,learn_rate)
    return Wt

def compute_gradient(Wt,attributeSet,type,learn_rate):
    current_Wt=Wt
    result=numpy.dot(attributeSet,current_Wt)   #权值*属性
    sigResult=[]
    for i in numpy.arange(result.shape[0]):   #进行激活，将结果归一到(0,1)
        sigResult.append(Sigmoid(result[i]))
    error=getError(sigResult,type)    #error是在当前权值下，求得得结果与实际值得差
    diff=learn_rate*numpy.dot(attributeSet.T,error)
    diff_T=diff.reshape(len(diff),1)
    new_Wt=numpy.add(current_Wt,diff_T)
    return new_Wt
    # print(numpy.subtract(current_Wt,[[1],[1],[1],[1]]))


def getError(sigResult,type):
    error=numpy.subtract(typeSet,sigResult)
    return error

def Sigmoid(num):
    return 1/(1+math.exp(-num))


def predict(attribute,Wt):
    result=numpy.dot(attribute,Wt)
    sigResult=Sigmoid(result)
    if(sigResult<0.5):
        print("该花种类是:山鸢尾（Iris-setosa）")
    else:
        print("该花种类是:杂色鸢尾（Iris-versicolor）")


if __name__=="__main__":
    dataSet,attributeSet,typeSet=CreateDataSet() #dataSet是将特征集合和类型集合合并之后的数组
    Wt=logistic_regression(attributeSet,type)
    print("权值是:{0}".format(Wt))
    # inX=input("请输入想要预测花的属性值(分别输入：花萼长度、花萼宽度、花瓣长度、花瓣宽度)用空格分开:")
    # attribute=inX.split(' ')
    # attribute=numpy.array(attribute,dtype=float)
    # predict(attribute,Wt)
    for i in numpy.arange(len(attributeSet)):
        predict(attributeSet[i],Wt)





