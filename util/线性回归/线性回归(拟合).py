import numpy
import pylab
import math
#生成数据集
def createData():
    dataSet=numpy.loadtxt('E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/Data.txt') #读取Data.txt文件
    dataSet=dataSet.astype(dtype=float)  #转化成float类型
    return dataSet

def line_regression(dataSet):      #目标函数设为   y=ax+b
    learn_rate=0.001    #学习速率
    init_a=0     #a初始化为0
    init_b=0     #b初始化为0
    iter_count=1000 #迭代次数
    a,b=optimization(dataSet,init_a,init_b,learn_rate,iter_count)  #寻找最优的参数a、b，使得拟合效果最佳
    print(a,b)
    plot_data(dataSet, a, b)

def plot_data(data,a,b):

    #plottting
    x = data[:,0]
    y = data[:,1]
    y_predict = a*x+b
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()


def optimization(dataSet,init_a,init_b,learn_rate,iter_count):
    a=init_a
    b=init_b
    for i in numpy.arange(0,iter_count): #迭代次数
        a,b=compute_gradient(a,b,dataSet,learn_rate)
        if(i%100==0):
            print('iter {0}:error={1}'.format(i, compute_error(a, b, dataSet)))
    return (a,b)

def compute_error(a,b,dataSet):
    sum=0
    for i in numpy.arange(len(dataSet)):
        x=dataSet[i,0]
        y=dataSet[i,1]
        y_predict=a*x+b
        sum+=math.pow(y-y_predict,2)
    return sum/len(dataSet)


def compute_gradient(a,b,dataSet,learn_rata):
    a_gradient = 0 #a的偏导数
    b_gradient = 0 #b的偏导数

    N = float(len(dataSet))
    # Two ways to implement this
    # first way
    for i in range(0,len(dataSet)):  #循环每一对数据，求a,b的偏导数
        x = dataSet[i,0]
        y = dataSet[i,1]

        #computing partial derivations of our error function
        #b_gradient = -(2/N)*sum((y-(m*x+b))^2)
        #m_gradient = -(2/N)*sum(x*(y-(m*x+b))^2)
        a_gradient += -(2 / N) * x * (y - ((a * x) + b))
        b_gradient += -(2/N)*(y-((a*x)+b))

    # Vectorization implementation
    # x = dataSet[:, 0]
    # y = dataSet[:, 1]
    # b_gradient = -(2 / N) * (y - m_current * x - b_current)
    # b_gradient = np.sum(b_gradient, axis=0)
    # m_gradient = -(2 / N) * x * (y - m_current * x - b_current)
    # m_gradient = np.sum(m_gradient, axis=0)
    # update our b and m values using out partial derivations
    #print(a_gradient,b_gradient)
    new_a = a - (learn_rata * a_gradient)  #如果a_gradient>0，说明在a点是递增的，所以按照梯度下降原理，我们要往相反方向走，所以要减
                                          #如果a_gradient<0，说明在a点是递减的，所以按照梯度下降原理，要按照这个方向走下去，所以减去一个
                                            #负值，相当于加上一个正值
    new_b = b - (learn_rata * b_gradient)  #原理同上
    return [new_a, new_b]

if __name__=="__main__":
    dataSet=createData()
    line_regression(dataSet)