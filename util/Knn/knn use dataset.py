import numpy
import math
import operator
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn


class TreeNode(object):
    __slots__ = ('data', 'type', 'split', 'leftChild', 'rightChild')

    def __init__(self, data, type):
        self.data = data
        self.type = type


class Stack(object):
    # 初始化栈为空列表
    def __init__(self):
        self.items = []

    # 判断栈是否为空，返回布尔值
    def is_empty(self):
        return self.items == []

    # 返回栈顶元素
    def peek(self):
        return self.items[len(self.items) - 1]

    # 返回栈的大小
    def size(self):
        return len(self.items)

    # 把新的元素堆进栈里面（程序员喜欢把这个过程叫做压栈，入栈，进栈……）
    def push(self, item):
        self.items.append(item)

    # 把栈顶元素丢出去（程序员喜欢把这个过程叫做出栈……）
    def pop(self):
        return self.items.pop()

    # 判断是否存在该元素
    def contains(self, item):
        return item in self.items


def creatDataSet():
    iris=load_iris();
    dataset=[]
    dataArray=numpy.array(iris.data)
    typeArray=numpy.array(iris.target)
    for index in numpy.arange(0,dataArray.shape[0]):
        dataset.append(TreeNode(dataArray[index],typeArray[index]))
    return numpy.array(dataset)



# 构造KD树 ，采用递归的方法
def createTree(dataset):
    nodeTemp = TreeNode(None, None)
    if (dataset.__len__() == 0):
        nodeTemp = None
    else:
        value = []
        for data in dataset:
            value.append(data.data)  # 把特征值（x,y）加入到value列表中来
        valueSet = numpy.array(value)  # 转换为矩阵
        variance = numpy.var(valueSet, axis=0)  # 求方差[第一列的方差，第二列的方差]
        split = variance.argmax()  # 求得最大方差的下标
        objectRow = valueSet.T[split]  # 得到包含目标根节点的数据
        sortRow = numpy.argsort(objectRow)  # 得到从大到小的原下标数组
        objectRowNum = sortRow[sortRow.shape[0] // 2]  # 得到根节点的行数
        nodeTemp = dataset[objectRowNum]  # 得到根节点
        nodeTemp.split = split  # 将最大方差的下标赋值个对象
        # print(node.data)  #输出节点的特征值（x,y）
        # print(node.type) #输出节点的类型
        dataset = numpy.delete(dataset, objectRowNum, 0)  # 把根节点从dataset中去掉
        leftDateSet = []
        rightDateSet = []
        for remainnode in dataset:
            if (remainnode.data[nodeTemp.split] > nodeTemp.data[nodeTemp.split]):
                rightDateSet.append(remainnode)
            else:
                leftDateSet.append(remainnode)
        nodeTemp.leftChild = createTree(leftDateSet)  # 递归，构造左子树
        nodeTemp.rightChild = createTree(rightDateSet)  # 递归，构造右子树
    return nodeTemp


def distant(array1, array2):
    x = numpy.array(array1)  # 转换成数组
    y = numpy.array(array2)  # 转换成数组
    diffArray = x - y  # 相减
    powArray = diffArray ** 2  # 平方
    addArray = powArray.sum()  # 平方和
    return addArray ** 0.5  # 开根号


# 递归遍历KD树
def preoderByRecursion(node):
    if node != None:
        print("{0} {1}".format(node.data,node.split))
        preoderByRecursion(node.leftChild)
        preoderByRecursion(node.rightChild)

def kdTreeSearch(node, inX, path, stack):
    if node not in path:
        path.append(node)
    stack.push(node)  # 进栈
    if (node.data[node.split] >= inX[node.split]):  # 判断是遍历左子树还是右子树 如果 目标节点的值要小，则遍历左子树
        if node.leftChild != None:  # 如果左子树不为空 递归遍历左子树
            return kdTreeSearch(node.leftChild, inX, path, stack)
    else:
        if node.rightChild != None:  # 如果右子树不为空 递归遍历右子树
            return kdTreeSearch(node.rightChild, inX, path, stack)
    # if node.leftChild == None or node.rightChild == None:  # 到了叶子节点
    dis = distant(inX, node.data)
    # print(node.data)
    target = stack.pop()  # 把当前节点去掉
    # print(dis)  # 找到叶子节点和查询节点之间的距离
    # 开始回溯
    while (stack.size() > 0):
        topNode = stack.pop()
        # print(topNode.data)
        if topNode not in path:
            path.append(topNode)
        split = topNode.split  # 这个地方必须是小于，不能是小于等于，因为就算是等于的话，也没有必要遍历它的弄一半子树的
        if math.fabs(topNode.data[split] - inX[split]) < dis:  # 如果出栈节点和查询节点在split方向的距离<dis，则需要遍历该节点的另一半子树
            if topNode.data[split] <= target.data[
                split]:  # 如果topNode.data[split]<target.data[split]的话，说明target在topNode的右边，因此要遍历左子树
                childNode = topNode.leftChild
                if childNode != None:
                    stack.push(childNode)  # 入栈
            else:
                childNode = topNode.rightChild
                if childNode != None:
                    stack.push(childNode)  # 入栈
            # print(distant(inX, topNode.data))
            if (distant(inX, topNode.data) < dis):
                #print("更新")
                dis = distant(inX, topNode.data)  # 更新dis
                target = topNode
    return (dis, target, path)  # 对结果进行处理，得出待分类目标的type


#对结果进行处理，得出待分类目标的type
def classify(path,inX,k): # path是遍历到的节点，inX是待确认点，k表示根据周围k个节点来进行分类
    disArray=[]  #用来保存历史遍历到的节点与inX的距离
    for historyNode in path:
        disArray.append(distant(historyNode.data,inX))
    array=numpy.argsort(disArray) #排序，得到由小到大的的坐标
    classData={}  #用来保存k个最邻近节点类型
    for index in numpy.arange(0,k):
        classData[path[array[index]].type]=classData.get(path[array[index]].type,0)+1
    return sorted(classData,key=lambda x:classData[x])[-1] #返回确认类型




if __name__ == "__main__":
    inX=input("请输入要预测的数（输入4个数，用空格分开）：")
    k=int(input("请输入k值："))
    inX=inX.split(' ')
    inX=numpy.array(inX,dtype=numpy.float32)
    dataset = creatDataSet()
    headNode = createTree(dataset)
    #preoderByRecursion(headNode)
    stack = Stack()
    path = []
    #inX=[1.1,10.2,10.3, 0.4]
    dis, target, path = kdTreeSearch(headNode,inX, path, stack)
    # print('最近的距离')
    # print(dis)
    # print('最近的节点')
    # print(target.data)
    # print('路径')
    # for data in path:
    #     print('{0} {1}'.format(data.data,data.type))
    type=classify(path,inX,5)
    outState='{0}的预测类型是{1}'
    print(outState.format(inX,type))
