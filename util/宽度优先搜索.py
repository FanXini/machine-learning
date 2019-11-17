import numpy
class Stack(object):
    def __init__(self):
        self.items=[]
    def push(self,item):
        self.items.append(item)
    def peek(self):
        return self.items[len(self.items)-1]
    def pop(self):
        return self.items.pop()
    def is_empty(self):
        return len(self.items)==0
    def size(self):
        return len(self.items)

class TreeNode(object):
    __slots__ = ('data', 'type', 'split', 'leftChild', 'rightChild', 'parent')

    def __init__(self, data, type):
        self.data = data
        self.type = type

def creatDataSet():
    dataset = []
    dataset.append(TreeNode((2, 3), 'A'))
    dataset.append(TreeNode((2.1, 3.1), 'A'))
    dataset.append(TreeNode((5, 4), 'A'))
    dataset.append(TreeNode((9, 6), 'B'))
    dataset.append(TreeNode((4, 7), 'B'))
    dataset.append(TreeNode((8, 1), 'B'))
    dataset.append(TreeNode((7, 2), 'B'))
    dataset.append(TreeNode((9.4, 6.2), 'B'))
    dataset.append(TreeNode((10.3, 8.1), 'B'))
    dataset.append(TreeNode((12, 4.3), 'B'))
    dataset.append(TreeNode((19, 6), 'B'))

    return numpy.array(dataset)


# 构造KD树 ，采用递归的方法
def createTree(dataset,parent):
    # nodeTemp = TreeNode(None, None)
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
        nodeTemp.parent = parent
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
        nodeTemp.leftChild = createTree(leftDateSet,nodeTemp)  # 递归，构造左子树
        nodeTemp.rightChild = createTree(rightDateSet,nodeTemp)  # 递归，构造右子树
    return nodeTemp

def BFS(stack,objectData,time,space): #time表示时间复杂度，space表示空间复杂度
    while stack.size()>0:
        time+=1
        if(stack.size()>space):
            space=stack.size()
        topNode=stack.pop()
        if(topNode.data==objectData):
            return topNode
        else:
            if topNode.leftChild!=None:
                if topNode.leftChild.data==objectData:
                    return topNode.leftChild,time,space
                else:
                    stack.push(topNode.leftChild)
            if topNode.rightChild!=None:
                if topNode.rightChild.data==objectData:
                    return topNode.rightChild,time,space
                else:
                    stack.push(topNode.rightChild)
    if stack.size==0:
        return None



if __name__=='__main__':
    dataset=creatDataSet()
    headNode=createTree(dataset,None)
    stack=Stack()
    stack.push(headNode)
    node,time,space=BFS(stack,(9.4,6.2),0,1)
    if node==None:
        print('搜索不到该数据')
    else:
        print('路径')
        while node!=None:
            print(node.data)
            node=node.parent
        print('时间复杂度')
        print(time)
        print('空间复杂度')
        print(space)

