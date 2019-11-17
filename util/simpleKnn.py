import  numpy
import operator

def getDateSet():
    group=numpy.array([[31,3],
                       [13,8],
                       [20,1],
                       [10,10]])
    lable=["三角形","圆形","正方形","圆形"]
    return (group,lable)


def knn(dataset,label,inX,K):
    row=dataset.shape[0]
    data=numpy.tile(inX,(row,1))
    diffDate=data-dataset
    powData=diffDate**2
    addSet=numpy.sum(powData,axis=1)
    distantSet=addSet**0.5
    sortDist=distantSet.argsort()
    result={}
    for i in K:
        type=label[sortDist[i]]
        result[type]=result.get(type,0)+1
    return result


if __name__=="__main__":
    dataSet,label=getDateSet()
    result=knn(dataSet,label,[10,9],numpy.arange(dataSet.shape[0]))
    print(result)
    num=0
    for key in result:
        print(num)
        if result.get(key)>num:
            num=result.get(key)
            label=key
    print(label)



