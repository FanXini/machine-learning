import numpy

def createData():
    dataSet=numpy.loadtxt('E:/Users/fanxin/Desktop/Data.txt')
    dataSet=dataSet.astype(dtype=float)
    print(dataSet)
    print(dataSet[:,0])  #data[a,b]相当于data[a][b]
    print(dataSet[:dataSet.shape[0]])
    a=numpy.array([[1,3,4],[1,2,3]])
    b=numpy.array([1,1,1])
    c=b.reshape(3,1)
    print(c)
    print(numpy.dot(a,c))

k = numpy.arange(8).reshape(2,2,2)  #层/行/列
print (k) # [[[0 1],[2 3]],[[4 5],[6 7]]]
print(k.shape)
print (k[1][0][0])  #4
# 轴变换 transpose 参数:由轴编号组成的元组
m = k.transpose((1,0,2)) # m[y][x][z] = k[x][y][z]
print (m) # [[[0 1],[4 5]],[[2 3],[6 7]]]
print (m[0][1][0])

a=[[1,2,3],[4,5,6]]
b=numpy.mat(a)
print(type(b))
c=numpy.array(a)
print(type(c))
errorCache = numpy.mat(numpy.zeros((10, 2)))
print(errorCache)

a=numpy.array([[1,2,3],[4,5,6]],order="F")
print(a)
b=numpy.zeros([5,5],dtype=[('x','i4'),('y','i4')])
print(b)
c=numpy.arange(1,5,2)
print(c)
d=numpy.linspace(1,2,5,endpoint=False,retstep=True)
print(d)
a = numpy.array([[1,2,3],[3,4,5],[4,5,6]])
print(a[0:,1])
print(a[[0,1,2],0:])
a=numpy.arange(0,60)
a.reshape((2,30))
for x in numpy.nditer(a):
    print(x)
a=numpy.array([[1,2],[3,4]])
b=numpy.array([[5,6],[7,8]])
print ('沿轴 0 堆叠两个数组：')
c=numpy.stack((a,b),axis=0)
print (c)
print ('\n')

print ('沿轴 1 堆叠两个数组：')
print ((numpy.stack((a,b),1)))

a=[[1,2,3],[4,5,6],[7,8,9]]
#a.insert(1,[2,3,4])
b=numpy.insert(a,2,[3,2,1],axis=1)

a=numpy.array([[1,2],[3,4]])
b=numpy.array([[5,6],[7,8]])
c=numpy.hstack((a,b))
d=numpy.vstack((a,b))
print(c)
print(d)
print(numpy.delete(c,1,axis=1))
a=numpy.arange(60)
b=a.reshape(2,30)
print(b)


a=numpy.random.uniform(low=-10,high=10,size=(100,2))
b=numpy.square(a)
print(a[0])
print(b[0])
c=numpy.sum(a,axis=1)
a=numpy.array([1,2])
b=numpy.array([1,2])
print(a*b)


