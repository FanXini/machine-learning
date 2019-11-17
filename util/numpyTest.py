import numpy
a=[1,2,3,4,5]
A=numpy.array(a)
print(A.shape[0])


b=[[1,2,3,4],[4,5,6,7],[7,8,9,10]]
B=numpy.array(b)
print(B.shape)
print(B.ndim)


x = numpy.zeros((2,3)) #创建一维长度为2，二维长度为3的二维0数组
y=numpy.ones((10,10),numpy.float32)
print(y)

z=numpy.array([1,24,3.4],dtype=numpy.float64)
print(z)
o=z.astype(numpy.int8)
print(o)

x=numpy.array(['1','3','5'],dtype=numpy.string_)
y=x.astype(numpy.int8)
print(x)
print(y)


x=numpy.array([2,4,5,6,7])
y=numpy.array(x>2,dtype=numpy.bool_)
z=numpy.ones((1,5),order=8)
print(z)
print(x+z)

chaodiao=[[[1,3],[1,3]],[[2,3],[2,3]]]
x=numpy.array(chaodiao)
print(x.shape)
print(x.ndim)

a=[1,2,3,4,5,5,6,6,]
x=numpy.array(a)
b=x[1:]
print(numpy.array(b).shape)

x = numpy.array([[1,2],[3,4],[5,6]])
print(x[1:])
print(x[:2,:1])

x = numpy.array([[1,2],[3,4],[5,6]])
print (x[[0,1]]) # [[1,2],[3,4]]
print(x[:2])
print (x[[0,1],[0,1]]) # [1,4] 打印x[0][0]和x[1][1]
print (x[[0,1]][:,[0,1]]) # 打印01行的01列 [[1,2],[3,4]]
y=numpy.array(x[:2])
print(y[[0,1]])

x=numpy.arange(1,5)
y=x.reshape((2,2))
print(x)
print(y)
print(y.T)
print(numpy.dot(y,y.T))

print()
print()
print()
# 高维数组的轴对象

k = numpy.arange(8).reshape(2,2,2)
print (k) # [[[0 1],[2 3]],[[4 5],[6 7]]]
print (k[1][0][0])
# 轴变换 transpose 参数:由轴编号组成的元组
m = k.transpose((1,0,2)) # m[y][x][z] = k[x][y][z]
print (m) # [[[0 1],[4 5]],[[2 3],[6 7]]]
print (m[0][1][0])

x=numpy.array([1,2,53,2,13])
y=numpy.array([23,32,42,13,1])
print(numpy.union1d(x,y))


x=numpy.array([[1,2,3],[3,4,1],[5,64,1]])
print(numpy.trace(x))

x = numpy.array([[1, 2, 3], [4, 5, 6]])
y = numpy.array([[7, 8, 9], [10, 11, 12]])
print(numpy.concatenate([x,y],axis=0))
print(numpy.concatenate([x,y],axis=1))
print(numpy.vstack((x,y)))
print(numpy.hstack((x,y)))
# dstack：按深度堆叠
print (numpy.split(x,2,axis=0))
a,b=numpy.split(x,2,axis=0)
print(a)
print(numpy.array(a).flatten())
# 按行分割 [array([[1, 2, 3]]), array([[4, 5, 6]])]
print (numpy.split(x,3,axis=1))
# 按列分割 [array([[1],[4]]), array([[2],[5]]), array([[3],[6]])]



# 堆叠辅助类
import numpy as np
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = np.random.randn(3, 2)
print ('r_用于按行堆叠')
print (np.r_[arr1, arr2])
'''
[[ 0.          1.        ]
 [ 2.          3.        ]
 [ 4.          5.        ]
 [ 0.22621904  0.39719794]
 [-1.2201912  -0.23623549]
 [-0.83229114 -0.72678578]]
'''
print ('c_用于按列堆叠')
print (np.c_[np.r_[arr1, arr2], arr])
'''
[[ 0.          1.          0.        ]
 [ 2.          3.          1.        ]
 [ 4.          5.          2.        ]
 [ 0.22621904  0.39719794  3.        ]
 [-1.2201912  -0.23623549  4.        ]
 [-0.83229114 -0.72678578  5.        ]]
'''
print ('切片直接转为数组')
print (np.c_[1:6, -10:-5])
'''
[[  1 -10]
 [  2  -9]
 [  3  -8]
 [  4  -7]
 [  5  -6]]
'''

# a=[1,2,4]
# x=numpy.tile(a,(2,2))
# print('2 2')
# print(x)

b=[0,1,2]
y=numpy.tile(b,(2,1,2))
print(y)
print()
print()
print()
b=[0,1,2]
y=numpy.tile(b,(2,2,2))
print(y)

a=[[1,2],[3,4]]
x=numpy.array(a)
print(x)
print(numpy.sum(x,axis=1))

a=(1,2,3)
b=numpy.array(a)
print(b)

a=[1,2,3]
b=[4,5,6]
print(numpy.multiply(a,b))
print(numpy.mat(numpy.zeros((2, 2))))
print(numpy.zeros((5,2)))

print("haha")
a=numpy.mat([1,2,3])
b=numpy.mat([2,3,4]).T
C=numpy.dot(a,b)
print(C)

a=numpy.mat([[5,2,6],[4,5,9],[7,8,4]])
b=numpy.mat([[-3,2,1],[2,-5,3],[4,5,-2]])
print(numpy.multiply(a,b))


