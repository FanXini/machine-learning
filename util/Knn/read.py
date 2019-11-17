import  numpy as np
x = np.loadtxt('E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/testIris.txt',delimiter=',',usecols=(0,1,2,3),dtype=float)  # 读入数据，第五列转换为类别012
print(x)