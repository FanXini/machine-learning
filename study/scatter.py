import matplotlib.pyplot as plt
import numpy as np
a = np.array([1,2,3,4])
b = np.array([1,2,3,4])
c = np.array([2,3,4,5])
d = np.array([2,3,4,5])
'''
figure的作用新建绘画窗口,独立显示绘画的图片
figsize 表示新建绘画窗口的大小
dpi是分辨率
'''
plt.figure(figsize = (8,5),dpi = 80)
'''
这个比较重要,需要重点掌握,参数有r,c,n三个参数
使用这个函数的重点是将多个图像画在同一个绘画窗口.
r            表示行数
c            表示列行
n            表示第几个
'''
plt.subplot(2,1,1)#表示一个绘画窗口下建立两个子图，选择第一个作为绘画图
plt.scatter(a,b,c = 'r')
#plt.scatter(c,d,c = 'b')
#将会在第一个图中画出两种颜色不一样的点
plt.subplot(2,1,2)
plt.scatter(a,b,c = 'black')
plt.show()