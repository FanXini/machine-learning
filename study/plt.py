from  matplotlib import pyplot as plt #pyplot是matplotlib的重要函数，用来绘制2d图性
import numpy as np

#绘制直线
# x=np.arange(1,11)
# z=np.arange(1,11)
# y=2*x+x*x
# plt.title("line y=2x+3z")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x,y,'ob')
# plt.show()

#绘制y=sinx
# x=np.arange(0,3*np.pi,0.1)
# y=np.sin(x)
# plt.title('y=sin(x)')
# plt.plot(x,y)
# plt.show()

# x=np.arange(0,3*np.pi,0.1)
# y_sinx=np.sin(x)
# y_cosx=np.cos(x)
# #激活第一个子图，高为2，宽为1，第2个
# plt.subplot(2,1,1)
# plt.plot(x,y_sinx)
# plt.title("y=sin(x)")
# plt.plot(x,y_sinx,'or')
# #激活第2个子图，高为2，宽为1，第2个
# plt.subplot(2,1,2)
# plt.title("y=cos(x)")
# plt.plot(x,y_cosx,'--b')
# plt.show()

x =  [5,8,10]
y =  [12,16,6]
x2 =  [6,9,11]
y2 =  [6,15,7]
plt.bar(x, y, align =  'center')
plt.bar(x2, y2, color =  'g', align =  'center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()