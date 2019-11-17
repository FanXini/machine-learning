# import pylab as  plt
# import  numpy as np
#
# # x=[1,2,3,4]
# # y=[1,2,3,4]
# # plt.plot(x,y,'bx')
# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)
# #print(t)
#
# # red dashes, blue squares and green triangles
# #plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
#
# x=np.arange(0,10,1)
# y=2*x+1
# x1=np.arange(0,10,1)
# y1=x*x
# line,line1=plt.plot(x,y,'b^',x1,y1,'gx-')
# plt.setp(line,'color','k')
# plt.setp(line1,'color','y')
# plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin) * np.random.rand(n) + vmin


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

xs = [1,2,3]
ys = [1,2,3]
zs = [1,2,3]
ax.scatter(xs, ys, zs, 'o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()