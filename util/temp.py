import  numpy as np

array=np.loadtxt("DataSet/Data.txt",dtype=float)
print(array)
print(len(array))

a=array.sum(axis=0)

print(a[0]+a[1])