import numpy as np
a=np.loadtxt("E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/TSP.txt",dtype=np.float32)
distance_x=a[...,1]/10
distance_y=a[...,2]/10
city_num=48
distance_graph = [ [0.0 for col in range(city_num)] for raw in range(city_num)]
for i in range(city_num):
    for j in range(city_num):
        temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
        temp_distance = pow(temp_distance, 0.5)
        distance_graph[i][j] = temp_distance
print(distance_graph)
totalDis=0
for i in range(city_num-1):
    totalDis+=distance_graph[i][i+1]

print(totalDis)