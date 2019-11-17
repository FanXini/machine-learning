import numpy as np
import math
import copy
import  sys

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0,2,0.7,100.0)
# 城市数，蚁群
(city_num, ant_num) = (48,48)
dataSet=np.loadtxt("E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/TSP.txt",dtype=np.float32)
distance_x=dataSet[...,1]/10
distance_y=dataSet[...,2]/10
# 城市坐标
# distance_x = [
#     178,272,176,171,650,499,267,703,408,437,491,74,532,
#     416,626,42,271,359,163,508,229,576,147,560,35,714,
#     757,517,64,314,675,690,391,628,87,240,705,699,258,
#     428,614,36,360,482,666,597,209,201,492,294]
# distance_y = [
#     170,395,198,151,242,556,57,401,305,421,267,105,525,
#     381,244,330,395,169,141,380,153,442,528,329,232,48,
#     498,265,343,120,165,50,433,63,491,275,348,222,288,
#     490,213,524,244,114,104,552,70,425,227,331]
# 城市距离和信息素
distance_graph = [ [0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [ [1.0 for col in range(city_num)] for raw in range(city_num)]
class Ant(object):
    def __init__(self,ID):
        self.ID=ID
        self.__init_data()


    def __init_data(self):
        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.history_city = [True for i in range(city_num)]  # 探索城市的状态
        self.current_city =  np.random.randint(0, city_num - 1)  # 随机初始出生点
        self.path.append(self.current_city)
        self.history_city[self.current_city] = False
        self.move_count = 1

    def search_path(self):
        self.__init_data()
        while self.move_count<city_num:
            next_city=self.__choise_next_city()
            self.__move(next_city)
        self.__get_total_distant()


    #选择下一个要遍历的城市下标
    def __choise_next_city(self):
        total_pro=0.0
        next_city=-1
        select_probability=[0.0 for i in range(city_num)]
        #先计算每一个未遍历城市被选中的概率
        for i in range(city_num):
            if self.history_city[i]:
                select_probability[i] =pow(pheromone_graph[self.current_city][i], ALPHA) * pow((1.0/distance_graph[self.current_city][i]), BETA)
                total_pro += select_probability[i]

        if total_pro>0:
            #随机生成一个概率
            temp_pro=np.random.uniform(0.0,total_pro)
            for i in range(city_num):
                if self.history_city[i]:
                    temp_pro-=select_probability[i]
                    if temp_pro<0.0:
                        next_city=i
                        break

        if(next_city==-1):
            next_city=np.random.randint(0,city_num-1)
            while((self.history_city[next_city])==False): #if==False,说明已经遍历过了
                next_city=np.random.randint(0,city_num-1)

        return next_city

    def __move(self,next_city):
        self.path.append(next_city) #更新路径
        self.history_city[next_city]=False #更新禁忌表
        self.move_count+=1 #遍历次数+1
        self.total_distance+=distance_graph[self.current_city][next_city] #更新路程
        self.current_city=next_city #更新当前位置

    def __get_total_distant(self):
        start=self.path[0]
        end=self.path[city_num-1]
        self.total_distance+=distance_graph[end][start]


class TSP(object):
    def __init__(self):
        self.iter=0
        self.max_iter=10000
        self.__cal_distant()
        self.ants=[Ant(id) for id in range(ant_num)]
        self.best_ant=Ant(-1)
        self.best_ant.total_distance=1<<31

    def __cal_distant(self):
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] =float(int(temp_distance + 0.5))

    #修改信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素，使用的是蚁周模型：当蚁群全部遍历一次地图后再修改信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1,city_num):
                start, end = ant.path[i-1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                #pheromone_graph[i][j] =pheromone_graph[i][j] * RHO + temp_pheromone[i][j]
                pheromone_graph[i][j] = (1 - RHO) * pheromone_graph[i][j]  + RHO * temp_pheromone[i][j]

    def run(self):
        while self.iter<self.max_iter:
            for ant in self.ants:
                ant.search_path()
                if(ant.total_distance<self.best_ant.total_distance):
                    self.best_ant=copy.deepcopy(ant)
            print("第{0}次迭代，目标最短距离是{1}".format(self.iter,self.best_ant.total_distance))
            self.__update_pheromone_gragh()
            self.iter+=1

if __name__=="__main__":
    TSP().run()

