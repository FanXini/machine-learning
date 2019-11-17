import math
class Util(object):
    @staticmethod
    def  get_variance(array):  #求方差
        sum = 0
        List = list(array)
        for num in List:
            sum += num
        ave = sum / List.__len__()
        sum = 0;
        for num in List:
            sum += math.pow(num - ave, 2)
        return sum / List.__len__()

    @staticmethod     #求欧几里得距离
    def get_distant(node1,node2):
        return math.sqrt(math.pow(node1.x - node2.x, 2) + math.pow(node1.y - node2.y, 2))