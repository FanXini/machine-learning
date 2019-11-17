# coding=utf-8
import math
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import accuracy_score


def trtype(s):  # 定义类别转换函数
    types = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return types[s]

x = np.loadtxt('E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/testIris.txt',delimiter=',',usecols=(0,1,2,3),dtype=float)  # 读入数据，第五列转换为类别012
y = np.loadtxt('E:/Users/fanxin/PycharmProjects/机器学习作业/util/DataSet/testIris.txt', delimiter=',', usecols=(4),dtype=int)
#x, y = np.split(data, (4,), axis=1)  # 切分data和label

pca = PCA(n_components=2)
x = pca.fit_transform(x)  # 为方便绘图，对x进行PCA降维至二维


# 划分测试集和训练集
def label_tr(y):  # 标签转换，将一维标签转换为三维
    l = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    ys = []
    for i in range(len(y)):
        ys.append(l[int(y[i])])
    return np.array(ys)


def inv_label_tr(y_1d):  # 标签转换逆过程

    y_pres = []
    for i in range(y_1d.shape[0]):
        for j in range(3):
            if (y_1d[i][j] == 1):
                y_lable = j
        y_pres.append(y_lable)

    return np.array(y_pres)


y = label_tr(y)
# 划分数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6,test_size=0.4)

random.seed(0)


def rand(a, b):  # 随机数函数
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):  # 矩阵生成函数
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):  # 激活函数
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):  # 激活函数求导
    return x * (1 - x)


class BPNeuralNetwork:  # BP神经网络类
    def __init__(self):  # 初始化
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        # 初始化输入、隐层、输出元数
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # 初始化神经元
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # 初始化权重矩阵
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # 初始化权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # 初始化偏置
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):

        # 激活输入层
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # 激活隐层
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # 激活输出层
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # 反向传播
        self.predict(case)
        # 求输出误差
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # 求隐层误差
        hidden_deltas = [0.0] * self.hidden_n
        print(hidden_deltas)
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # 更新输出权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # 更新输入权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 求全局误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=1000, learn=0.05, correct=0.1):
        # 训练神经网络
        print("start")
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def fit(self, x_test):  # 离散预测函数用于输出数据
        y_pre_1d = []
        for case in x_test:
            y_pred = self.predict(case)
            for i in range(len(y_pred)):
                if (y_pred[i] == max(y_pred)):
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
            y_pre_1d.append(y_pred)
        return inv_label_tr(np.array(y_pre_1d))

    def fit2(self, x_test):  # 连续预测函数用于画图
        y_pre_1d = []
        for case in x_test:
            w = np.array([0, 1, 2])
            y_pred = self.predict(case)
            y_pre_1d.append(np.array(y_pred).dot(w.T))
        return np.array(y_pre_1d)


if __name__ == '__main__':  # 主函数
    nn = BPNeuralNetwork()
    nn.setup(2, 5, 3)  # 初始化
    nn.train(x_train, y_train, 100000, 0.05, 0.1)  # 训练
    y_pre_1d = nn.fit(x_test)  # 测试
    y_test_1d = inv_label_tr(y_test)
    print(accuracy_score(y_pre_1d, y_test_1d))  # 打印测试精度

    # 画图
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点

    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    grid_hat = nn.fit2(grid_test)  # 预测结果
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
    plt.title(u'BPNN二特征分类', fontsize=15)
    plt.show()
    print(grid_hat.shape)