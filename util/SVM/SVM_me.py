
from numpy import *
import time
import matplotlib.pyplot as plt

def calcKernelValue(matrix_x, sample_x):
    numSamples = matrix_x.shape[0]
    matrix_x=mat(matrix_x)
    sample_x=mat(sample_x)
    kernelValue = mat(zeros((numSamples, 1)))
    kernelValue = matrix_x * sample_x.T
    return  kernelValue



# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples))) #num*num矩阵，存储内积的值
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :])
    return kernelMatrix

class SVMClass:
    def __init__(self,train_x,train_y,C,toler):
        self.train_x=train_x
        self.train_y=train_y
        self.C=C
        self.toler=toler
        self.numSampels=train_x.shape[0]
        self.alphas=mat(zeros((self.numSampels,1)))
        self.b=0
        self.errorCache=mat(zeros((self.numSampels,2)))
        self.kernelMat = calcKernelMatrix(self.train_x)
def calcError(SVM,i):
    w=mat(zeros((SVM.train_x.shape[1]))).T
    for j in range(SVM.train_x.shape[0]):   #求得法向量W W=alphat*y*x
        w+=float(SVM.alphas[j]*SVM.train_y[j])*mat(SVM.train_x[j]).T
    fi=float(dot(mat(SVM.train_x[i]),w)+SVM.b)
    error=fi-float(SVM.train_y[i])
    return error

# select alpha j which has the biggest step
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]  # mark as valid(has been optimized)
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0]  # mat.A return array
    maxStep = 0;
    alpha_j = 0;
    error_j = 0

    # find the alpha with max iterative step
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    # if came in this loop first time, we select alpha j randomly
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.numSampels))
        error_j = calcError(svm, alpha_j)

    return alpha_j, error_j



# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]


def innerLoop(svm,alpha_i):
    error_i=calcError(svm,alpha_i)
    #开始判断是否违背KTT条件
    #满足KTT条件的三种情况
    # 1.y(wx+b)>1 and alphas=0
    # 2.y(wx+b)==1 and 0<alphas<C
    # 3.y(wx+b)<1 and alphas=C
    #不满足KTT条件的三种情况
    #1.y(wx+b)>1 and alphas>0
    #2.y(wx+b)==1  该点在边界上，不需要优化alphas
    #3.y(wx+b)<1 and alphas<C
    #借助上面求得的error_i=(wx+b)-y=f(i)-yi
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    if (svm.train_y[alpha_i]*error_i<-svm.toler and svm.alphas[alpha_i]<svm.C) or(svm.train_y[alpha_i]*error_i>svm.toler and svm.alphas[alpha_i]>0):
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        #step 3:计算 eta
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
              - svm.kernelMat[alpha_j, alpha_j]
        if eta >= 0:
            return 0


        #step 4: updata alphas[alphas_j]
        svm.alphas[alpha_j]-=(float(svm.train_y[alpha_j])*(error_i-error_j))/eta

        # step 5: clip alpha j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # step 6: if alpha j not moving enough, just return
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            return 0

        # step 7: update alpha i after optimizing aipha j
        svm.alphas[alpha_i]+=float(svm.train_y[alpha_i])*float(svm.train_y[alpha_i])*(alpha_j_old-svm.alphas[alpha_j])

        # step 8: update threshold b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # step 9: update error cache for alpha i, j after optimize alpha i, j and b
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0






def train_SVM(train_x,train_y,C,toler,maxIter):
    startTime=time.time()
    svm=SVMClass(train_x,train_y,C,toler)  #把用一个SVM类存储训练数据和参数


    #start training
    entireSet=True
    alphaPairsChanged=0
    iterCount=0
    # Iteration termination condition:
    # 迭代终止条件
    # 	Condition 1: reach max iteration  迭代到最后了
    # 	Condition 2: no alpha changed after going through all samples,
    # 				 in other words, all alpha (samples) fit KKT condition  所有的样本都满足了ktt条件

    while(iterCount<maxIter) and ((alphaPairsChanged>0) or entireSet):
        alphaPairsChanged=0

        # update alphas over all training examples
        if entireSet:
            for i in range(svm.numSampels):
                alphaPairsChanged += innerLoop(svm, i)
            print('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1
        # update alphas over examples where alpha is not 0 & not C (not on boundary)
        else:
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)
            print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1

        # alternate loop over all examples and non-boundary examples
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    print
    'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return svm

# testing your trained svm model given test set
def testSVM(svm, test_x, test_y):
    numTestSamples = test_x.shape[0]
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    supportVectors = svm.train_x[supportVectorsIndex]
    supportVectorLabels = mat(svm.train_y[supportVectorsIndex])
    supportVectorAlphas = svm.alphas[supportVectorsIndex].T
    matchCount = 0
    for i in range(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, test_x[i, :])
        predict = dot(multiply(supportVectorLabels, supportVectorAlphas),kernelValue) + svm.b
        if predict>=0:
            if test_y[i]==1:
                matchCount+=1
        else:
            if test_y[i]==-1:
                matchCount+=1
    accuracy = float(matchCount) / numTestSamples
    return accuracy


# show your trained svm model only available with 2-D data
def showSVM(svm):
    if svm.train_x.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # draw all samples
    for i in range(svm.numSampels):
        if svm.train_y[i] == -1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')
        elif svm.train_y[i] == 1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')

    # mark support vectors
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

    # draw the classify line
    w = zeros((2, 1))
    for i in supportVectorsIndex:
        w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)
    min_x = min(svm.train_x[:, 0])[0, 0]
    max_x = max(svm.train_x[:, 0])[0, 0]
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.show()
