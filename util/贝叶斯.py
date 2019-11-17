import numpy

def getDataset():#年龄，收入，学生，信用，买了电脑
    dataset=[['<30', '高', '否', '一般', '否'],
             ['<30', '高', '否', '好', '否'],
             ['30-40', '高', '否', '一般', '是'],
             ['>40', '中', '否', '一般', '是'],
             ['>40', '低', '是', '一般', '是'],
             ['>40', '低', '是', '好', '否'],
             ['30-40', '低', '是', '好', '是'],
             ['<30', '中', '否', '一般', '否'],
             ['<30', '低', '是', '一般', '是'],
             ['>40', '中', '是', '一般', '是'],
             ['<30', '中', '是', '好', '是'],
             ['30-40', '中', '否', '好', '是'],
             ['30-40', '高', '是', '一般', '是'],
             ['>40', '中', '否', '好', '否']]
    TdataSet=[]
    for line in dataset:
        transLine=[]
        if line[0]=='<30':
            transLine.append(0)
        elif line[0]=='30-40':
            transLine.append(1)
        else:
            transLine.append(2)
        if line[1]=='高':
            transLine.append(2)
        elif line[1]=='中':
            transLine.append(1)
        else:
            transLine.append(0)
        if line[2]=='否':
            transLine.append(0)
        else:
            transLine.append(1)
        if line[3]=='好':
            transLine.append(1)
        else:
            transLine.append(0)
        if line[4]=='否':
            transLine.append(0)
        else:
            transLine.append(1)
        TdataSet.append(transLine)
    return TdataSet

def caculataPro(dataSet):
    DataSet=numpy.array(dataSet,dtype=int)
    TDataSet=DataSet.T
    column=TDataSet.shape[1] #有多少组样本
    row=TDataSet.shape[0] #每组样本有多少特征
    probability = {}
    i=0
    haveBuy=numpy.where(TDataSet[4]==1)
    haveBuyLen=numpy.array(haveBuy).shape[1]
    haveNotBuy=numpy.where((TDataSet[4]==0))
    haveNotBuyLen=numpy.array(haveNotBuy).shape[1]
    probability['haveBuyCompute']=haveBuyLen/column
    probability['haveNotBuyCompute'] = haveNotBuyLen / column
    for line in TDataSet:
        if i==0: #对年龄进行处理
            ageLine=numpy.array(line)
            ageLess30=numpy.where(ageLine==0)
            ageBetween30to40=numpy.where(ageLine==1)
            ageBig40=numpy.where(ageLine==2)
            probability['ageLess30WithBuy']=(len(numpy.intersect1d(ageLess30,haveBuy))/haveBuyLen)
            probability['ageLess30WithNotBuy']=len(numpy.intersect1d(ageLess30,haveNotBuy))/haveNotBuyLen
            probability['ageBetween30to40WithBuy']=len(numpy.intersect1d(ageBetween30to40,haveBuy))/haveBuyLen
            probability['ageBetween30to40WithNotBuy']=len(numpy.intersect1d(ageBetween30to40,haveNotBuy))/haveNotBuyLen
            probability['ageBig40WithBuy']=len(numpy.intersect1d(ageBig40,haveBuy))/haveBuyLen
            probability['ageBig40WithNotBuy']=len(numpy.intersect1d(ageBig40,haveNotBuy))/haveNotBuyLen
        if i==1:#对收入进行处理
            incomeLine=numpy.array(line)
            incomeHigh=numpy.where(incomeLine==2)
            incomeMiddle=numpy.where(incomeLine==1)
            incomeLow=numpy.where(incomeLine==0)
            probability['incomeHighWithBuy']=len(numpy.intersect1d(incomeHigh,haveBuy))/haveBuyLen
            probability['incomeHighWithNotBuy']=len(numpy.intersect1d(incomeMiddle,haveNotBuy))/haveNotBuyLen
            probability['incomeMiddleWithBuy']=len(numpy.intersect1d(incomeMiddle,haveBuy))/haveBuyLen
            probability['incomeMiddleWithNotBuy']=len(numpy.intersect1d(incomeMiddle,haveNotBuy))/haveNotBuyLen
            probability['incomeLowWithBuy']=len(numpy.intersect1d(incomeLow,haveBuy))/haveBuyLen
            probability['incomeLowWithNotBuy'] = len(numpy.intersect1d(incomeLow, haveNotBuy)) / haveNotBuyLen
        if i==2:#对学生身份进行处理
            studentLine=numpy.array(line)
            isStudent=numpy.where(studentLine==1)
            isNotStudent=numpy.where(studentLine==0)
            probability['isStudentWithBuy']=len(numpy.intersect1d(isStudent,haveBuy))/haveBuyLen
            probability['isStudentWithNotBuy'] = len(numpy.intersect1d(isStudent, haveNotBuy)) / haveNotBuyLen
            probability['isNotStudentWithBuy'] = len(numpy.intersect1d(isNotStudent, haveBuy)) / haveBuyLen
            probability['isNotStudentWithNotBuy'] = len(numpy.intersect1d(isNotStudent, haveNotBuy)) / haveNotBuyLen
        if i==3:#对信用进行处理
            creditLine=numpy.array(line)
            creditGood=numpy.where(creditLine==1)
            credigJustsoso=numpy.where(creditLine==0)
            probability['creditGoodWithBuy']=len(numpy.intersect1d(creditGood,haveBuy))/haveBuyLen
            probability['creditGoodWithNotBuy']=len(numpy.intersect1d(creditGood,haveNotBuy))/haveNotBuyLen
            probability['creditJustsosoWithBuy']=len(numpy.intersect1d(credigJustsoso,haveBuy))/haveBuyLen
            probability['creditJustsosoWithNotBuy']=len(numpy.intersect1d(credigJustsoso,haveNotBuy))/haveNotBuyLen
        if i==4: #对是否买了电脑进行处理
            pass
        i+=1
    return probability

def predic(probability):
    needPro={}
    age = input("请输入年龄:")
    income = input("请输入收入水平(high/middle/low)")
    student = input("是否是学生(yes/no)")
    credit = input("请输入信用等级(好/一般)")
    if float(age) < 30:
        ageProwithBuy = probability['ageLess30WithBuy']
        ageProwithNotBuy=probability['ageLess30WithNotBuy']
    elif 40 > float(age) > 30:
        ageProwithBuy = probability['ageBetween30to40WithBuy']
        ageProwithNotBuy=probability['ageBetween30to40WithNotBuy']
    else:
        ageProwithBuy = probability['ageBig40WithBuy']
        ageProwithNotBuy=probability['ageBig40WithNotBuy']
    if income == 'high':
        incomeProWithBuy = probability['incomeHighWithBuy']
        incomeProWithNotBuy=probability['incomeHighWithNotBuy']
    elif income == 'middle':
        incomeProWithBuy = probability['incomeMiddleWithBuy']
        incomeProWithNotBuy=probability['incomeMiddleWithNotBuy']
    else:
        incomeProWithBuy = probability['incomeLowWithBuy']
        incomeProWithNotBuy=probability['incomeLowWithNotBuy']
    if student == 'yes':
        studentProWithBuy = probability['isStudentWithBuy']
        studentProWithNotBuy = probability['isStudentWithNotBuy']
    else:
        studentProWithBuy = probability['isNotStudentWithBuy']
        studentProWithNotBuy = probability['isNotStudentWithNotBuy']
    if credit == '好':
        creditProWithBuy = probability['creditGoodWithBuy']
        creditProWithNotBuy = probability['creditGoodWithNotBuy']
    else:
        creditProWithBuy = probability['creditJustsosoWithBuy']
        creditProWithNotBuy = probability['creditJustsosoWithNotBuy']
    willBuyPro=ageProwithBuy*incomeProWithBuy*studentProWithBuy*creditProWithBuy*probability['haveBuyCompute']
    willNotBuyPro=ageProwithNotBuy*incomeProWithNotBuy*studentProWithNotBuy*creditProWithNotBuy*probability['haveNotBuyCompute']
    print("预测会买的概率是：{0}".format(willBuyPro))
    print("预测不会买的概率是：{0}".format(willNotBuyPro))
    if(willBuyPro>willNotBuyPro):
        print('预测结果是：会买')
    else:
        print('预测结果是：不会买')


if __name__=='__main__':
    dataSet=getDataset()
    probability=caculataPro(dataSet)
    needPro=predic(probability)



