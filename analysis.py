import pandas as pd
<<<<<<< HEAD
import seaborn as sns
import matplotlib.pyplot as plt


def getProcessList(pth) -> (list, list, list, list):
    train_data = pd.read_csv(pth)
    # 1.描述性统计分析,确定无用数据
    # print("Step1: brief description\n")
    # print(train_data.describe())
    # print(train_data.head())
    uselessList = ['PassengerId', 'Name', 'Ticket', 'Cabin']  # 前三个属于无用的数据，对于Cabin来说，其不好进行归类或处理为离散值，且存在大量缺失值，故删去。
    print('-' * 50)

    # 2.检查各个属性值中是否有缺失值
    print("step2: check nans\n")
    nanList = []
    for idx in train_data.columns:
        if train_data[idx].hasnans:
            print(idx)
            nanList.append(idx)
    # print('_' * 50)

    # 3.检查各个属性值的类型：连续值（需要归一化）、字符值（需要处理为离散值）
    print("step3: check value type\n")
    for idx in train_data.columns:
        print(train_data[idx].describe())
    print("\n")
    strList = ['Sex', 'Embarked']  # 需要转换为离散值的属性：Sex, Embarked
    scaleList = ['Age', 'Fare']  # 需要归一化的属性值：Age, Fare
    # 4.相关性分析
    train_data.drop(uselessList, axis=1, inplace=True)  # 把无用的数据先剔除
    for idx in strList:  # 需要先将字符型数值替换为离散性数值才能进行相关性分析
        train_data[idx].replace(list(set(train_data[idx].values)), range(len(set(train_data[idx].values))),
                                inplace=True)

    sns.heatmap(train_data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size': 20})
    fig = plt.gcf()
    fig.set_size_inches(18, 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    # 通过相关性分析可以得出初步的结论：
    # 与存活率成正相关: Fare, Parch
    # 与存活率成负相关: Pclass, Sex, Age, SibSp, Embarked
    return uselessList, nanList, scaleList, strList


getProcessList("dataset/titanic_train.csv")
=======
import matplotlib.pyplot as plt
import os

train_data = pd.read_csv("dataset/titanic_train.csv")

# 先查看数据的大致信息
print(train_data.describe())
print(train_data.head())

train_data['Survived']
>>>>>>> 2933a15a304f351c314b6665baf73fcc43630f69
