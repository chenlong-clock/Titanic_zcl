# Cl Zhang's Titanic 监督学习综合实验
## Author
* Chenlong Zhang
## Catalog
* 问题及数据集简介
* 数据集的分析
* 数据预处理
* 模型的选择、训练和预测
* 模型效果的评价及解释
* 参数调优
* 总结
## 问题及数据集简介
_问题是以大家熟悉的泰坦尼克号为背景展开的，本次任务的目的就是构建一个可以根据乘客个人信息，如性别、年龄、船舱等级等来推测乘客是否生存的**分类模型**。_
## 数据集的分析
* **训练集：**
![](src/img/train_dataset.jpg)

文件名为“titanic_train.csv”,文件中的数据共有891行和12列，这代表本训练集共有891条数据，每条数据有12类属性。这12个属性的列名和含义如下所示：

| 属性名 | 属性含义 |
|----|----|
|PassengerId|乘客ID|
|Survived|获救情况（1为获救，0为未获救)|
|Pclass|船舱等级(1/2/3等舱位)|
|Name|乘客姓名|
|Sex|性别|
|Age|年龄|
|SibSp|乘客在船上的兄弟/姐妹个数|
|Parch|乘客在船上的父母与小孩个数|
|Ticket|船票编号|
|Fare|票价|
|Cabin|舱位|
|Embarked|登船港口|


* **测试集：**
![](src/img/test_dataset.jpg)

文件名为“titanic_test.csv”,文件中的数据共有418行和11列，与训练集相比少了Survived属性。

* **探索性数据分析**[[analysis.py]](analysis.py)
1. 描述性统计分析,确定无用数据![](src/img/analysis1.png)
经过分析，PassengerId, Name, Ticket, Cabin(其不好进行归类或处理为离散值，且存在大量缺失值)为干扰数据，可直接删去
2. 检查各个属性值中是否有缺失值![](src/img/check_nan.jpg)
经过分析，以上三个属性存在缺失值，故后续操作需要补全/直接删去
3. 检查各个属性值的类型：连续值（需要归一化）、字符值（需要处理为离散值）
通过describe函数直接观察:

需要转换为离散值的属性：Sex, Embarked

需要归一化的属性值：Age, Fare
4. 相关性分析![](src/img/heatmap.png)
通过相关性分析可以得出初步的结论：

与存活率成正相关: Fare, Parch

与存活率成负相关: Pclass, Sex, Age, SibSp, Embarked

需要说明的是Sex和 Embarked是由字符型数值转变的，因此其正负相关的判别与转换编码的形式有关，无实际意义
## 数据预处理
[[preprocess.py]](preprocess.py)

数据的预处理包括：数据的清洗、数据的采样、数据集拆分、特征选择、特征降维、特征编码、规范化等。

scaleList: ['Age', 'Fare'] 

* 无关特征的删除:
    * 有些项对预测分析是没有帮助的，可以直接删除。通过分析：uselessList: ['PassengerId', 'Name', 'Ticket', 'Cabin']直接删除 
* 缺失值填充
    * nanList: ['Age', 'Cabin', 'Embarked'] 存在缺失值，在训练时直接将缺失记录删除，在测试时将缺失值用上一条/下一条有效记录填充（需要说明的在测试时，这样的操作会导致测试精度下降）
* 编码转换
    * strList: ['Sex', 'Embarked'] “性别”和“登船港口” 列的属性值是字符型，需要进行转换，把这些类别映射到一个数值。（Sex:0, 1; Embarked:0, 1, 2）
* 数据缩放
    * scaleList: ['Age', 'Fare'] “年龄”和“票价”两列的属性值相对其他列明显太大，在模型训练中会影响模型的准确性，因此需要把这两列的数值变换到一个0-1之间的数值。b在本方案中使用了Z-score归一化

处理前后数据：
![](src/img/process.jpg)
## 模型的选择、训练和预测

要求：

**在学习过的分类算法中至少选择3种**分别进行模型的选择、训练和预测

Chenlong Zhang中使用的方法：
LogisticRegression

DecisionTreeClassifier

KNeighborsClassifier 

SVC

GaussianNBAdaBoostClassifier

RandomForestClassifier

GradientBoostingClassifier

使用了10折交叉验证计算，得到训练集和测试集的准确率，并画出混淆矩阵
准确率：
![](src/img/model_result.jpg)
混淆矩阵：
![](src/img/confusion_matrix.png)
## 模型效果的评价及解释

要求：

使用**多次交叉验证法**评价不同的模型；
要说明实验过程中是否有过拟合现象，若有，是如何处理的。


## 参数调优

大多数机器学习算法都包含大量的参数，使用最适合数据集的参数才能让机器学习算法发挥最大的效果。但各种参数的排列组合数量巨大，这时使用自动化参数调优就可以在很大程度上减少工作量并提升工作效率。下面介绍一种常用的参数调优方法——暴力搜索。

暴力搜索寻优：网格搜索为自动化调参的常见技术之一，Scikit-learn提供的GridSearchCV函数可以根据给定的模型自动进行交叉验证，通过调节每一个参数来跟踪评分结果。从本质上说，该过程代替了进行参数搜索时使用的for循环过程。

## 总结

