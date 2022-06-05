from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from analysis import getProcessList
from preprocess import process


# 使用测试集进行结果提交
def eval():
    uselessList, nanList, scaleList, strList = getProcessList("dataset/titanic_test.csv")
    train_data = process(uselessList, nanList, scaleList, strList, 'dataset/titanic_train.csv', True)
    test_data = process(uselessList, nanList, scaleList, strList, "dataset/titanic_test.csv", False)
    model_list = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(),
                  GaussianNB(), AdaBoostClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
    X, y = train_data[train_data.columns[1:]].values, train_data[train_data.columns[:1]].values.ravel()
    predict_X = test_data.values
    res_data = pd.read_csv('dataset/titanic_test.csv')
    for model in model_list:
        model.fit(X, y)
        res = model.predict(predict_X)
        predict_data = res_data[['PassengerId']]
        predict_data.insert(1, 'Survived', res)
        predict_data.to_csv('./result/' + str(model)[:-2] + '_submission.csv', index=False)


eval()
