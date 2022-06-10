from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import pandas as pd
from analysis import getProcessList
from preprocess import process


def opt_SVC_eval():
    uselessList, nanList, scaleList, strList = getProcessList("dataset/titanic_test.csv")
    train_data = process(uselessList, nanList, scaleList, strList, 'dataset/titanic_train.csv', True)
    test_data = process(uselessList, nanList, scaleList, strList, "dataset/titanic_test.csv", False)
    X, y = train_data[train_data.columns[1:]].values, train_data[train_data.columns[:1]].values.ravel()
    predict_X = test_data.values
    res_data = pd.read_csv('dataset/titanic_test.csv')
    opt_SVC = SVC(C=0.6, gamma=0.2, kernel='rbf')
    opt_SVC.fit(X, y)
    res = opt_SVC.predict(predict_X)
    predict_data = res_data[['PassengerId']]
    predict_data.insert(1, 'Survived', res)
    predict_data.to_csv('./result/opt_SVC_submission.csv', index=False)


def opt_adb_eval():
    uselessList, nanList, scaleList, strList = getProcessList("dataset/titanic_test.csv")
    train_data = process(uselessList, nanList, scaleList, strList, 'dataset/titanic_train.csv', True)
    test_data = process(uselessList, nanList, scaleList, strList, "dataset/titanic_test.csv", False)
    X, y = train_data[train_data.columns[1:]].values, train_data[train_data.columns[:1]].values.ravel()
    predict_X = test_data.values
    res_data = pd.read_csv('dataset/titanic_test.csv')
    opt_adb = AdaBoostClassifier(learning_rate=0.7, n_estimators=200)
    opt_adb.fit(X, y)
    res = opt_adb.predict(predict_X)
    predict_data = res_data[['PassengerId']]
    predict_data.insert(1, 'Survived', res)
    predict_data.to_csv('./result/opt_Adaboost_submission.csv', index=False)


opt_SVC_eval()
opt_adb_eval()
