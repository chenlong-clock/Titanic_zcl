from analysis import getProcessList
from preprocess import process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    # 1.数据集的分析
    print("Analysis:")
    uselessList, nanList, scaleList, strList = getProcessList("dataset/titanic_train.csv")  # 通过分析获得需要处理的属性的列表
    # 2.数据预处理
    print("Preprocess:")
    train_data = process(uselessList, nanList, scaleList, strList, 'dataset/titanic_train.csv', True)  # 对这些属性进行预处理
    train, val = train_test_split(train_data, random_state=42, test_size=0.2)
    train_X, train_Y = train[train.columns[1:]].values, train[train.columns[:1]].values.ravel()
    val_X, val_Y = val[val.columns[1:]].values, val[val.columns[:1]].values.ravel()
    # 3.模型的训练与评估
    print("Train models :")
    model_list = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(),
                  GaussianNB(), AdaBoostClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
    res_dict = {}
    for model in model_list:
        train_acc = np.mean(cross_val_score(model, train_X, train_Y))
        print(model, ':\nTrain Acc:', train_acc, end='\t')
        model.fit(train_X, train_Y)
        val_acc = model.score(val_X, val_Y)
        print("Val Acc:", val_acc)
        res_dict[str(model)[:-2]] = np.asarray([train_acc, 0, val_acc, 0])
    f, ax = plt.subplots(3, 3, figsize=(12, 10))
    y_pred = cross_val_predict(SVC(kernel='rbf'), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[0, 0], annot=True, fmt='2.0f')
    ax[0, 0].set_title('SVC')
    y_pred = cross_val_predict(AdaBoostClassifier(), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[0, 1], annot=True, fmt='2.0f')
    ax[0, 1].set_title('Adaboost')
    y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[0, 2], annot=True, fmt='2.0f')
    ax[0, 2].set_title('KNN')
    y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[1, 0], annot=True, fmt='2.0f')
    ax[1, 0].set_title('RandomForests')
    y_pred = cross_val_predict(LogisticRegression(), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[1, 1], annot=True, fmt='2.0f')
    ax[1, 1].set_title('Logistic Regression')
    y_pred = cross_val_predict(DecisionTreeClassifier(), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[1, 2], annot=True, fmt='2.0f')
    ax[1, 2].set_title('Decision Tree')
    y_pred = cross_val_predict(GaussianNB(), train_X, train_Y, cv=10)
    sns.heatmap(confusion_matrix(train_Y, y_pred), ax=ax[2, 0], annot=True, fmt='2.0f')
    ax[2, 0].set_title('Naive Bayes')
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
    # 通过K折交叉验证之后进行测试，观察到决策树和梯度提升算法有一定的过拟合情况:
    # 对于决策树算法，画出决策树，发现，当决策树的深度特别深特别深以至于叶子节点中的对象只剩下一个或者很少，导致决策树的模型过于复杂，容易造成过拟合问题，泛化能力下降。因此可以通过预剪枝解决此问题。
    # 对于梯度提升算法，其原因与决策树类似。
    # DTree = DecisionTreeClassifier()
    # DTree.fit(train_X, train_Y)
    # plot_tree(DTree)
    # plt.show()
    # 4.模型参数调优
    print("-" * 50, "\nGrid Search:")
    # 1) 决策树算法调优
    print("1)DecisionTree:")  # 对于决策树算法，主要的超参数有max depth, max leaf nodes  criterion
    max_depth = range(1, 20)
    max_leaf_nodes = range(10, 500, 10)
    criterion = ["gini", "entropy"]
    hyper = {'max_depth': max_depth, 'max_leaf_nodes': max_leaf_nodes, 'criterion': criterion}
    GS = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=hyper, verbose=True, n_jobs=8)
    GS.fit(train_X, train_Y)
    print("Best Params: ", GS.best_params_)
    print("Best Train ACC:", GS.best_score_)
    val_acc = GS.score(val_X, val_Y)
    print("Val ACC:", val_acc)
    res_dict['DecisionTreeClassifier'][1] = GS.best_score_
    res_dict['DecisionTreeClassifier'][3] = val_acc

    # 2) KNN算法调优
    print("2)KNN:")  # 对于决策树算法，主要的超参数有max depth, max leaf nodes  criterion
    n_neighbors = range(1, 20)
    hyper = {'n_neighbors': n_neighbors}
    GS = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyper, verbose=True, n_jobs=8)
    GS.fit(train_X, train_Y)
    print("Best Params: ", GS.best_params_)
    print("Best Train ACC:", GS.best_score_)
    val_acc = GS.score(val_X, val_Y)
    print("Val ACC:", val_acc)
    res_dict['KNeighborsClassifier'][1] = GS.best_score_
    res_dict['KNeighborsClassifier'][3] = val_acc

    # 3) SVM调优
    print("3)SVC:")  # 对于SVM算法，主要的超参数有C, gamma和核函数
    C = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kernel = ['rbf', 'linear', 'sigmoid']
    hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}
    GS = GridSearchCV(estimator=SVC(), param_grid=hyper, verbose=True, n_jobs=8)
    GS.fit(train_X, train_Y)
    print("Best Params: ", GS.best_params_)
    print("Best Train ACC:", GS.best_score_)
    val_acc = GS.score(val_X, val_Y)
    print("Val ACC:", val_acc)
    res_dict['SVC'][1] = GS.best_score_
    res_dict['SVC'][3] = val_acc

    # 4) Adaboost调优
    print("4)Adaboost:")
    n_estimators = list(range(100, 1000, 100))
    learning_rate = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    hyper = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
    GS = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hyper, verbose=True, n_jobs=8)
    GS.fit(train_X, train_Y)
    print("Best Params: ", GS.best_params_)
    print("Best Train ACC:", GS.best_score_)
    val_acc = GS.score(val_X, val_Y)
    print("Val ACC:", val_acc)
    res_dict['AdaBoostClassifier'][1] = GS.best_score_
    res_dict['AdaBoostClassifier'][3] = val_acc

    # 5) 随机森林调优
    print("5)RandomForest:")
    n_estimators = range(100, 1000, 100)
    hyper = {'n_estimators': n_estimators}
    GS = GridSearchCV(estimator=RandomForestClassifier(), param_grid=hyper, verbose=True, n_jobs=8)
    GS.fit(train_X, train_Y)
    print("Best Params: ", GS.best_params_)
    print("Best Train ACC:", GS.best_score_)
    val_acc = GS.score(val_X, val_Y)
    print("Val ACC:", val_acc)
    res_dict['RandomForestClassifier'][1] = GS.best_score_
    res_dict['RandomForestClassifier'][3] = val_acc

    # 6) 梯度提升算法调优
    print("6)GradientBoosting:")
    learning_rate = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    n_estimators = range(100, 1000, 100)
    hyper = {'learning_rate': learning_rate, 'n_estimators': n_estimators}
    GS = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=hyper, verbose=True, n_jobs=8)
    GS.fit(train_X, train_Y)
    print("Best Params: ", GS.best_params_)
    print("Best Train ACC:", GS.best_score_)
    val_acc = GS.score(val_X, val_Y)
    print("Val ACC:", val_acc)
    res_dict['GradientBoostingClassifier'][1] = GS.best_score_
    res_dict['GradientBoostingClassifier'][3] = val_acc

    _, ax = plt.subplots(2, 4, sharex='all', sharey='all')
    plt.ylim(0.6, 0.9)
    plt.xticks(range(4), ['Train Acc', 'Opt:Train Acc', 'Val Acc', 'Opt:Val Acc'])
    ax = [i for arr in ax for i in arr]
    for idx, n in enumerate(res_dict):
        ax[idx].set_title(n + ' Result')
        ax[idx].bar([0, 2], res_dict[n][0:4:2], color='r')
        ax[idx].bar([1, 3], res_dict[n][1:4:2], color='b')
        for a, b in zip(range(4), res_dict[n]):
            ax[idx].text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    # plt.show()


if __name__ == '__main__':
    main()
