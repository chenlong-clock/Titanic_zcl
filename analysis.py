import pandas as pd
import matplotlib.pyplot as plt
import os

train_data = pd.read_csv("dataset/titanic_train.csv")

# 先查看数据的大致信息
print(train_data.describe())
print(train_data.head())

train_data['Survived']