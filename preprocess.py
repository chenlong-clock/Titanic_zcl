import pandas as pd


def process(dropList, nanList, scaleList, strList, pth, is_train):
    data = pd.read_csv(pth)
    print("Properties to peprocess:")
    print("uselessList:", dropList, "\nnanList:", nanList, "\nscaleList:", scaleList, "\nstrList:", strList)
    print("-" * 50, "\nSource Data:")
    print(data.head())
    data.drop(dropList, axis=1, inplace=True)  # 删去去无用属性
    if is_train:
        data.dropna(inplace=True)  # 删除缺失值
    else:  # 若为测试，不能删除缺失值，则只能填充
        data.fillna(method='backfill', inplace=True)
        data.fillna(method='ffill', inplace=True)
    for idx in scaleList:  # 使用Z-score标准化
        data[idx] = (data[idx] - data[idx].mean()) / data[idx].std()
    for idx in strList:  # 将字符型数值替换为离散性数值 如: Sex: male, female -> 0. 1
        data[idx].replace(list(set(data[idx].values)), range(len(set(data[idx].values))),
                          inplace=True)
    print("Processed Data:\n", data.head())
    print("-" * 50)
    return data
