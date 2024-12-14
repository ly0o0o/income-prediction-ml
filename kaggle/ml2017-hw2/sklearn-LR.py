# -*- coding: utf-8
# !/usr/bin/env python

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn import metrics

# 读取数据
X_train = pd.read_csv('X_train')
X_train = np.array(X_train)

Y_train = pd.read_csv('Y_train')
Y_train = np.array(Y_train).ravel()  # 使用ravel()将列向量转换为1维数组

# 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3)

# 标准化
nor = Normalizer()
X_train = nor.fit_transform(X_train)
X_test = nor.fit_transform(X_test)

# 训练模型
model = LogisticRegression(C=50)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

# 输出准确率
print("acc is :", metrics.accuracy_score(prediction, Y_test))
