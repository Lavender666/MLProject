# --encoding:utf-8--
u'''主要提供一些SVM算法的构建及应用'''

import numpy as np
import pandas as pd
import random
import feature, acc
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib

default_music_csv_file_path = '../data/music_feature.csv'
default_model_file_path = '../data/music_model.pkl'

# 加载label和标签的值的映射
index_label_dict = feature.fetch_index_label()


def poly(X, Y):
    """
    进行svm模型训练，并返回最终构建好的模型及在训练集上的正确率
    """
    # 模型构建
    clf = svm.SVC(kernel='poly', C=1, probability=True,decision_function_shape = 'ovo', random_state=0)
    # 模型训练
    clf.fit(X, Y)

    # 使用模型预测训练集得到正确率
    res = clf.predict(X)
    resTrain = acc.get(res, Y)

    return clf, resTrain


def fit_dump_model(train_percentage=0.7, fold=1, music_csv_file_path=None, model_out_f=None):
    """
    进行fold次训练，将准确率最高的模型输出到文件中
    train_percentage: 训练过程中，训练数据集的比例，范围: (0,1)
    fold: 训练的次数, 必须为大于0的整数
    music_csv_file_path: 数据存储文件路径
    NOTE: 使用训练集上的准确率和测试集上的准确率之间的和作为最终的准确率的评定指标; 计算公式为: source = 0.35*train_source + 0.65*test_source
    """
    # 1. 数据读取
    if not music_csv_file_path:
        music_csv_file_path = default_music_csv_file_path
    data = pd.read_csv(music_csv_file_path, sep=',', header=None, encoding='utf-8')

    # 2. 进行循环处理
    max_train_source = None
    max_test_source = None
    max_source = None
    best_clf = None
    flag = True
    for index in range(1, int(fold) + 1):
        # 2.1 进行数据抽取/数据分隔
        shuffle_data = shuffle(data)
        X = shuffle_data.T[:-1].T
        Y = np.array(shuffle_data.T[-1:])[0]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_percentage)


        # 2.2 模型训练
        (clf, train_source) = poly(x_train, y_train)

        # 2.3 获取得到测试集正确率, 并计算最终模型的正确率
        y_predict = clf.predict(x_test)
        test_source = acc.get(y_predict, y_test)
        source = 0.35 * train_source + 0.65 * test_source

        # 2.4 将最大的模型保存
        if flag:
            max_source = source
            max_train_source = train_source
            max_test_source = test_source
            best_clf = clf
            flag = False
        else:
            if max_source < source:
                max_source = source
                max_train_source = train_source
                max_test_source = test_source
                best_clf = clf

        # 2.5 将source输出
        print("第%d次训练，测试集上正确率为: %.2f, 训练集上正确率为：%.2f, 加权平均正确率为: %.2f" % (index, train_source, test_source, source))

    # 3. 进行最优模型输出
    print("最优模型效果：测试集上正确率为: %.2f, 训练集上正确率为：%.2f, 加权平均正确率为: %.2f" % (max_train_source, max_test_source, max_source))
    print("*" * 5 + "最优模型" + "*" * 5)
    print(best_clf)
    print("*" * 15)

    # 4. 模型输出
    if not model_out_f:
        model_out_f = default_model_file_path
    joblib.dump(best_clf, model_out_f)


def load_model(model_f=None):
    """
    模型加载, 根据给定路径加载模型
    """
    if not model_f:
        model_f = default_model_file_path
    clf = joblib.load(model_f)
    return clf


def internal_cross_validation(X, Y):
    """
    进行交叉验证，得到最优解
    """
    # 1. 给定训练参数
    parameters = {
        'kernel': ('linear', 'rbf', 'poly'),
        'C': [0.1, 1],
        'probability': [True, False],
        'decision_function_shape': ['ovo', 'ovr']
    }
    # 2. 模型构建及训练
    clf = GridSearchCV(svm.SVC(random_state=0), param_grid=parameters, cv=5)
    print("开始进行最优参数构建")
    clf.fit(X, Y)
    # 3. 模型最优参数输出
    print("最优参数:", end="")
    print(clf.best_params_)
    print("最优模型的准确率:", end="")
    print(clf.best_score_)


def cross_validation(music_csv_file_path=None, data_percentage=0.7):
    """
    进行交叉验证
    music_csv_file_path: 数据存储的文件路径
    data_percentage: 进行交叉验证的数据量百分比，默认使用原始数据的70%进行验证；取值范围: (0,1)
    """
    # 1. 加载数据
    if not music_csv_file_path:
        music_csv_file_path = default_music_csv_file_path
    print("开始读取数据:" + music_csv_file_path)
    data = pd.read_csv(music_csv_file_path, sep=',', header=None, encoding='utf-8')

    # 2. 数据抽取并转换
    sample_fact = 0.7
    if isinstance(data_percentage, float) and 0 < data_percentage < 1:
        sample_fact = data_percentage
    data = data.sample(frac=sample_fact).T
    X = data[:-1].T
    Y = np.array(data[-1:])[0]

    # 3. 交叉验证
    internal_cross_validation(X, Y)


def fetch_predict_label(clf, X):
    """
    进行模型训练，返回模型的标签
    """
    label_index = clf.predict(X)
    label = index_label_dict[label_index[0]]
    return label
