# --encoding:utf-8--

import svm

# 一、模型训练
# 1. 交叉验证
# svm.cross_validation(data_percentage=0.9)

# 2. 根据交叉验证修改代码参数后进行模型训练并输出
# svm.fit_dump_model(train_percentage=0.9, fold=100)

# # 4. 模型调用进行预测
import feature
# path = "../data/test/孙燕姿 - 我也很想他 - 怀旧.mp3"
path = "../data/test/Maize - I Like You-浪漫.mp3"
# path = "../data/test/Lasse Lindh - Run To You.mp3" # 清新

music_feature = feature.extract(path)
clf = svm.load_model()
label = svm.fetch_predict_label(clf, music_feature)
print("预测标签为:%s" % label)
