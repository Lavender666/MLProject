# -*- encoding:utf-8 -*-
u'''这个主要进行一下jieba工具的操作; jieba安装: pip install jieba'''

import jieba
import jieba.posseg

# jieba分词的重点在于：自定义词典
# 词典格式为: 单词 词频(可选的) 词性(可选的)
# 词典构建方式：一般都是基于jieba分词之后的效果进行人工干预的
jieba.load_userdict('word.txt')

# 分词
# 长文本采用精准的分隔模式
cut1 = jieba.cut("北风网是一家具有培训资质的IT培训公司；公司总部在越秀大厦附近")
print(" / ".join(cut1))

cut2 = jieba.cut("北风网是一家具有培训资质的IT培训公司；公司总部在越秀大厦附近", HMM=False)
print(" / ".join(cut2))

# 短文本可以考虑使用全分隔模式
cut3 = jieba.cut("北风网是一家具有培训资质的IT培训公司；公司总部在越秀大厦附近", cut_all=True)
print(" / ".join(cut3))

# 词性的获取
# 一般情况下，在短文本处理过程中，有可能还需要考虑词性；并且还可能将分隔好的单词进行组合
cut4 = jieba.posseg.cut("北风网是一家具有培训资质的IT培训公司；公司总部在越秀大厦附近")
for item in cut4:
    print(item.word, end="<====>")
    print(item.flag)

