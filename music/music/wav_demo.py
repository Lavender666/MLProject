# --encoding:utf-8--
u'''这个是一个测试wav格式文件MFCC特征属性读取的案例'''

import scipy.io.wavfile
from python_speech_features import mfcc
import numpy as np

# 双声道数据
path = "../data/1KHz-stero.wav"
# 单声道的数据
path = "../data/童年.wav"
# path = "../data/擦肩而过.wav"
# path = "../data/123.wav"

(rate, data) = scipy.io.wavfile.read(path)
print("频率:%d" % rate)
print("=" * 5 + "数据" + "=" * 5)
print(data)
print("shape:", end="")
print(data.shape)
print("=" * 15)

# MFCC特征属性获取
# numcep：给定MFCC在进行倒谱的时候，每个抽取出来的帧使用多少个特征属性来表示特征，默认是13个
mfcc_feat = mfcc(data, samplerate=rate, numcep=5, nfft=2048)
print("=" * 5 + "特征属性数据" + "=" * 5)
print(mfcc_feat)
print("shape:", end="")
print(mfcc_feat.shape)
print("=" * 15)