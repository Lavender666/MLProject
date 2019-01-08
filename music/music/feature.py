# --encoding:utf-8--
u'''主要提供一些特征属性操作的方法'''

import os
import glob
from pydub import AudioSegment
from python_speech_features import mfcc
import numpy as np
import pandas as pd
import scipy.io.wavfile

music_info_csv_file_path = "../data/music_info.csv"
music_index_label_path = "../data/music_index_label.csv"
music_feature_file_path = "../data/music_feature.csv"
music_audio_dir = "../data/music/*.mp3"


def extract(file):
    """
    这个函数的主要功能就是从对应文件路径的音乐文件中读取MFCC提取出来的特征属性
    并以array返回
    """
    # 1. 获取文件所属的格式
    items = file.split('.')
    file_format = items[-1].lower()
    file_name = file[:-(len(file_format) + 1)]

    # 2. 判断文件格式，如果是非wav格式的数据，进行转码操作
    if file_format != 'wav':
        try:
            # 2.1 进行数据的读取操作
            song = AudioSegment.from_file(file, format=file_format)
            # 2.2 输出的wav格式文件构建
            file = file_name + ".wav"
            # 2.3 输出
            song.export(out_f=file, format='wav')
        except Exception as e:
            print("Error, " + file_format + " to wav throw exception. msg:=", end="")
            print(e)

    # 3. 进行数据读取操作
    try:
        # 3.1 读取wav格式文件对应的数据, eg: (采样频率，数据)
        (rate, data) = scipy.io.wavfile.read(file)

        # 3.2 使用MFCC提取特征属性
        mfcc_feat = mfcc(data, rate, numcep=13, nfft=2048)

        # 3.3 将mfcc的结果数据进行转置操作
        mm = np.transpose(mfcc_feat)

        # 3.4 求每行/每个角度/每个特征属性的均值
        mf = np.mean(mm, axis=1)

        # 3.5 协方差矩阵
        cf = np.cov(mm)

        # 3.6 计算最终结果， 最终的特征数量为: 13 + 13 + 12 + 11 + 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 104
        result = mf
        for i in range(mm.shape[0]):
            # 获取协方差矩阵上的对角线上的内容，添加到result中
            result = np.append(result, np.diag(cf, i))

        # 3.7 结果返回
        return result
    except Exception as e:
        print(e)
    finally:
        # 如果输出文件不是wav格式的文件，那么进行删除临时文件
        if file_format != 'wav':
            os.remove(file)

def extract_and_export():
    """
    进行特征属性的抽取以及特征工程后的数据输出
    """
    # 1. 读取csv文件中的音乐名称以及音乐类型
    df = extract_label()
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t: (t[0], t[1]), name_label_list))
    labels = set(name_label_dict.values())
    label_index_dict = dict(zip(labels, np.arange(len(labels))))

    # 2.获取文件夹下所有的音乐文件
    all_music_files = glob.glob(music_audio_dir)
    # 对所有音乐文件进行排序操作
    all_music_files.sort()

    # 3.遍历进行特征工程
    loop_count = 0
    flag = True
    # 构建最终返回对象
    all_mfcc = np.array([])
    for file_name in all_music_files:
        print("开始处理:" + file_name)
        # 1. 获取文件对应的列名称
        music_name = file_name.split("\\")[-1].split(".")[-2].split("-")[-1]
        music_name = music_name.strip()
        if music_name in name_label_dict:
            # 获取标签对应的index索引值
            label_index = label_index_dict[name_label_dict[music_name]]

            # 获取音乐文件的特征属性
            ff = extract(file_name)

            # 将标签id添加到特征属性中
            ff = np.append(ff, label_index)

            # 追加到最终集合中
            if flag:
                all_mfcc = ff;
                flag = False
            else:
                all_mfcc = np.vstack([all_mfcc, ff])
        else:
            print("没法处理:" + file_name +"; 原因是：找不到对应的label")

        print("loooping----%d" % loop_count)
        print("all_mfcc.shape:", end="")
        print(all_mfcc.shape)
        loop_count += 1

    # 4. 进行数据保存
    label_index_list = []
    for k in label_index_dict:
        label_index_list.append([k, label_index_dict[k]])
    pd.DataFrame(label_index_list).to_csv(music_index_label_path, header=None, index=False, encoding='utf-8')
    pd.DataFrame(all_mfcc).to_csv(music_feature_file_path, header=None, index=False, encoding='utf-8')
    print("标签数量为:%d" % len(label_index_list))
    return all_mfcc

def extract_label():
    """
    提取标签信息
    """
    data = pd.read_csv(music_info_csv_file_path)
    data = data[["name", "tag"]]
    return data

def fetch_index_label():
    """
    从文件中读取index和label之间的映射关系，并返回dict
    """
    data = pd.read_csv(music_index_label_path, header=None, encoding='utf-8')
    name_label_list = np.array(data).tolist()
    index_label_dict = dict(map(lambda t: (t[1], t[0]), name_label_list))
    return index_label_dict

def extract_all(audio_dir):
    """
    提取文件中的所有音乐文件数据，并返回最终的MFCC提取的特征属性
    """
    # 获取文件夹下所有的音乐文件
    all_music_files = glob.glob(audio_dir)
    # 对所有音乐文件进行排序操作
    all_music_files.sort()
    # 构建最终返回对象
    all_mfcc = np.array([])

    loop_count = 0

    flag = True
    for file_name in all_music_files:
        # 提取单个文件的数据
        ff = extract(file_name)

        # 追加到最终集合中
        if flag:
            all_mfcc = ff;
            flag = False
        else:
            all_mfcc = np.vstack([all_mfcc, ff])

        print("loooping----%d" % loop_count)
        print("all_mfcc.shape:", end="")
        print(all_mfcc.shape)
        loop_count += 1

    # 返回最终结果
    return all_mfcc
