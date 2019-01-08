# --encoding:utf-8--
u'''将mp3格式数据转换为wav的'''

from pydub import AudioSegment

path = "../data/童年.mp3"
song = AudioSegment.from_file(file=path, format='mp3')
# 可以只转换一部分数据
song = song[: 30 * 1000] # 这个表示获取前30秒的数据
song.export(out_f='../data/123.wav', format='wav')
