#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Utilities for tokenizing text, create vocabulary and so on"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import jieba
import gzip
import os
import re
import tarfile
import pandas as pd
import time

from sklearn.externals import joblib
from six.moves import urllib

from tensorflow.python.platform import gfile
import numpy as np
# Special vocabulary symbols - we always put them at the start.
_PAD = "<pad>"
_GO = "<go>"
_EOS = "<eos>"
_UNK = "<unknown>"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#将原始文章先清洗、分词保存以留备用
# Regular expressions used to tokenize.
#表明原文在处理时似乎是没有去掉标点的;其实原文只做了以下预处理；也没有去停用词
'''
特殊字符：去除特殊字符，如：“「，」,￥,…”；
括号内的内容：如表情符，【嘻嘻】，【哈哈】
日期：替换日期标签为TAG_DATE，如：***年*月*日，****年*月，等等
超链接URL：替换为标签TAG_URL；
删除全角的英文：替换为标签TAG_NAME_EN；
替换数字：TAG_NUMBER；
在对文本进行了预处理后，准备训练语料： 我们的Source序列，是新闻的正文，待预测的Target序列是新闻的标题。
我们截取正文的分词个数到MAX_LENGTH_ENC=120个词，是为了训练的效果正文部分不宜过长。标题部分截取到MIN_LENGTH_ENC = 30，即生成标题不超过30个词
原文：https://blog.csdn.net/rockingdingo/article/details/55224282 
'''


def basic_tokenizer(sentence):
  return sentence.strip().split(' ')
  # #南 都 讯 记!者 刘?凡 周 昌和 任 笑 一 继 推出 日 票 后 TAG_NAME_EN 深圳 今后 将 设 地铁 TAG_NAME_EN 头 等 车厢 TAG_NAME_EN 设 坐 票制
  # """Very basic tokenizer: split the sentence into a list of tokens."""
  # words = []
  # #将每一句的句首和句尾的空白字符(换行符)去掉，然后按空格分割
  # for space_separated_fragment in sentence.strip().split():
  #   print('space_separated_fragment',space_separated_fragment)
  #   #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表)
  #   words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  #   print('words',words)#words ['南', '都', '讯', '记', '!', '者', '刘', '?', '凡', '周', '昌和'],取其中某一步的words
  # return [w for w in words if w]#w不为空就返回words中的w,组合成列表sentence_split ['南', '都', '讯', '记', '!', '者', '刘', '?', '凡', '周', '昌和', '任', '笑', '一', '继', '推出',

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  # if not os.path.exists(vocabulary_path):
  if True:
      print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
      vocab = {}  #(词，词频)对
    # with gfile.GFile(data_path, mode="rb") as f:


      # counter = 0

      df=pd.read_parquet(data_path)
      for index,row in df.iterrows():
        line=row['split_text']
      # for line in f:
      #   counter += 1
        if index % 100000 == 0:
          print("  processing line %d" % index)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for word in tokens:
          # #分词后将每个词中的数字替换为0，如果开启normalize_digits
          # word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w

          #统计每个词以及出现的次数
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      #开始列表相加；字段按照值排序(逆序)后，返回键的列表dict.get(key,default=None)获取键对应的值,default -- 如果指定键的值不存在时，返回该默认值值。
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      #前面一步表示按词频从高到低排列，下一步表示如果词汇量大于50000，则取前50000个词汇
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      all_vocabs_id_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
      print("all_vocabs_id_dict",all_vocabs_id_dict)
      joblib.dump(all_vocabs_id_dict,vocabulary_path)#
      # return vocab_list,all_vocabs_id_dict


      # with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      #   for w in vocab_list:
      #     vocab_file.write(w + b"\n")






def word2id_func(text_path,word2id_save_path,all_vocabs_id_dict_path):
    word2id_dic={'<pad>': 0, '<go>': 1, '<eos>': 2, '<unknown>': 3}
    all_vocabs_id_dict=joblib.load(all_vocabs_id_dict_path)#
    data_df=pd.read_parquet(text_path)
    for index,each_row in data_df.iterrows():
        time_start = time.time()
        # print('开始处理第{}行数据'.format(index))
        line=each_row['split_text']
        # print('line',line)
        tokens =line.strip().split(' ')
        # print('tokens',tokens)
        # break
        for each_word in tokens:

          ######unk
          word2id_dic[each_word]=all_vocabs_id_dict.get(each_word, UNK_ID)

        if index==10000:
          time_end = time.time()
          print('10000条数据耗时：{} s'.format(time_end-time_start))
          print('总数据约耗时{} hour'.format(data_df.shape[0]/10000*(time_end-time_start)/3600 ))
          # break
    print(word2id_dic)
    joblib.dump(word2id_dic,word2id_save_path)#word2id.pkl
def id2word_func(word2id_read_path,id2word_save_path):
    word2id_dic=joblib.load(word2id_read_path, )#
    print('word2id_dic',word2id_dic)
    #id=3对应的都是unk
    # id2word = {v: k  for k, v in word2id_dic.items()}

    id2word={v:  k if v!=3 else _UNK for k, v in word2id_dic.items()  }

    # # #只需要对不是新词进行id2word转换，因为新词对id应的都是unk,与3对应的是一样的;然后把3对应的unk添上
    # id2word = {v: k  for k, v in word2id_dic.items() if v != 3 }
    # id2word[3]=_UNK

    joblib.dump(id2word, id2word_save_path)#id2word.pkl
    print(id2word)

def trainingSamples_func(text_path,word2id_read_path,trainingSamples_save_path):
    word2id_dic = joblib.load(word2id_read_path)#'word2id.pkl'
    print('word2id',word2id_dic)
    # print(word2id_dic['本文'])
    trainingSamples=[]
    data_df = pd.read_parquet(text_path)
    print(data_df.head())


    for index, each_row in data_df.iterrows():
        content_split = each_row['split_text'].strip().split(' ')#字符串后面一定要split变成列表
        # print('type(content_split)',type(content_split))
        # for each in content_split:
        #     print(each)
        label =each_row['label']
        #####unk
        content_split_id=[word2id_dic.get(word,UNK_ID) for word in content_split]

        line=[content_split_id,label]
        trainingSamples.append(line)
    joblib.dump(trainingSamples, trainingSamples_save_path)#'trainingSamples.pkl'
    print(trainingSamples)


def final_data(word2id_path,id2word_path,trainingSamples_path,all_data_save_path):
    all_data={}
    word2id=joblib.load(word2id_path)#'word2id.pkl'
    id2word=joblib.load(id2word_path)#'id2word.pkl'
    trainingSamples=joblib.load(trainingSamples_path)#
    all_data['word2id']=word2id
    all_data['id2word']=id2word
    all_data['trainingSamples']=trainingSamples
    joblib.dump(all_data,all_data_save_path)#'../../data/data_cleaned/hotel-vocabSize50000.pkl'


if __name__=='__main__':
    data_path='../../data/data_cleaned/fruit_split.parquet.gzip'
    save_path='chinese_fruit'

    # data_path = '../../data/data_cleaned/hotel_split.parquet.gzip'
    # save_path = 'chinese_hotel'


    # print('create_vocabulary')
    # create_vocabulary(vocabulary_path=os.path.join(save_path,'fruit_all_vocabs_id_dict.pkl'), data_path=data_path, max_vocabulary_size=50000,
    #                 tokenizer=None, normalize_digits=False)
    # print('go word2id')
    # word2id_func(text_path=data_path,
    #              word2id_save_path=os.path.join(save_path,'fruit_word2id.pkl'),
    #              all_vocabs_id_dict_path=os.path.join(save_path,'fruit_all_vocabs_id_dict.pkl'))
    # print('go id2word')
    # id2word_func(word2id_read_path=os.path.join(save_path,'fruit_word2id.pkl'),
    #              id2word_save_path=os.path.join(save_path,'fruit_id2word.pkl'))#{0: '<pad>', 1: '<go>', 2: '<eos>', 3: '<unknown>', 7629: '专程', 941: '成都', 4394: '绵阳
    print('go trainingSamples')
    trainingSamples_func(text_path=data_path,word2id_read_path=os.path.join(save_path,'word2id.pkl')
                         ,trainingSamples_save_path=os.path.join(save_path,'trainingSamples.pkl'))
    print('go final')
    final_data(word2id_path=os.path.join(save_path,'word2id.pkl'),
               id2word_path=os.path.join(save_path,'id2word.pkl'),
               trainingSamples_path=os.path.join(save_path,'trainingSamples.pkl'),
               all_data_save_path=os.path.join(save_path,'vocabSize50000.pkl'))

    #
    # p1='../../data/data_cleaned/hotel-vocabSize50000.pkl'
    # p2='all_vocabs_id_dict.pkl'
    p3='chinese_hotel/trainingSamples.pkl'
    trainingSamples=joblib.load(p3)
    print('trainingSamples',np.array(trainingSamples).shape)
    # all_vocabs_id_dict=joblib.load(p1)
    # print(len(all_vocabs_id_dict['word2id']),all_vocabs_id_dict['word2id'])
    '''
    28694 {'<pad>': 0, '<go>': 1, '<eos>': 2, '<unknown>': 3, '专程': 7629, '成都': 941, '绵阳': 4394, 
    '''
    # print(all_vocabs_id_dict['id2word'])
    # sample_length3=np.array(all_vocabs_id_dict['trainingSamples'][0:3])
    # for each in sample_length3:
    #     print(each[1])
    #
    #
    # print('all_vocabs_id_dict',len(all_vocabs_id_dict),all_vocabs_id_dict)

    # p3='../../nn_senti_analysis/data_processed/all_vocabs_id_dict.pkl'
    # print(joblib.load(p3))

    '''
[
    [list([13066, 10285, 3, 1187, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5527, 860, 3, 13476, 3854, 3, 48, 3, 233, 1630, 3, 3, 44, 3, 4987, 10285, 3, 3, 1378, 3, 78, 3, 13122, 1894, 3, 3, 3058, 308, 3, 3, 3, 3, 3, 118, 2459, 3, 257, 1351, 3, 20586, 1022, 3, 3, 13486, 3, 397, 3, 3, 1022, 3, 316, 3, 3464, 32, 3, 26542, 3, 3, 437, 3, 3, 1009, 3, 3, 1012, 3, 3, 3, 3, 1875, 3, 13, 733, 3, 1875, 3, 3, 3, 3, 595, 3, 3, 2836, 3, 8590, 3, 1773, 11021, 3593, 961, 12852, 9737, 3, 3, 233, 3, 2596, 159, 3, 3058, 308, 3, 1380, 2459, 3, 2596, 1659, 3, 3, 118, 3, 36, 3, 8744, 6985, 3, 3400, 10885, 3, 3, 108, 3, 3, 108, 3, 6046, 11439, 3, 6046, 11439, 3, 3, 3, 3, 1380, 2459, 3, 2596, 1659, 3, 3, 3, 95, 3, 11, 3, 3, 3, 3, 3, 118, 3, 3, 16884, 3, 3, 3, 3, 11, 3, 11, 3, 3, 291, 3, 3, 6398, 3, 3, 6398, 3, 32, 3, 3, 6398, 3, 3745, 3, 3, 3, 3, 3, 3, 18330, 3, 2836, 3, 8590, 3, 1773, 11021, 3593, 961, 12852, 9737, 3, 198, 3, 65, 3, 3, 3, 3, 718, 242, 3, 3, 3473, 3, 320, 664, 3, 921, 1146, 3, 13, 3, 118, 2459, 3, 3058, 308, 3, 3291, 1148, 3, 718, 242, 3, 3, 3, 3, 687, 3, 1658, 3, 3, 3, 5527, 3, 490, 3, 3])
  1]
 [list([608, 3, 3, 6132, 3, 3, 118, 3, 3, 1037, 695, 3, 2511, 3, 3, 13, 3, 3, 6132, 3, 11, 3, 3, 3, 3, 7877, 291, 3971, 3, 3, 6132, 3, 157, 3, 11, 3, 5863, 3, 3, 830, 3, 308, 3, 2574, 1655, 3, 3, 1054, 3, 3, 641, 3, 3, 3, 3, 3, 5863, 3, 3, 830, 3, 3, 6132, 3, 3, 118, 3, 12773, 3, 3, 32, 3, 595, 10321, 3, 11, 3, 3, 291, 3, 291, 3, 550, 22, 3, 1380, 797, 3, 3, 3056, 3, 100, 3, 1054, 3, 2574, 1925, 3, 3, 149, 3, 2574, 1655, 3, 26542, 3, 3, 3, 3, 641, 3, 3035, 11, 3, 291, 3, 3693, 643, 3, 3, 3, 3, 3, 3, 291, 3, 3058, 308, 3, 1158, 3, 21788, 3, 36, 76, 3, 8364, 3, 3, 3, 3, 3, 7948, 99, 3, 4208, 6713, 3, 3, 3, 3, 744, 299, 3, 6451, 907, 3, 3, 1894, 3, 2574, 1655, 3, 437, 3, 99, 99, 3, 3724, 3, 3, 550, 3, 3, 1870, 558, 3, 424, 3, 3, 1845, 4704, 3, 4208, 3211, 3, 24230, 1894, 3, 3, 108, 3, 3155, 3, 3, 3, 108, 3, 2597, 7956, 3, 4129, 761, 3, 761, 738, 3, 13930, 3, 7634, 3, 142, 159, 3, 784, 784, 3, 3, 3846, 2940, 3, 10, 256, 3, 3058, 308, 3, 1497, 8050, 3, 10, 3, 3, 95, 3, 3, 291, 3, 22951, 3, 3, 308, 3, 1497, 8050, 3, 25, 3, 3, 257, 1049, 3, 3, 27236, 7094, 3, 6819, 2109, 3, 756, 3, 385, 3, 3, 3, 3, 4208, 3211, 3, 2146, 3, 3, 31, 3, 21316, 3, 3, 3, 3, 3, 8174, 3, 3, 3, 3, 17475, 1800, 22, 1037, 3, 2761, 3, 3, 3, 1054, 3, 42, 3, 3, 3, 641, 3, 3, 291, 3, 36, 25, 3, 26542, 3, 3, 3, 3, 3, 3, 797, 11694, 3, 1611, 7862, 3, 44, 1380, 3, 3, 3, 2836, 3, 8590, 3, 6096, 3, 2836, 12852, 3, 3, 3, 3, 3, 3, 3, 3, 1380, 797, 3, 157, 3, 1569, 6369, 3, 3, 3, 3, 8174, 3, 3, 157, 3, 3035, 11, 3, 38, 3, 7854, 3, 3, 9745, 3, 3, 512, 3, 7854, 3, 3, 3, 10158, 3, 2574, 3, 3, 26542, 3, 3, 339, 3, 3, 42, 3, 1611, 7862, 3, 3, 44, 3, 3, 291, 3, 320, 664, 3, 17475, 1800, 22, 1037, 3, 1264, 3, 579, 3, 3, 3, 3, 7934, 10235, 3, 436, 3, 3, 2291, 9745, 3, 2574, 1655, 3, 579, 3, 3, 26423, 3, 3, 3])
  1]
 [list([3, 1080, 3, 938, 3, 3, 65, 3, 153, 3, 3, 26685, 3, 7948, 3, 2146, 3, 3, 3, 10, 3, 13, 3, 3, 3, 3, 44, 3, 10, 3, 4444, 1152, 3, 11, 3, 3, 2563, 3, 3058, 308, 3, 3, 3, 3, 44, 11, 3, 3, 797, 3, 3, 1779, 3, 3058, 308, 3, 1658, 201, 3, 433, 2764, 3, 3846, 2940, 3, 48, 3, 3, 3, 3, 2818, 299, 3, 3, 17233, 3, 320, 664, 3, 2761, 3, 1037, 3, 3700, 1800, 3, 477, 3, 3, 10, 3, 1213, 3854, 3, 433, 711, 3, 32, 3, 3, 3, 3, 3, 3, 2358, 1213, 3, 3])
  1]
]
    '''


