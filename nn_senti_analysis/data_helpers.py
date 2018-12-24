from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk
import numpy as np
import pickle
import random
import re
import jieba
import datetime;
import random;
module_path = os.path.dirname(__file__)#文件所在脚本的相对路径
#当前文件所在目录的上级目录
pre_dir=os.path.join(module_path,'..')#相对路径的上级路径
#自定义词典要放在并行化程序之前，否则不起作用
jieba.load_userdict(os.path.join(pre_dir,'data/self_define_dict.txt'))

#nltk.download()
padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    #batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def loadDataset(filename):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    return word2id, id2word, trainingSamples

def createBatch(samples):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    # batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    # max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        #将source进行反序并PAD值本batch的最大长度
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        #将target进行PAD，并添加END符号
        target = sample[1]
        # target=[target]
        # pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target)
        #batch.target_inputs.append([goToken] + target + pad[:-1])

    return batch

def getBatches(data, batch_size,training_flag=True):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    #训练之前，每个epoch之前都要进行样本的shuffle
    if training_flag:
        nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
        #设置当前时间为随机数种子，确保每一次shuffle都是不同的
        random.seed(nowTime)
        random.shuffle(data)
        # 先不shauffle看结果

    #
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

#添加停用词词典
with open(os.path.join(pre_dir,'data/stopWord.txt')) as stopwords:
    stopwords_set=set()
    for each in stopwords:
        stopwords_set.add(each.strip())
#print('stopwords_set',stopwords_set)
TAG_DATE_PATTERN='\d{1,}年\d{1,}月\d{1,}日|\d{1,}年\d{1,}月|\d{1,}月\d{1,}日|\d{1,}年|\d{1,}月|\d{1,}日'
TAG_NUMBER_PATTERN='\d{1,}'
TAG_URL_PATTERN='[a-zA-z]+://[^\s]*'

counts = 0
def cut_process(input_str):
    global counts
    input_str=re.sub(TAG_DATE_PATTERN,'TAG_DATE',input_str)
    input_str=re.sub(TAG_NUMBER_PATTERN,'TAG_NUMBER',input_str)
    input_str=re.sub(TAG_URL_PATTERN,'TAG_URL',input_str)


    str_split=jieba.cut(input_str.strip())
    cleaned_str_split=[each for each in str_split if each not in stopwords_set]

    if counts<3:
        counts=counts+1
 #       print('input', input_str)
        print('cleaned_str_split',cleaned_str_split)
    return cleaned_str_split
def sentence2enco(sentence, word2id,label=0):#label默认为0，此处只是为了程序不出错，预测时并没有用到label
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    #分词
    tokens =cut_process(sentence)
   # tokens = nltk.word_tokenize(sentence)

   #将每个单词转化为id
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #调用createBatch构造batch
    batch = createBatch([[wordIds, label]])
    return batch

if __name__=='__main__':
    # data_path = '../data/data_cleaned/fruit-vocabSize50000.pkl'
    data_path='../data/data_cleaned/hotel-vocabSize50000.pkl'#迁移学习时，词汇个数不一样维度就不一样
    batch_size=300
    word2id, id2word, trainingSamples = loadDataset(data_path)

    # print('word2id',word2id)
    # print('id2word',id2word)
    # print('trainingSamples',trainingSamples[0][0])
    print([id2word[each] for each in trainingSamples[0][0]])
    # batches = getBatches(trainingSamples, batch_size)
    # for index,each in enumerate(batches):
    #     print('index each.encoder_inputs_length',index,np.array(each.encoder_inputs).shape,len(each.encoder_inputs_length),each)

'''
word2id {'<pad>': 0, '<go>': 1, '<eos>': 2, '<unknown>': 3, '专程': 7629, '成都': 941, '绵阳': 4394, '第四届': 13921, '科博会': 13922, '万达': 4395, '广场':
id2word {0: '<pad>', 1: '<go>', 2: '<eos>', 3: '<unknown>', 7629: '专程', 941: '成都', 4394: '绵阳', 13921: '第四届', 13922: '科博会', 4395: '万达', 559: '广场', 48: '吃', 1333:
'''


'''
index each.encoder_inputs_length 0 300
index each.encoder_inputs_length 1 300
***
index each.encoder_inputs_length 32 300
index each.encoder_inputs_length 33 100
'''
'''
data_x_batchs=[
#      [
#         [1,1,3],
#         [3,2,0],
#         [5,2,0],
#         [5,2,0],
#         [0,7,0],
#
#      ],
#     [
#         [3,1,6],
#         [8,0,0],
#         [6,2,0],
#         [0,0,0],
#         [0,0,0],
#      ],
#    [
#         [1,8,3],
#         [9,2,0],
#        [1, 9, 3],
#         [5,2,0],
#         [0,0,0],
#
#      ],
#     [
#         [3,7,6],
#         [9,0,0],
#         [0,0,0],
#         [0,0,0],
#         [0,0,0],
#      ],
#
#     ]
#
# data_y_batch=[
#     [0,1],
#     [1,0],
#     [0,1],
#     [0,1],
# ]

'''