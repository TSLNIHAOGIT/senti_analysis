# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import numpy as np
from nn_senti_analysis.data_helpers import loadDataset,getBatches, sentence2enco
from tqdm import tqdm
from sklearn.externals import  joblib
'''
（1）padding:dynamic处理变长序列时，取最大长度序列，不足的序列补0；
 (2) mask:设置sequence_length，这样输出时补零的time_step部分输出也为0
（3）此时计算cost不是取最后一个time_step的hidden_state而是取最后一个不为零的：cost这里其实是不用mask的，因为label和prdiciton都是0
（4）预测的时候，即训练结束后，放入softmax分类时，不是取最后一个time_step的hidden_state而是取最后一个不为零的
'''

# ssl._create_default_https_context = ssl._create_unverified_context
#
# tf.set_random_seed(1)  # set random seed

data_path='../data/data_cleaned/hotel-vocabSize50000.pkl'

word2id, id2word, trainingSamples = loadDataset(data_path)

# 导入数据
# mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


#
# data_x_batch=[
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

# (batchsize,time_step,vec_size)=(4,5,3)

# hyperparameters
lr = 0.001  # learning rate

training_iters = 100000  # train step 上限
batch_size = 400

# n_inputs = 3  # MNIST data input(img shape:28*28)
# n_steps = 5  # time steps

n_hidden_units = 300  # neurons in hidden layer
n_classes = 2  # MNIST classes(0-9 digits)
keep_prob = tf.placeholder(tf.float32)
# LSTM layer 的层数
layer_num = 3
embedding_size=300 #n_hidden_units与embedding_size的关系;两者大小相等。

# x y placeholder
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) #(4,5,3)
# y = tf.placeholder(tf.float32, [None, n_classes])  #(4, 2)



encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')

pretrained_word2vec=np.float32(joblib.load('data_processed/pretrained_word2vec_list.pkl'))
# pretrained_word2vec=tf.placeholder(tf.float32,[None, None], name='pretrained_word2vec')
# encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

#embedding 被不断的训练，开始时是随机初始化，然后不断的调用之前训练好的
#词表总共有50000个词汇，编号从0-50000（0，1，2，3分别为padToken, goToken, eosToken, unknownToken）
#因此embedding0-50000的词向量要一一对应
#初始化embedding为常量时不需要指定大小
embedding = tf.get_variable('embedding',initializer=pretrained_word2vec)

#不初始化的embedding
# embedding = tf.get_variable('embedding', [len(word2id), embedding_size])



encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, encoder_inputs)

# batch_size = tf.placeholder(tf.int32, [], name='batch_size')
# keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

decoder_targets = tf.placeholder(tf.int32, [None,], name='decoder_targets')
print('decoder_targets 0',decoder_targets)

label_one_hot=tf.one_hot(decoder_targets,2)
print('decoder_targets 1',decoder_targets)
# decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
# # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。用一个sequence_mask的例子来说明起作用
# #  tf.sequence_mask([1, 3, 2], 5)
# #  [[True, False, False, False, False],
# #  [True, True, True, False, False],
# #  [True, True, False, False, False]]
# max_target_sequence_length = tf.reduce_max(decoder_targets_length, name='max_target_len')
# mask = tf.sequence_mask(decoder_targets_length, max_target_sequence_length, dtype=tf.float32, name='masks')



# 对weights biases初始值的定义
weights = {
    # shape(28,128)
    # "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape(128,10)
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape(128,)
    # "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # shape(10,)
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

#计算序列中padding后，返回原始序列长度
def length(sequence):
      used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
      length = tf.reduce_sum(used, 1)
      length = tf.cast(length, tf.int32)
      return length

#计算预测值时会用到，
def last_relevant(output, length):
      batch_size = tf.shape(output)[0]
      max_length = tf.shape(output)[1]
      out_size = int(output.get_shape()[2])
      index = tf.range(0, batch_size) * max_length + (length - 1)
      flat = tf.reshape(output, [-1, out_size])
      relevant = tf.gather(flat, index)
      return relevant


#用于分类时是不是用mask结果都是一样的
def cost_self_define(target,prediction):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(length, tf.float32)
    return tf.reduce_mean(cross_entropy)



def RNN(X, weights, biases):
    # hidden layer for input to cell
    # 此处相当于增加了全连接层，此处去掉，直接加多层lstm
    ###################################################################################
    # 原始的X是3维数据，我们需要把它变成2维数据才能使用weights的矩阵乘法
    # X==>(128 batch * 28 steps, 28 inputs)
    # X = tf.reshape(X, [-1, n_inputs])
    #
    # # X_in = W*X+b
    # X_in = tf.matmul(X, weights["in"]) + biases["in"]
    # # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    # X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # # cell
    # ####################################################################################
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X,sequence_length=length(X), initial_state=init_state, time_major=False)


    #计算序列padd过后的序列有效长度
    X_length=length(X)
    # MultiRNNCel
    ####################################################################################
    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    # lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden_units)  #cell可以选择lstm也可以用gru
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size],
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    outputs, final_state_ = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, sequence_length=X_length,initial_state=init_state, time_major=False)
    #final_state估计是（layer_num,(cell_state,hidden_state)）
    print('outputs',outputs)
    #针对有padding时的state
    last_relevant_state = last_relevant(outputs, X_length)
    print('last_relevant_state ',last_relevant_state)

    # #这两个h_state都是最后一个state,这是没有padding的时候
    # final_state = outputs[:, -1, :]  # 或者 final_state = state[-1][1]
    #
    # print('outputs.shape',outputs.shape)#outputs.shape (128, 28, 128)
    #
    # print('final_state',type(final_state))
    # print('final_state[0]',final_state[0][0].shape)
    # print('final_state[2]',final_state[2][0].shape)

    # 方法1
    results = tf.matmul(last_relevant_state, weights['out']) + biases['out']


    # # 方法2
    # # hidden layer for output as the final results
    # ####################################################################################
    # # 把outputs变成列表[(batch,outputs)...]*steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # # 选取最后一个output
    # results = tf.matmul(outputs[-1], weights["out"]) + biases["out"]

    return results

print('encoder_inputs_embedded',encoder_inputs_embedded)
logits = RNN(encoder_inputs_embedded, weights, biases)

#加入last_relavent有问题
pred = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_one_hot))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# Evaluate mode
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    for each in tf.all_variables():
        print('each',each)
    # Run the initializer
    sess.run(init)
    # print('encoder_inputs_embedded',sess.run(encoder_inputs_embedded))
    step = 0
    batches = getBatches(trainingSamples, batch_size)
    for nextBatch in tqdm(batches, desc="Training"):

    # while step * batch_size < training_iters:
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        batch_xs, batch_ys=nextBatch.encoder_inputs,nextBatch.decoder_targets

        # print('batch_xs, batch_ys shape',np.array(batch_xs).shape, np.array(batch_ys).shape)
        # print('batch_xs, batch_ys\n',batch_xs,'\n',batch_ys)
        '''
         [[3, 3, 797, 1380, 3, 3, 2146, 3, 19740, 3, 230, 3035, 3, 10, 10, 3, 3, 734, 3, 2674, 111, 3, 308, 3058, 3, 10, 3, 3, 308, 3058, 3, 11, 3, 198, 3, 3, 320, 553, 3, 3, 3, 3, 3, 299, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 3, 433, 3, 12852, 2836, 3, 6096, 3, 8590, 3, 2836, 3, 9745, 580, 3, 3, 4587, 3, 641, 3, 3, 3, 25, 3, 3, 4743, 762, 3, 3, 129, 3, 7166, 7934, 3, 3, 1149, 3, 3, 3, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 266, 32, 3, 3, 2761, 3, 2707, 3, 3, 299, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 3, 433, 3, 198, 3, 3, 3, 4252, 3, 3, 1213, 710, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 308, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 3, 233, 3, 11, 3035, 3, 12852, 2836, 3, 6096, 3, 8590, 3, 2836, 3, 25, 3, 36, 3, 3, 10235, 3, 3, 2761], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3579, 3, 10, 3, 32, 2418, 3, 3, 23360, 3, 1084, 3, 3, 3, 8364, 3, 159, 32, 3, 961, 3, 32, 2418, 3, 291, 9848, 595, 3, 3, 3, 3, 7183, 5217, 3, 42, 3, 37, 3, 3, 3, 36, 3, 1677, 25, 3, 11694, 797, 3, 159, 32, 3, 2563, 1115, 3, 961, 3, 32, 2418, 3, 291, 9848, 3, 595, 3, 641, 3, 3, 32, 2418, 3, 3, 23360, 3, 11, 3035, 3, 641, 1115, 3, 320], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 11981, 3, 3, 2764, 433, 3, 2761, 3, 3, 1655, 2574, 3, 23388, 710, 3, 436, 3, 344, 3, 3, 13, 3, 3, 1711, 3, 215, 3, 10, 3, 159, 32], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 44, 3, 3, 3, 3, 3, 1630, 48, 3, 1084, 550, 3, 35, 3473, 3, 3, 3, 1634, 3, 3056, 3, 3, 141, 3, 4366, 21349, 3, 931, 695, 3, 3, 13886, 3, 3, 8364, 308, 3058, 3, 737, 3, 3, 339, 3, 643, 3693, 3, 1148, 10069, 3082, 3, 419, 3, 168, 3, 38, 3, 4753, 5630, 3, 1148, 10069, 3082, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 419, 3, 3, 38, 1148, 3, 32, 858, 3, 695, 3, 3822]] 
         [[1], [0], [0], [1]]
         
         
         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 10285, 4987, 3, 3, 3, 3, 1616, 3, 3, 10270, 3, 11, 3035, 3, 3, 17273, 3, 17475, 1037, 3, 3, 92, 3, 13811, 6099, 3, 2563, 3, 3, 308, 3058, 3, 3, 2616, 3, 78, 3, 3, 10285, 4987, 3, 838, 3056, 3, 385, 3, 4198, 3, 3, 32, 3, 3, 10285, 4987, 3, 13, 3, 1616, 3, 159, 32, 3, 201, 797, 3, 3579, 1115, 3, 78, 3, 14604, 3, 3, 32, 3, 797, 2146, 3, 3874, 3874, 3, 10285, 4987, 3, 299, 3, 36, 3, 3, 4533, 3, 31, 3, 12033, 12387, 3, 3, 3, 3, 3, 838, 3, 664, 320, 3, 3, 3, 3, 3, 3, 3, 20108, 1859, 3, 37, 3, 3, 38, 31, 3, 3, 4324, 11437, 3, 1213, 2358, 3, 3, 3, 3, 3854, 1213, 3, 308, 3058], [3, 3, 95, 3, 3, 22, 3464, 3, 308, 3058, 3, 76, 3, 3, 797, 4151, 3, 11, 3, 3, 3, 3, 641, 149, 3, 3, 6164, 3, 3, 2486, 3, 579, 3, 3, 3, 3, 3, 3, 373, 3, 3, 3, 3, 111, 3, 3, 424, 3, 3, 233, 3, 24035, 215, 3, 1148, 3291, 3, 35, 3, 159, 32, 3, 797, 32, 3, 419, 3, 3, 3, 32, 711, 3, 23793, 3, 3, 266, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 3, 3, 3, 159, 32, 3, 159, 3, 111, 3, 3, 3, 3, 3, 11, 3, 3, 3, 3, 3, 159, 32, 3, 3, 3, 3, 11439, 3, 3, 2662, 1558, 3, 11439, 3, 3, 2662, 1558, 3, 3, 284, 3, 9745, 436, 3, 1037, 22, 3, 3, 2761, 3, 664, 320, 3, 159, 32, 3, 233, 3, 3, 3, 433, 3, 643, 3693, 3, 308, 3058, 3, 32, 3, 266, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 12852, 2836, 3, 6096, 3, 8590, 3, 2836, 3, 10285, 4987, 3, 44, 3, 3, 23793, 3, 3, 266, 3, 9737, 12852, 961, 3593, 11021, 1773, 3, 8590, 3, 2836, 3, 12852, 2836, 3, 6096, 3, 8590, 3, 2836, 3, 3, 342, 3, 3, 3, 3, 198, 3, 3, 3, 5273, 3, 3, 3, 3, 641, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 11, 3, 641, 149, 3, 1148, 3, 3, 1148, 32, 3, 3, 6398, 3, 3, 1092, 1334, 3, 1018, 547, 3, 99, 3, 1894, 3610, 3, 159, 32, 3, 451, 94, 3, 550, 3, 1655, 2574, 3, 1092, 1334, 3, 3, 3, 3, 3, 7719, 3, 4704, 1845, 3, 37, 3, 11, 3035, 3, 3, 6170, 3, 100, 3, 4479, 3, 3, 37, 3, 1092, 1334, 158, 3, 308, 3058, 3, 134, 3, 3, 1711, 3, 32, 2418, 3, 22182, 5296, 3, 78, 3, 3, 3, 4425, 3, 3, 3, 3, 1237, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 2506, 3, 7293, 10885, 3, 3, 3, 3, 32, 3, 10033, 6132, 3, 10, 3, 44, 1380, 3, 22, 3464, 3, 1054, 3, 3, 3, 424, 3, 22, 3464, 3, 32, 3, 10033, 6132, 3, 22, 3464, 3, 22951, 550, 3, 1436, 1115, 3, 170]] 
          [1, 0, 1, 0]
        '''

        # batch_xs, batch_ys
        # shape(4, 5, 3)(4, 2)


        sess.run([train_op], feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys,keep_prob:0.9})
        if step % 1 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys,keep_prob:0.9})
            print("step" + str(step) + ",Minibatch Loss=" + "{:.4f}".format(loss)
                  + ",Training Accuracy=" + "{:.3f}".format(acc))
        step += 1

        #只有一个batch,运行过就结束
        # break

    print("Optimization Finished!")

    # # calculate accuracy for 128 mnist test image
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_inputs))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,keep_prob:0.9}))


'''
未使用预训练词向量初始化
step0,Minibatch Loss=0.6895,Training Accuracy=0.570
Training:   4%|▍         | 1/25 [02:50<1:08:07, 170.30s/it]step1,Minibatch Loss=0.6928,Training Accuracy=0.520
Training:   8%|▊         | 2/25 [06:09<1:10:43, 184.50s/it]step2,Minibatch Loss=0.6768,Training Accuracy=0.575
Training:  12%|█▏        | 3/25 [11:08<1:21:44, 222.92s/it]step3,Minibatch Loss=0.6304,Training Accuracy=0.642
Training:  20%|██        | 5/25 [20:14<1:20:58, 242.93s/it]step4,Minibatch Loss=0.6500,Training Accuracy=0.572
Training:  24%|██▍       | 6/25 [24:41<1:18:10, 246.89s/it]step5,Minibatch Loss=0.6285,Training Accuracy=0.645
step6,Minibatch Loss=0.6129,Training Accuracy=0.710
Training:  32%|███▏      | 8/25 [32:01<1:08:02, 240.17s/it]step7,Minibatch Loss=0.6168,Training Accuracy=0.663
Training:  36%|███▌      | 9/25 [36:38<1:05:08, 244.31s/it]step8,Minibatch Loss=0.5830,Training Accuracy=0.715
step9,Minibatch Loss=0.5372,Training Accuracy=0.780
Training:  44%|████▍     | 11/25 [44:42<56:53, 243.83s/it]  step10,Minibatch Loss=0.5364,Training Accuracy=0.765
step11,Minibatch Loss=0.5433,Training Accuracy=0.757
Training:  52%|█████▏    | 13/25 [54:35<50:23, 251.97s/it]step12,Minibatch Loss=0.5877,Training Accuracy=0.710
Training:  56%|█████▌    | 14/25 [58:13<45:45, 249.56s/it]step13,Minibatch Loss=0.5186,Training Accuracy=0.793
Training:  60%|██████    | 15/25 [1:02:35<41:43, 250.35s/it]step14,Minibatch Loss=0.4939,Training Accuracy=0.817
step15,Minibatch Loss=0.5040,Training Accuracy=0.800
Training:  64%|██████▍   | 16/25 [1:05:50<37:02, 246.89s/it]step16,Minibatch Loss=0.5161,Training Accuracy=0.795
Training:  72%|███████▏  | 18/25 [1:12:00<28:00, 240.04s/it]step17,Minibatch Loss=0.4871,Training Accuracy=0.817
step18,Minibatch Loss=0.5488,Training Accuracy=0.755
Training:  76%|███████▌  | 19/25 [1:18:49<24:53, 248.90s/it]step19,Minibatch Loss=0.5387,Training Accuracy=0.765
Training:  80%|████████  | 20/25 [1:22:42<20:40, 248.12s/it]step20,Minibatch Loss=0.5344,Training Accuracy=0.755

'''

'''
pre vec
  TqdmSynchronisationWarning)
Training:   4%|▍         | 1/25 [01:35<38:14, 95.61s/it]step0,Minibatch Loss=0.7792,Training Accuracy=0.533
step1,Minibatch Loss=0.7183,Training Accuracy=0.592
Training:   8%|▊         | 2/25 [03:05<35:34, 92.80s/it]step2,Minibatch Loss=0.7037,Training Accuracy=0.608
Training:  16%|█▌        | 4/25 [08:32<44:52, 128.20s/it]step3,Minibatch Loss=0.7316,Training Accuracy=0.575
step4,Minibatch Loss=0.7116,Training Accuracy=0.600
Training:  24%|██▍       | 6/25 [11:50<37:28, 118.35s/it]step5,Minibatch Loss=0.7511,Training Accuracy=0.560
step6,Minibatch Loss=0.7353,Training Accuracy=0.577
Training:  28%|██▊       | 7/25 [13:29<34:42, 115.69s/it]step7,Minibatch Loss=0.7201,Training Accuracy=0.592
Training:  32%|███▏      | 8/25 [15:02<31:58, 112.86s/it]step8,Minibatch Loss=0.7581,Training Accuracy=0.555
Training:  40%|████      | 10/25 [19:12<28:48, 115.23s/it]step9,Minibatch Loss=0.7018,Training Accuracy=0.610
Training:  44%|████▍     | 11/25 [20:20<25:53, 110.98s/it]step10,Minibatch Loss=0.7109,Training Accuracy=0.603
Training:  48%|████▊     | 12/25 [22:07<23:58, 110.65s/it]step11,Minibatch Loss=0.7166,Training Accuracy=0.598
step12,Minibatch Loss=0.7298,Training Accuracy=0.582
Training:  56%|█████▌    | 14/25 [24:07<18:57, 103.37s/it]step13,Minibatch Loss=0.7333,Training Accuracy=0.580
step14,Minibatch Loss=0.7673,Training Accuracy=0.545
Training:  64%|██████▍   | 16/25 [27:53<15:41, 104.57s/it]step15,Minibatch Loss=0.7685,Training Accuracy=0.545
Training:  68%|██████▊   | 17/25 [29:51<14:02, 105.36s/it]step16,Minibatch Loss=0.7046,Training Accuracy=0.608
step17,Minibatch Loss=0.7281,Training Accuracy=0.585
Training:  72%|███████▏  | 18/25 [31:11<12:07, 103.98s/it]step18,Minibatch Loss=0.7687,Training Accuracy=0.545
Training:  76%|███████▌  | 19/25 [31:53<10:04, 100.71s/it]step19,Minibatch Loss=0.7035,Training Accuracy=0.610
Training:  80%|████████  | 20/25 [32:54<08:13, 98.74s/it] step20,Minibatch Loss=0.7758,Training Accuracy=0.538
Training:  84%|████████▍ | 21/25 [34:42<06:36, 99.17s/it]step21,Minibatch Loss=0.7302,Training Accuracy=0.582
Training:  92%|█████████▏| 23/25 [38:03<03:18, 99.26s/it] step22,Minibatch Loss=0.7311,Training Accuracy=0.582
Training:  96%|█████████▌| 24/25 [39:15<01:38, 98.13s/it]step23,Minibatch Loss=0.7059,Training Accuracy=0.608
Training: 100%|██████████| 25/25 [42:22<00:00, 101.69s/it]step24,Minibatch Loss=0.7605,Training Accuracy=0.553

'''