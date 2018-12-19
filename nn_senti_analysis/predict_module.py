# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import numpy as np
from nn_senti_analysis.data_helpers import loadDataset, getBatches, sentence2enco
from tqdm import tqdm
import os
import sys

'''
（1）padding:dynamic处理变长序列时，取最大长度序列，不足的序列补0；
 (2) mask:设置sequence_length，这样输出时补零的time_step部分输出也为0
（3）此时计算cost不是取最后一个time_step的hidden_state而是取最后一个不为零的：cost这里其实是不用mask的，因为label和prdiciton都是0
（4）预测的时候，即训练结束后，放入softmax分类时，不是取最后一个time_step的hidden_state而是取最后一个不为零的
'''

# ssl._create_default_https_context = ssl._create_unverified_context
#
# tf.set_random_seed(1)  # set random seed

data_path = '../data/data_cleaned/hotel-vocabSize50000.pkl'
# data_path='../data/data_cleaned/fruit-vocabSize50000.pkl'#迁移学习时，词汇个数不一样维度就不一样

word2id, id2word, trainingSamples = loadDataset(data_path)
print('trainingSamples',trainingSamples)

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
lr = 0.00005  # learning rate

training_iters = 100000  # train step 上限

# n_inputs = 3  # MNIST data input(img shape:28*28)
# n_steps = 5  # time steps

n_hidden_units = 300  # neurons in hidden layer
n_classes = 2  # MNIST classes(0-9 digits)
# LSTM layer 的层数
layer_num = 2
embedding_size = 300  # n_hidden_units与embedding_size的关系;两者大小相等。
numEpochs = 10
model_fruit_path = 'model_fruit'
model_fruit_transform_path = 'model_fruit_transform'
model_hotel_path = 'model_hotel_reconstruct_bi_attention'
# x y placeholder
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) #(4,5,3)
# y = tf.placeholder(tf.float32, [None, n_classes])  #(4, 2)



encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
# encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

embedding = tf.get_variable('embedding', [28694, embedding_size])  ##len(word2id)
encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, encoder_inputs)
keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32, [], name='batch_sizee')  # 300
batch_size_flag = 300

decoder_targets = tf.placeholder(tf.int32, [None, ], name='decoder_targets')
print('decoder_targets 0', decoder_targets)

label_one_hot = tf.one_hot(decoder_targets, 2)
print('decoder_targets 1', decoder_targets)


# 计算序列中padding后，返回原始序列长度
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


# 计算预测值时会用到，
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


# 用于分类时是不是用mask结果都是一样的
def cost_self_define(target, prediction):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(length, tf.float32)
    return tf.reduce_mean(cross_entropy)


def attention_self_define(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def bi_single_rnn_cell(rnn_size, keep_prob):
    # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
    # 的列表中最终模型会发生错误
    single_cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
    single_cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
    # 添加dropout
    cell_drop_fw = tf.contrib.rnn.DropoutWrapper(single_cell_fw, output_keep_prob=keep_prob)
    cell_drop_bw = tf.contrib.rnn.DropoutWrapper(single_cell_bw, output_keep_prob=keep_prob)
    return cell_drop_fw, cell_drop_bw


def RNN(_inputs):
    X_length = length(_inputs)
    print('_inputs', _inputs)  # Tensor("encoder/embedding_lookup/Identity:0", shape=(?, ?, 1024), dtype=float32)
    ###
    # 5-50／10000 L2；


    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")
    all_layer_final_state = []
    for index, _ in enumerate(range(layer_num)):
        # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
        # 恶心的很.如果不在这加的话,会报错的.
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            # print(index, '_inputs o', _inputs)
            # 这个结构每次要重新加载，否则会把之前的参数也保留从而出错
            rnn_cell_fw, rnn_cell_bw = bi_single_rnn_cell(n_hidden_units, keep_prob)

            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs,
                                                              sequence_length=X_length,
                                                              initial_state_fw=initial_state_fw,
                                                              initial_state_bw=initial_state_bw,
                                                              dtype=tf.float32)

            # print('index,output', index, output)



            _inputs = tf.concat(output, 2)


    encoder_outputs = _inputs

    attention_cls = attention_self_define(inputs=encoder_outputs, attention_size=n_hidden_units, time_major=False,
                                          return_alphas=False)
    last_relevant_state = attention_cls

    print('last_relevant_state ',
          last_relevant_state)  # Tensor("GatherV2:0", shape=(?, 300), dtype=float32)batch_size*dim
    # with tf.variable_scope("logis",reuse=None):

    logits = tf.layers.dense(inputs=last_relevant_state, units=n_classes, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.zeros_initializer(),

                             )
    return logits




print('encoder_inputs_embedded', encoder_inputs_embedded)
logits = RNN(encoder_inputs_embedded)

# 加入last_relavent有问题
pred = tf.nn.softmax(logits)



# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_one_hot))


# Evaluate mode
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# start training
with tf.Session() as sess:
    for each in tf.all_variables():
        print('variable name', each.name, each)

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(model_hotel_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):  # and False:
        print('Reloading model parameters..')
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('model no exists')

    sys.stdout.write("sentence:> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()


    # sys.stdout.write("label:> ")
    # sys.stdout.flush()
    # label=sys.stdin.readline()

    while sentence:
        # label=1#neg
        # label=0#pos
        batch = sentence2enco(sentence, word2id)
        print('batch', batch)
        prediction = sess.run( pred,
                               feed_dict={encoder_inputs: batch.encoder_inputs, decoder_targets: batch.decoder_targets,
                                keep_prob: 1, batch_size: len(batch.encoder_inputs)})

        print('prediction',prediction)
        print("sentence > ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()


        # sys.stdout.write("label:> ")
        # sys.stdout.flush()
        # label = sys.stdin.readline()






