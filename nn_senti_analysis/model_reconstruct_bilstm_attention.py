# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import numpy as np
from nn_senti_analysis.data_helpers import loadDataset, getBatches, sentence2enco
from tqdm import tqdm
import os

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
lr = 0.0001  # learning rate

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
model_fruit_transform_path='model_fruit_transform'
model_hotel_path='model_hotel_reconstruct_bi_attention'
# x y placeholder
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) #(4,5,3)
# y = tf.placeholder(tf.float32, [None, n_classes])  #(4, 2)



encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
# encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

embedding = tf.get_variable('embedding', [28694, embedding_size])##len(word2id)
encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, encoder_inputs)
keep_prob = tf.placeholder(tf.float32)
batch_size=tf.placeholder(tf.int32, [], name='batch_sizee')#300
batch_size_flag=300



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
    
def bi_single_rnn_cell(rnn_size,keep_prob):
        # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
        # 的列表中最终模型会发生错误
        single_cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
        single_cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
        #添加dropout
        cell_drop_fw = tf.contrib.rnn.DropoutWrapper(single_cell_fw, output_keep_prob=keep_prob)
        cell_drop_bw = tf.contrib.rnn.DropoutWrapper(single_cell_bw, output_keep_prob=keep_prob)
        return cell_drop_fw,cell_drop_bw
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
            print(index, '_inputs o', _inputs)
            '''
            0 _inputs o Tensor("encoder/embedding_lookup/Identity:0", shape=(?, ?, 1024), dtype=float32)
            1 _inputs o Tensor("encoder/concat:0", shape=(?, ?, 2048), dtype=float32)
            2 _inputs o Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)
            '''

            # 这个结构每次要重新加载，否则会把之前的参数也保留从而出错
            rnn_cell_fw, rnn_cell_bw = bi_single_rnn_cell(n_hidden_units,keep_prob)

            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs,
                                                              sequence_length=X_length,
                                                              initial_state_fw=initial_state_fw,
                                                              initial_state_bw=initial_state_bw,
                                                              dtype=tf.float32)

            print('index,output', index, output)
            '''
            output输出每次都是1024，不管输入是多少维度的，相当与接了1024的全链接层
            index,output 0 (<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)
            index,output 1 (<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn_1/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)
            index,output 2 (<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn_2/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)

            '''
            # print('''type state[0].c''', type(state[0].c))
            # type state[0].c <class 'tensorflow.python.framework.ops.Tensor'>

            _inputs = tf.concat(output, 2)
            #分类时没有使用state,只使用了outputs
            # encoder_final_state_c = tf.concat(
            #     (state[0].c, state[1].c), 1)
            #
            # encoder_final_state_h = tf.concat(
            #     (state[0].h, state[1].h), 1)
            #
            # encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
            #     c=encoder_final_state_c,
            #     h=encoder_final_state_h
            # )
            # all_layer_final_state.append(encoder_final_state)

            # print('index:{},state:{}'.format(index, state))
            '''
            num_layers=3时
            index:0,state:(
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), 
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))

            index:1,state:(
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), 
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))

            index:2,state:(
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), 
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))


            '''

    encoder_outputs = _inputs
    # encoder_state = tuple(all_layer_final_state)  # state#
    # print('encoder_state', encoder_state)

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
    '''
    encoder_state (
    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/concat_2:0' shape=(?, 2048) dtype=float32>), 
    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_2:0' shape=(?, 2048) dtype=float32>), 
    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/concat_2:0' shape=(?, 2048) dtype=float32>),
    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_3/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_3/concat_2:0' shape=(?, 2048) dtype=float32>))

    '''

    # print('encoder_outputs', encoder_outputs)
    '''
    encoder_outputs Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)

    '''

    # print('state', state)  #
    '''
    state (
     LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>),
     LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))

    '''

    # encoder_state，应该为每一层最后时刻的lstmtuple()组成的tuple
    # print('encoder_state', encoder_state)
    '''
    (
    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/concat_2:0' shape=(?, 2048) dtype=float32>), 
    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_2:0' shape=(?, 2048) dtype=float32>))

    '''
    pass

# def RNN(X):
#     # 计算序列padd过后的序列有效长度
#     X_length = length(X)
#     # MultiRNNCel
#     ####################################################################################
#     # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
#     # lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden_units)  #cell可以选择lstm也可以用gru
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
#
#     # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
#     lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
#
#     # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
#     mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
#
#     # **步骤5：用全零来初始化state
#     init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
#
#     # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
#     # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
#     # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
#     # ** state.shape = [layer_num, 2, batch_size, hidden_size],
#     # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
#     # ** 最后输出维度是 [batch_size, hidden_size]
#     outputs, final_state_ = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, sequence_length=X_length, initial_state=init_state,
#                                               time_major=False)
#     # final_state估计是（layer_num,(cell_state,hidden_state)）
#     print('outputs', outputs)#outputs Tensor("rnn/transpose_1:0", shape=(400, ?, 300), dtype=float32),batch_size*step*dim
#     # 针对有padding时的state
#
#
#     seq_len_num=X_length.get_shape()
#     print('seq_len_num',seq_len_num)
#
#     # last_relevant_state = last_relevant(outputs, X_length)
#
#
#     #inputs=outputs没有对outputs进行掩膜，即取前t个不是0的step,attention_size可简单的设为n_hidden_units
#     attention_cls=attention_self_define(inputs=outputs, attention_size=n_hidden_units, time_major=False, return_alphas=False)
#     last_relevant_state=attention_cls
#
#     print('last_relevant_state ', last_relevant_state)#Tensor("GatherV2:0", shape=(?, 300), dtype=float32)batch_size*dim
#     # with tf.variable_scope("logis",reuse=None):
#
#     logits=tf.layers.dense(inputs=last_relevant_state, units=n_classes, activation=None,
#                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                            bias_initializer = tf.zeros_initializer(),
#
#                            )
#     return logits


print('encoder_inputs_embedded', encoder_inputs_embedded)
logits = RNN(encoder_inputs_embedded)

# 加入last_relavent有问题
pred = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_one_hot))

# Training summary for the current batch_loss
tf.summary.scalar('loss', cost)
summary_op = tf.summary.merge_all()
step = tf.Variable(0, trainable=False)
train_op = tf.train.AdamOptimizer(lr).minimize(cost, global_step=step)

# Evaluate mode
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# start training
with tf.Session() as sess:
    for each in tf.all_variables():
        print('variable name', each.name,each)

    # for n in tf.get_default_graph().as_graph_def().node:
    #     print('node.name', n.name)
    # Run the initializer

    transfer_learning = False
    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(model_hotel_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :#and False:
        print('Reloading model parameters..')
        if transfer_learning:
            restore_vaiables=[each for each in tf.global_variables() if 'dense' not in each.name]
            print('载入瓶颈层参数，开始迁移学习')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(restore_vaiables,
                                   max_to_keep=3)
            saver.restore(sess, ckpt.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(model_fruit_transform_path, graph=sess.graph)
        else:
            print('重载所有变量继续训练模型')
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=2)
            saver.restore(sess, ckpt.model_checkpoint_path)
            if 'hotel' in data_path:
                summary_writer = tf.summary.FileWriter(model_hotel_path, graph=sess.graph)
            else:
                summary_writer = tf.summary.FileWriter(model_fruit_path, graph=sess.graph)




    else:
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=2)
        print('Created new model parameters..')
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        if 'hotel' in data_path:
            summary_writer = tf.summary.FileWriter(model_hotel_path, graph=sess.graph)
        else:
            summary_writer = tf.summary.FileWriter(model_fruit_path, graph=sess.graph)
    # for each in tf.all_variables():
    #     print('each var', each)
    # print('encoder_inputs_embedded',sess.run(encoder_inputs_embedded))

    # summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)
    for e in range(numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, numEpochs))
        batches = getBatches(trainingSamples, batch_size_flag)
        for nextBatch in tqdm(batches, desc="Training"):
        # for i in range(1):
        #     nextBatch=batches[-1]

            batch_xs, batch_ys = nextBatch.encoder_inputs, nextBatch.decoder_targets
            #最后一个batch大小只有100，但是看到的是300所有有问题了；遇到最后一个batch时

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

            _, current_step = sess.run([train_op, step],
                                       feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys, keep_prob: 0.5,batch_size:len(batch_xs)})#
            if current_step % 1 == 0:
                loss, acc, summary = sess.run([cost, accuracy, summary_op],
                                              feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys,
                                                         keep_prob: 0.5,batch_size:len(batch_xs)})#len(batch_xs)


                summary_writer.add_summary(summary, current_step)
                tqdm.write("----- Step %d -- Loss %.5f -- acc %.5f" % (current_step, loss, acc))



                if transfer_learning:
                    checkpoint_path = os.path.join(model_fruit_transform_path, 'senti_analysis.ckpt')
                    saver.save(sess, checkpoint_path, global_step=current_step)
                else:
                    if 'hotel' in data_path:
                        checkpoint_path = os.path.join(model_hotel_path, 'senti_analysis.ckpt')
                        saver.save(sess, checkpoint_path, global_step=current_step)
                    else:
                        checkpoint_path = os.path.join(model_fruit_path, 'senti_analysis.ckpt')
                        saver.save(sess, checkpoint_path, global_step=current_step)



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


在hotel训练一个epoch即25个step之后开始迁移学习，与fruit数据集
----- Step 26 -- Loss 0.69319 -- acc 0.50500
Training:   4%|▍         | 1/25 [00:16<06:29, 16.23s/it]----- Step 27 -- Loss 0.69141 -- acc 0.53000
Training:   8%|▊         | 2/25 [00:26<05:07, 13.35s/it]----- Step 28 -- Loss 0.69259 -- acc 0.50000
Training:  12%|█▏        | 3/25 [00:41<05:05, 13.88s/it]----- Step 29 -- Loss 0.68703 -- acc 0.54500
Training:  16%|█▌        | 4/25 [00:54<04:45, 13.59s/it]----- Step 30 -- Loss 0.69093 -- acc 0.51750
Training:  20%|██        | 5/25 [01:31<06:04, 18.22s/it]----- Step 31 -- Loss 0.68708 -- acc 0.54750
Training:  24%|██▍       | 6/25 [01:31<04:49, 15.26s/it]----- Step 32 -- Loss 0.67911 -- acc 0.60000
Training:  28%|██▊       | 7/25 [01:50<04:44, 15.81s/it]----- Step 33 -- Loss 0.66969 -- acc 0.59000
Training:  32%|███▏      | 8/25 [02:41<05:42, 20.16s/it]----- Step 34 -- Loss 0.65140 -- acc 0.64000
Training:  36%|███▌      | 9/25 [02:41<04:47, 17.97s/it]----- Step 35 -- Loss 0.66303 -- acc 0.60750
Training:  40%|████      | 10/25 [03:03<04:34, 18.31s/it]----- Step 36 -- Loss 0.63000 -- acc 0.68500
Training:  44%|████▍     | 11/25 [04:46<06:04, 26.07s/it]----- Step 37 -- Loss 0.63506 -- acc 0.63500
Training:  48%|████▊     | 12/25 [04:54<05:18, 24.51s/it]----- Step 38 -- Loss 0.60170 -- acc 0.71500
Training:  52%|█████▏    | 13/25 [04:54<04:31, 22.66s/it]----- Step 39 -- Loss 0.59673 -- acc 0.69750
Training:  56%|█████▌    | 14/25 [05:26<04:16, 23.30s/it]


Training:   4%|▍         | 1/25 [00:15<06:17, 15.73s/it]----- Step 212 -- Loss 0.69394 -- acc 0.47250
Training:   8%|▊         | 2/25 [01:08<13:09, 34.33s/it]----- Step 213 -- Loss 0.69213 -- acc 0.51000
Training:  12%|█▏        | 3/25 [01:09<08:27, 23.06s/it]----- Step 214 -- Loss 0.69281 -- acc 0.51750
Training:  16%|█▌        | 4/25 [01:23<07:19, 20.92s/it]----- Step 215 -- Loss 0.69379 -- acc 0.48000
Training:  20%|██        | 5/25 [02:04<08:19, 24.95s/it]----- Step 216 -- Loss 0.69259 -- acc 0.50250
Training:  24%|██▍       | 6/25 [02:05<06:36, 20.86s/it]----- Step 217 -- Loss 0.69363 -- acc 0.49250

不用迁移学习直接训练
Training:   0%|          | 0/25 [00:14<?, ?it/s]----- Step 1 -- Loss 0.69295 -- acc 0.53500
Training:   4%|▍         | 1/25 [00:29<11:37, 29.07s/it]----- Step 2 -- Loss 0.69233 -- acc 0.53750
Training:   8%|▊         | 2/25 [00:59<11:18, 29.52s/it]----- Step 3 -- Loss 0.69269 -- acc 0.49750
Training:  12%|█▏        | 3/25 [01:12<08:53, 24.24s/it]----- Step 4 -- Loss 0.69312 -- acc 0.47750
Training:  16%|█▌        | 4/25 [01:23<07:19, 20.94s/it]----- Step 5 -- Loss 0.69145 -- acc 0.56500
Training:  20%|██        | 5/25 [01:48<07:13, 21.69s/it]----- Step 6 -- Loss 0.69188 -- acc 0.53500
Training:  24%|██▍       | 6/25 [01:48<05:44, 18.14s/it]----- Step 7 -- Loss 0.68690 -- acc 0.59000
Training:  28%|██▊       | 7/25 [02:23<06:08, 20.47s/it]----- Step 8 -- Loss 0.68891 -- acc 0.53000
Training:  32%|███▏      | 8/25 [02:36<05:31, 19.50s/it]----- Step 9 -- Loss 0.67693 -- acc 0.61000
Training:  36%|███▌      | 9/25 [02:47<04:57, 18.60s/it]----- Step 10 -- Loss 0.66162 -- acc 0.64250
Training:  40%|████      | 10/25 [02:47<04:11, 16.78s/it]----- Step 11 -- Loss 0.64099 -- acc 0.64500
Training:  44%|████▍     | 11/25 [04:13<05:23, 23.08s/it]----- Step 12 -- Loss 0.64497 -- acc 0.61500
Training:  48%|████▊     | 12/25 [04:27<04:49, 22.29s/it]----- Step 13 -- Loss 0.70930 -- acc 0.50500
Training:  52%|█████▏    | 13/25 [04:27<04:07, 20.61s/it]----- Step 14 -- Loss 0.61598 -- acc 0.75000


'''
'''
维度不一致造成的，sequence_length=100,但是输入数据是任然是300；
File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1334, in _do_call
    return fn(*args)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1319, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1407, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [Expected shape for Tensor bidirectional-rnn/bidirectional_rnn/fw/fw/sequence_length:0 is ] [100] [ but saw shape: ] [300]
	 [[{{node bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert}} = Assert[T=[DT_STRING, DT_INT32, DT_STRING, DT_INT32], summarize=3, _device="/job:localhost/replica:0/task:0/device:CPU:0"](bidirectional-rnn/bidirectional_rnn/fw/fw/All, bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert/data_0, bidirectional-rnn/bidirectional_rnn/fw/fw/stack, bidirectional-rnn_1/bidirectional_rnn/fw/fw/Assert/Assert/data_2, bidirectional-rnn_1/bidirectional_rnn/fw/fw/Shape)]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/ozintel/Downloads/Tsl_python_progect/local_ml/senti_analysis/nn_senti_analysis/model_reconstruct_bilstm_attention.py", line 462, in <module>
    feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys, keep_prob: 0.5})
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 929, in run
    run_metadata_ptr)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1152, in _run
    feed_dict_tensor, options, run_metadata)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1328, in _do_run
    run_metadata)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1348, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [Expected shape for Tensor bidirectional-rnn/bidirectional_rnn/fw/fw/sequence_length:0 is ] [100] [ but saw shape: ] [300]
	 [[node bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert (defined at /Users/ozintel/Downloads/Tsl_python_progect/local_ml/senti_analysis/nn_senti_analysis/model_reconstruct_bilstm_attention.py:220)  = Assert[T=[DT_STRING, DT_INT32, DT_STRING, DT_INT32], summarize=3, _device="/job:localhost/replica:0/task:0/device:CPU:0"](bidirectional-rnn/bidirectional_rnn/fw/fw/All, bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert/data_0, bidirectional-rnn/bidirectional_rnn/fw/fw/stack, bidirectional-rnn_1/bidirectional_rnn/fw/fw/Assert/Assert/data_2, bidirectional-rnn_1/bidirectional_rnn/fw/fw/Shape)]]

Caused by op 'bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert', defined at:
  File "/Users/ozintel/Downloads/Tsl_python_progect/local_ml/senti_analysis/nn_senti_analysis/model_reconstruct_bilstm_attention.py", line 369, in <module>
    logits = RNN(encoder_inputs_embedded)
  File "/Users/ozintel/Downloads/Tsl_python_progect/local_ml/senti_analysis/nn_senti_analysis/model_reconstruct_bilstm_attention.py", line 220, in RNN
    dtype=tf.float32)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/rnn.py", line 439, in bidirectional_dynamic_rnn
    time_major=time_major, scope=fw_scope)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/rnn.py", line 651, in dynamic_rnn
    [_assert_has_shape(sequence_length, [batch_size])]):
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/rnn.py", line 646, in _assert_has_shape
    packed_shape, " but saw shape: ", x_shape])
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped
    return _add_should_use_warning(fn(*args, **kwargs))
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/control_flow_ops.py", line 159, in Assert
    return gen_logging_ops._assert(condition, data, summarize, name="Assert")
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/gen_logging_ops.py", line 52, in _assert
    name=name)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py", line 488, in new_func
    return func(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 3274, in create_op
    op_def=op_def)
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1770, in __init__
    self._traceback = tf_stack.extract_stack()

InvalidArgumentError (see above for traceback): assertion failed: [Expected shape for Tensor bidirectional-rnn/bidirectional_rnn/fw/fw/sequence_length:0 is ] [100] [ but saw shape: ] [300]
	 [[node bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert (defined at /Users/ozintel/Downloads/Tsl_python_progect/local_ml/senti_analysis/nn_senti_analysis/model_reconstruct_bilstm_attention.py:220)  = Assert[T=[DT_STRING, DT_INT32, DT_STRING, DT_INT32], summarize=3, _device="/job:localhost/replica:0/task:0/device:CPU:0"](bidirectional-rnn/bidirectional_rnn/fw/fw/All, bidirectional-rnn/bidirectional_rnn/fw/fw/Assert/Assert/data_0, bidirectional-rnn/bidirectional_rnn/fw/fw/stack, bidirectional-rnn_1/bidirectional_rnn/fw/fw/Assert/Assert/data_2, bidirectional-rnn_1/bidirectional_rnn/fw/fw/Shape)]]


'''