# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import numpy as np
from nn_senti_analysis.data_helpers import loadDataset, getBatches, sentence2enco
from tqdm import tqdm
import os
import pandas as pd

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



text_split_path='../data/data_cleaned/hotel_split.parquet.gzip'
df_text=pd.read_parquet(text_split_path)

word2id, id2word, trainingSamples = loadDataset(data_path)

df_label=df_text['label'].values
print('df_label',df_label)


#评估时使用当前的word2id，trainingSamples使用要评估的语料生成
print('trainingSamples label;shape',np.array(trainingSamples)[:,1],np.array(trainingSamples).shape)
'''
df_label [1 1 1 ... 0 0 0]
trainingSamples label [1 1 1 ... 0 0 0]
'''

#检验两者label是否一致
C=np.array(df_label)==np.array(trainingSamples)[:,1]
if False in C:
    print('存在False')
else:
    print('两者一样')



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
# pred = tf.nn.softmax(logits)
#将losits转为概率，不输出概率时可以不用



# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))


# Evaluate mode
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# start training
with tf.Session() as sess:
    for each in tf.all_variables():
        print('variable name', each.name, each)

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(model_hotel_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :#and False:
        print('Reloading model parameters..')
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('model no exists')



    batches = getBatches(trainingSamples, batch_size_flag,training_flag=False)
    current_step=0
    all_pred_label=[]
    sum_acc=0.0
    for nextBatch in tqdm(batches, desc="Eval"):
        batch_xs, batch_ys = nextBatch.encoder_inputs, nextBatch.decoder_targets
        # 最后一个batch大小只有100，但是看到的是300所有有问题了；遇到最后一个batch时
        # print('batch_xs',np.array(batch_xs).shape)
        # print('batch_xs 0', batch_xs[0])
        # print('batch_xs 1', batch_xs[1])
        print('go 0',[id2word[each] for each in batch_xs[0]])
        print('go 1',[id2word[each] for each in batch_xs[1]])

        if current_step % 1 == 0:
            loss, acc, logits_value = sess.run([cost, accuracy,logits],
                                          feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys,
                                                     keep_prob: 1, batch_size: len(batch_xs)})  # len(batch_xs)  #预测时要关闭dropout
            tqdm.write("----- Step %d -- Loss %.5f -- acc %.5f" % (current_step, loss, acc))
            # print('pred',prediction.shape,prediction)#pred (300, 2)
            batch_pred_label=np.argmax(logits_value, 1)
            # print('predict label',batch_pred_label)
            all_pred_label.extend(batch_pred_label)
            sum_acc=sum_acc+len(batch_xs)*acc


        current_step =current_step+1
        # break
    # df_text=df_text[0:300]
    print('avg acc',sum_acc/len(trainingSamples))#avg acc 0.9145000022649765

    df_text['pred_label']=all_pred_label
    print('df_text',df_text.head(),'\n',df_text.tail())
    df_text=df_text[df_text['label'] != df_text['pred_label']]
    df_text.to_csv('chinese_hotel_eval.csv', index=False)

    print(" Eval Finished!")

'''
数据原来顺序预测
Eval:   0%|          | 0/34 [00:00<?, ?it/s]/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tqdm/_monitor.py:89: TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481)
  TqdmSynchronisationWarning)
Eval:   3%|▎         | 1/34 [00:31<17:09, 31.19s/it]----- Step 0 -- Loss 0.50324 -- acc 0.80667
Eval:   6%|▌         | 2/34 [01:04<17:07, 32.12s/it]----- Step 1 -- Loss 0.53603 -- acc 0.76000
----- Step 2 -- Loss 0.52437 -- acc 0.78000
Eval:  12%|█▏        | 4/34 [01:56<14:36, 29.21s/it]----- Step 3 -- Loss 0.57079 -- acc 0.72333
----- Step 4 -- Loss 0.50944 -- acc 0.79667
Eval:  15%|█▍        | 5/34 [02:15<13:03, 27.02s/it]----- Step 5 -- Loss 0.50712 -- acc 0.79000
Eval:  21%|██        | 7/34 [03:17<12:43, 28.26s/it]----- Step 6 -- Loss 0.53870 -- acc 0.77000
Eval:  24%|██▎       | 8/34 [03:36<11:42, 27.02s/it]----- Step 7 -- Loss 0.55630 -- acc 0.73333
Eval:  26%|██▋       | 9/34 [04:09<11:32, 27.69s/it]----- Step 8 -- Loss 0.52017 -- acc 0.78333
Eval:  29%|██▉       | 10/34 [04:33<10:56, 27.37s/it]----- Step 9 -- Loss 0.52069 -- acc 0.78000
Eval:  32%|███▏      | 11/34 [04:53<10:14, 26.71s/it]----- Step 10 -- Loss 0.50816 -- acc 0.78667
Eval:  35%|███▌      | 12/34 [05:31<10:06, 27.59s/it]----- Step 11 -- Loss 0.51251 -- acc 0.78667
----- Step 12 -- Loss 0.52523 -- acc 0.78667
Eval:  41%|████      | 14/34 [07:02<10:03, 30.18s/it]----- Step 13 -- Loss 0.53613 -- acc 0.78333
Eval:  44%|████▍     | 15/34 [07:52<09:58, 31.52s/it]----- Step 14 -- Loss 0.52243 -- acc 0.77667
----- Step 15 -- Loss 0.55253 -- acc 0.73333
Eval:  50%|█████     | 17/34 [08:35<08:35, 30.32s/it]----- Step 16 -- Loss 0.49490 -- acc 0.81667
Eval:  53%|█████▎    | 18/34 [09:10<08:08, 30.56s/it]----- Step 17 -- Loss 0.41798 -- acc 0.90333
Eval:  56%|█████▌    | 19/34 [09:30<07:30, 30.01s/it]----- Step 18 -- Loss 0.45216 -- acc 0.86667
Eval:  59%|█████▉    | 20/34 [10:05<07:03, 30.28s/it]----- Step 19 -- Loss 0.42413 -- acc 0.89667
Eval:  62%|██████▏   | 21/34 [10:32<06:31, 30.10s/it]----- Step 20 -- Loss 0.40744 -- acc 0.92333
----- Step 21 -- Loss 0.43490 -- acc 0.88333
Eval:  68%|██████▊   | 23/34 [11:28<05:29, 29.92s/it]----- Step 22 -- Loss 0.40609 -- acc 0.91333
Eval:  71%|███████   | 24/34 [11:45<04:54, 29.41s/it]----- Step 23 -- Loss 0.39829 -- acc 0.92667
----- Step 24 -- Loss 0.42124 -- acc 0.90333
Eval:  74%|███████▎  | 25/34 [12:24<04:27, 29.77s/it]----- Step 25 -- Loss 0.43479 -- acc 0.88333
Eval:  79%|███████▉  | 27/34 [13:17<03:26, 29.55s/it]----- Step 26 -- Loss 0.41428 -- acc 0.91333
Eval:  82%|████████▏ | 28/34 [13:40<02:55, 29.29s/it]----- Step 27 -- Loss 0.40758 -- acc 0.92000
----- Step 28 -- Loss 0.41563 -- acc 0.91000
Eval:  88%|████████▊ | 30/34 [14:41<01:57, 29.38s/it]----- Step 29 -- Loss 0.41844 -- acc 0.90000
----- Step 30 -- Loss 0.41962 -- acc 0.90333
Eval:  91%|█████████ | 31/34 [14:57<01:26, 28.95s/it]----- Step 31 -- Loss 0.41421 -- acc 0.90333
Eval:  97%|█████████▋| 33/34 [15:54<00:28, 28.93s/it]----- Step 32 -- Loss 0.41996 -- acc 0.90667
Eval: 100%|██████████| 34/34 [16:01<00:00, 28.27s/it]
----- Step 33 -- Loss 0.43187 -- acc 0.90000
avg acc 0.8385000026226044
 Eval Finished!

'''
