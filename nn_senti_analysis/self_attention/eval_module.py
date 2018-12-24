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




# tf.set_random_seed(1)  # set random seed

# data_path = '../../data/data_cleaned/hotel-vocabSize50000.pkl'
data_path = '../../data/data_cleaned/fruit-vocabSize50000.pkl'  # 迁移学习时，词汇个数不一样维度就不一样


text_split_path='../../data/data_cleaned/hotel_split.parquet.gzip'
df_text=pd.read_parquet(text_split_path)

word2id, id2word, trainingSamples = loadDataset(data_path)

print('all corpus length max', np.max([len(sample[0]) for sample in trainingSamples]))

# hyperparameters
lr = 0.00005  # learning rate

training_iters = 100000  # train step 上限

# n_inputs = 3  # MNIST data input(img shape:28*28)
# n_steps = 5  # time steps

n_hidden_units = 512  # neurons in hidden layer
n_classes = 2  # MNIST classes(0-9 digits)
# LSTM layer 的层数
layer_num = 2
embedding_size = 100  #
numEpochs = 10
model_fruit_path = '../model_fruit_self_attention'
model_fruit_transform_path = '../model_fruit_transform_self_attention'
model_hotel_path = '../model_hotel_self_attention'


# 使用抛弃rnn结构，使用self-attention结构捕捉时序关系，encoder-decoder attention 用于分类（解码）

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def multihead_attention(emb,
                        queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,  # mask使用，在decoder端才会用到
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def attention_label_define(inputs, attention_size, time_major=False, return_alphas=False):
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


# x y placeholder
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) #(4,5,3)
# y = tf.placeholder(tf.float32, [None, n_classes])  #(4, 2)

is_training = True
num_heads = 8
num_blocks = 6

keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32, [], name='batch_sizee')  # 200
batch_size_flag = 400

decoder_targets = tf.placeholder(tf.int32, [None, ], name='decoder_targets')
print('decoder_targets 0', decoder_targets)

label_one_hot = tf.one_hot(decoder_targets, n_classes)
print('decoder_targets 1', decoder_targets)

all_sentences_max_length = np.max([len(sample[0]) for sample in trainingSamples])  # 613#300
# encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')


encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
# encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
# encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, encoder_inputs)

# embedding = tf.get_variable('embedding', [len(word2id), embedding_size])  ##len(word2id),28694
## embedding





# emb = embedding(encoder_inputs, vocab_size=len(word2id), num_units=n_hidden_units, scale=True, scope="enc_embed")
# enc = emb + embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(encoder_inputs)[1]), 0), [tf.shape(encoder_inputs)[0], 1]),
#                                       vocab_size=embedding_size,num_units=n_hidden_units, zero_pad=False, scale=False,scope="enc_pe")

emb = embedding(encoder_inputs, vocab_size=len(word2id), num_units=n_hidden_units, scale=True, scope="enc_embed")
enc = emb + embedding(
    tf.tile(tf.expand_dims(tf.range(tf.shape(encoder_inputs)[1]), 0), [tf.shape(encoder_inputs)[0], 1]),
    vocab_size=all_sentences_max_length, num_units=n_hidden_units, zero_pad=False, scale=False, scope="enc_pe")
# positiencode:vocab_size是一句话中单词的个数最大值


## Dropout
enc = tf.layers.dropout(enc, rate=1 - keep_prob,
                        training=tf.convert_to_tensor(is_training))

# self-attention用于获取时序信息
## Blocks
for i in range(num_blocks):
    with tf.variable_scope("num_blocks_{}".format(i)):
        ### Multihead Attention
        enc = multihead_attention(emb=emb,
                                  queries=enc,
                                  keys=enc,
                                  num_units=n_hidden_units,
                                  num_heads=num_heads,
                                  dropout_rate=1 - keep_prob,
                                  is_training=is_training,
                                  causality=False)

### Feed Forward
outputs = feedforward(enc, num_units=[4 * n_hidden_units, n_hidden_units])
# outs.shape (400, 50, 512);outs.shape (400, 57, 512)

print('outputs', outputs)

# encoder和label之间的attention
attention_cls = attention_label_define(inputs=outputs, attention_size=n_hidden_units, time_major=False,
                                       return_alphas=False)

# Final linear projection
logits = tf.layers.dense(inputs=attention_cls, units=n_classes, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         bias_initializer=tf.zeros_initializer(),

                         )

print('logits', logits)

'''
outputs Tensor("multihead_attention/ln/add_1:0", shape=(?, ?, 512), dtype=float32)
logits Tensor("dense/BiasAdd:0", shape=(?, ?, 2), dtype=float32)
'''

# 加入last_relavent有问题
# pred = tf.nn.softmax(logits)

y_smoothed = label_smoothing(label_one_hot)
# y_smoothed=label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
print('y_smoothed', y_smoothed)  # y_smoothed Tensor("add_1:0", shape=(?, 2), dtype=float32)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed))

# step = tf.Variable(0, trainable=False)
# train_op = tf.train.AdamOptimizer(lr).minimize(cost, global_step=step)

# Evaluate mode
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training summary for the current batch_loss

# log_var = tf.Variable(0.0)
# tf.summary.scalar('loss', log_var)

# tf.summary.scalar('loss', cost)
# summary_op = tf.summary.merge_all()


# start training
with tf.Session() as sess:
    for each in tf.all_variables():
        print('variable name', each.name, each)

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(model_fruit_path)#model_fruit_self_attention
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
