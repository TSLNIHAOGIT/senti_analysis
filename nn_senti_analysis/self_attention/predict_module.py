# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import numpy as np
from nn_senti_analysis.data_helpers import loadDataset, getBatches, sentence2enco
from tqdm import tqdm
import os
import sys
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
pred = tf.nn.softmax(logits)


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
    ckpt = tf.train.get_checkpoint_state(model_fruit_path)
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






