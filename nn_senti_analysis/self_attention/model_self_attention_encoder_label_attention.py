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

#
# tf.set_random_seed(1)  # set random seed

# data_path = '../data/data_cleaned/hotel-vocabSize50000.pkl'
data_path = '../data/data_cleaned/fruit-vocabSize50000.pkl'  # 迁移学习时，词汇个数不一样维度就不一样

word2id, id2word, trainingSamples = loadDataset(data_path)

print('all corpus length max',np.max([len(sample[0]) for sample in trainingSamples]))



# hyperparameters
lr = 0.00005 # learning rate

training_iters = 100000  # train step 上限

# n_inputs = 3  # MNIST data input(img shape:28*28)
# n_steps = 5  # time steps

n_hidden_units = 512  # neurons in hidden layer
n_classes = 2  # MNIST classes(0-9 digits)
# LSTM layer 的层数
layer_num = 2
embedding_size = 100  #
numEpochs = 10
model_fruit_path = 'model_fruit_self_attention'
model_fruit_transform_path = 'model_fruit_transform_self_attention'
model_hotel_path = 'model_hotel_self_attention'




#使用抛弃rnn结构，使用self-attention结构捕捉时序关系，encoder-decoder attention 用于分类（解码）

def normalize(inputs,
              epsilon = 1e-8,
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
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
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

is_training=True
num_heads = 8
num_blocks = 6

keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32, [], name='batch_sizee')  # 200
batch_size_flag = 400

decoder_targets = tf.placeholder(tf.int32, [None, ], name='decoder_targets')
print('decoder_targets 0', decoder_targets)

label_one_hot = tf.one_hot(decoder_targets, n_classes)
print('decoder_targets 1', decoder_targets)

all_sentences_max_length=np.max([len(sample[0]) for sample in trainingSamples])#613#300
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
enc = emb + embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(encoder_inputs)[1]), 0), [tf.shape(encoder_inputs)[0], 1]),
                                      vocab_size=all_sentences_max_length,num_units=n_hidden_units, zero_pad=False, scale=False,scope="enc_pe")
#positiencode:vocab_size是一句话中单词的个数最大值


## Dropout
enc = tf.layers.dropout(enc, rate=1-keep_prob, 
                                    training=tf.convert_to_tensor(is_training))

#self-attention用于获取时序信息
## Blocks
for i in range(num_blocks):
    with tf.variable_scope("num_blocks_{}".format(i)):
        ### Multihead Attention
        enc = multihead_attention(emb = emb,
                                       queries=enc, 
                                        keys=enc, 
                                        num_units=n_hidden_units, 
                                        num_heads=num_heads, 
                                        dropout_rate=1-keep_prob,
                                        is_training=is_training,
                                        causality=False)



### Feed Forward
outputs = feedforward(enc, num_units=[4*n_hidden_units, n_hidden_units])
#outs.shape (400, 50, 512);outs.shape (400, 57, 512)

print('outputs',outputs)



#encoder和label之间的attention
attention_cls = attention_label_define(inputs=outputs, attention_size=n_hidden_units, time_major=False,
                                          return_alphas=False)




# Final linear projection
logits = tf.layers.dense(inputs=attention_cls, units=n_classes, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         bias_initializer=tf.zeros_initializer(),

                         )




print('logits',logits)

'''
outputs Tensor("multihead_attention/ln/add_1:0", shape=(?, ?, 512), dtype=float32)
logits Tensor("dense/BiasAdd:0", shape=(?, ?, 2), dtype=float32)
'''

# 加入last_relavent有问题
# pred = tf.nn.softmax(logits)

y_smoothed = label_smoothing(label_one_hot)
# y_smoothed=label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
print('y_smoothed',y_smoothed)#y_smoothed Tensor("add_1:0", shape=(?, 2), dtype=float32)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed))

step = tf.Variable(0, trainable=False)
train_op = tf.train.AdamOptimizer(lr).minimize(cost, global_step=step)

# Evaluate mode
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training summary for the current batch_loss

# log_var = tf.Variable(0.0)
# tf.summary.scalar('loss', log_var)

tf.summary.scalar('loss', cost)
summary_op = tf.summary.merge_all()

# start training
with tf.Session() as sess:
    for each in tf.all_variables():
        print('variable name', each.name, each)

    # for n in tf.get_default_graph().as_graph_def().node:
    #     print('node.name', n.name)
    # Run the initializer

    transfer_learning = False
    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(model_hotel_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :#and False:
        print('Reloading model parameters..')
        if transfer_learning:
            restore_vaiables = [each for each in tf.global_variables() if 'dense' not in each.name]
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
                # summary_writer = tf.summary.FileWriter(os.path.join(model_hotel_path,'plot_loss'), graph=sess.graph)
                # summary_writer2 = tf.summary.FileWriter(os.path.join(model_hotel_path, 'plot_accuracy'), graph=sess.graph)
            else:
                summary_writer = tf.summary.FileWriter(model_fruit_path, graph=sess.graph)




    else:
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=2)
        print('Created new model parameters..')
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        if 'hotel' in data_path:
            # summary_writer = tf.summary.FileWriter(os.path.join(model_hotel_path, 'plot_loss'), graph=sess.graph)
            # summary_writer2 = tf.summary.FileWriter(os.path.join(model_hotel_path, 'plot_accuracy'), graph=sess.graph)
            summary_writer = tf.summary.FileWriter(model_hotel_path, graph=sess.graph)



        else:
            summary_writer = tf.summary.FileWriter(model_fruit_path, graph=sess.graph)
    # for each in tf.all_variables():
    #     print('each var', each)
    # print('encoder_inputs_embedded',sess.run(encoder_inputs_embedded))

    # summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)
    for e in range(numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, numEpochs))
        batches = getBatches(trainingSamples, batch_size_flag,training_flag=True)



        for nextBatch in tqdm(batches, desc="Training"):
            # for i in range(1):
            #     nextBatch=batches[-1]

            batch_xs, batch_ys,batch_inputs_length = nextBatch.encoder_inputs, nextBatch.decoder_targets,nextBatch.encoder_inputs_length
            # 最后一个batch大小只有100，但是看到的是300所有有问题了；遇到最后一个batch时
            print('nextBatchs.encoder_inputs_length',nextBatch.encoder_inputs_length)
            print([len(each) for each in batch_xs])
            '''
            
            
            '''




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


            feed_dict = {encoder_inputs: batch_xs, decoder_targets: batch_ys, keep_prob: 0.5,
                         batch_size: len(batch_xs)}#encoder_inputs_length:len(batch_xs[0])每条数据padding之后长度都相同，这里取第一条的长度

            outs,logits_,y_smot=sess.run([outputs,logits,y_smoothed],
                                       feed_dict=feed_dict)
            print('outs.shape',outs.shape)#outs.shape (400, 50, 512);outs.shape (400, 57, 512)
            print('logits_ shape',logits_.shape)#logits_ shape (400, 50, 2)
            print('y_smot shape',y_smot.shape)#y_smot shape (400, 2)


            _, current_step = sess.run([train_op, step],
                                       feed_dict=feed_dict)  #
            if current_step % 1 == 0:
                loss, acc, summary = sess.run([cost, accuracy, summary_op],
                                              feed_dict=feed_dict)  # len(batch_xs)

                summary_writer.add_summary(summary, current_step)
                # summary_writer.flush()
                #
                #
                # summary2=



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
'''

----- Epoch 1/10 -----
Training:   0%|          | 0/25 [00:00<?, ?it/s]nextBatchs.encoder_inputs_length [17, 15, 15, 10, 17, 7, 4, 6, 12, 18, 11, 15, 20, 29, 14, 5, 14, 22, 25, 6, 12, 12, 7, 12, 38, 17, 6, 34, 21, 9, 14, 13, 7, 12, 10, 8, 11, 7, 11, 11, 6, 15, 15, 12, 14, 5, 21, 10, 14, 9, 21, 13, 11, 7, 9, 24, 12, 20, 9, 10, 10, 19, 15, 5, 11, 38, 28, 13, 22, 16, 11, 10, 8, 4, 11, 8, 12, 5, 8, 17, 8, 15, 8, 10, 7, 6, 10, 15, 11, 17, 27, 8, 8, 16, 20, 10, 18, 6, 13, 25, 14, 10, 11, 8, 9, 22, 5, 5, 12, 22, 4, 16, 26, 17, 9, 7, 26, 6, 11, 34, 8, 31, 8, 24, 26, 10, 7, 8, 10, 5, 8, 12, 10, 12, 9, 6, 13, 19, 11, 22, 18, 6, 6, 9, 9, 30, 21, 8, 4, 43, 5, 22, 4, 16, 27, 42, 10, 43, 18, 8, 14, 4, 7, 9, 10, 6, 28, 8, 7, 8, 19, 12, 7, 16, 11, 5, 5, 29, 7, 26, 19, 7, 20, 45, 4, 15, 10, 7, 10, 17, 5, 3, 14, 10, 11, 7, 6, 9, 11, 13, 20, 27, 7, 5, 12, 30, 12, 20, 7, 7, 12, 9, 5, 10, 7, 12, 7, 18, 7, 8, 6, 13, 10, 14, 8, 26, 5, 11, 25, 7, 20, 10, 5, 18, 6, 12, 11, 7, 9, 10, 16, 44, 16, 28, 16, 19, 14, 7, 24, 46, 11, 8, 28, 12, 7, 5, 25, 12, 7, 6, 26, 10, 5, 7, 14, 13, 36, 14, 6, 18, 10, 9, 11, 5, 9, 10, 18, 7, 16, 14, 13, 42, 7, 10, 5, 29, 7, 10, 10, 24, 44, 6, 14, 9, 6, 17, 19, 7, 12, 15, 11, 12, 10, 21, 5, 4, 11, 13, 9, 4, 11, 9, 7, 14, 4, 13, 35, 6, 7, 19, 18, 5, 5, 12, 9, 9, 8, 14, 5, 30, 5, 24, 11, 11, 7, 16, 5, 27, 27, 5, 7, 20, 8, 29, 7, 8, 20, 16, 7, 10, 8, 8, 22, 7, 7, 50, 15, 28, 7, 12, 6, 6, 13, 6, 13, 8, 7, 19, 13, 7, 7, 25, 18, 8, 9, 9, 14, 14, 9, 7, 10, 12, 8, 11, 18, 11, 9, 8, 9, 14, 13, 11, 38, 3, 15, 13, 24, 16, 46, 17]
[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
outs.shape (400, 50, 512)
logits_ shape (400, 2)
y_smot shape (400, 2)
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tqdm/_monitor.py:89: TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481)
  TqdmSynchronisationWarning)
Training:   0%|          | 0/25 [00:22<?, ?it/s]----- Step 1 -- Loss 0.62650 -- acc 0.94250
Training:   4%|▍         | 1/25 [00:23<09:12, 23.04s/it]nextBatchs.encoder_inputs_length [59, 5, 21, 12, 16, 8, 20, 14, 11, 8, 9, 13, 7, 14, 16, 9, 6, 44, 16, 10, 3, 16, 12, 8, 11, 10, 7, 7, 10, 12, 10, 4, 14, 7, 13, 5, 13, 8, 14, 14, 10, 10, 7, 7, 12, 9, 7, 18, 16, 8, 15, 6, 10, 7, 17, 21, 13, 6, 21, 13, 7, 25, 19, 5, 31, 6, 7, 5, 5, 7, 9, 10, 10, 31, 7, 8, 7, 46, 21, 7, 9, 5, 9, 28, 22, 8, 5, 9, 13, 5, 10, 10, 11, 7, 5, 20, 20, 17, 7, 6, 14, 4, 41, 6, 8, 13, 62, 10, 7, 9, 11, 12, 16, 24, 18, 7, 4, 9, 6, 127, 9, 9, 10, 12, 29, 7, 9, 10, 3, 6, 8, 12, 17, 26, 40, 14, 12, 8, 25, 26, 21, 29, 13, 13, 15, 8, 66, 12, 7, 50, 11, 5, 18, 14, 15, 7, 5, 18, 14, 6, 12, 5, 8, 8, 8, 9, 7, 18, 20, 7, 15, 11, 14, 13, 8, 15, 21, 9, 33, 9, 15, 16, 13, 11, 11, 10, 10, 25, 10, 11, 21, 5, 12, 13, 7, 10, 9, 5, 20, 9, 11, 9, 29, 14, 7, 5, 9, 20, 37, 20, 8, 134, 6, 64, 7, 14, 8, 6, 6, 10, 11, 24, 18, 9, 9, 7, 25, 11, 19, 10, 6, 45, 9, 13, 9, 14, 11, 14, 29, 10, 10, 11, 8, 30, 9, 7, 5, 10, 7, 11, 14, 8, 11, 10, 5, 8, 6, 26, 10, 24, 13, 36, 15, 9, 17, 12, 7, 8, 6, 5, 25, 19, 7, 8, 5, 9, 9, 11, 7, 9, 7, 9, 7, 14, 25, 12, 14, 10, 14, 15, 8, 7, 5, 20, 9, 42, 14, 9, 7, 8, 9, 7, 7, 17, 14, 37, 4, 12, 12, 7, 8, 7, 16, 4, 10, 9, 15, 10, 5, 7, 6, 14, 9, 10, 10, 11, 9, 25, 35, 8, 6, 7, 29, 10, 30, 10, 16, 16, 7, 23, 8, 21, 15, 11, 8, 7, 13, 9, 8, 13, 16, 11, 11, 7, 13, 5, 5, 7, 8, 24, 28, 15, 6, 12, 4, 7, 9, 20, 11, 6, 6, 19, 7, 17, 11, 8, 15, 14, 18, 12, 9, 20, 7, 30, 26, 6, 22, 13, 21, 83, 9, 10, 4, 22, 31, 7, 11, 10, 8, 7]
[134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134]
outs.shape (400, 134, 512)
logits_ shape (400, 2)
y_smot shape (400, 2)
Training:   4%|▍         | 1/25 [01:46<42:39, 106.66s/it]----- Step 2 -- Loss 0.57370 -- acc 1.00000
Training:   8%|▊         | 2/25 [01:47<20:39, 53.89s/it] nextBatchs.encoder_inputs_length [7, 7, 15, 12, 11, 48, 29, 17, 8, 9, 11, 7, 12, 10, 26, 5, 17, 9, 18, 7, 9, 23, 8, 11, 7, 4, 14, 5, 12, 9, 35, 8, 35, 10, 4, 12, 12, 6, 6, 8, 6, 11, 10, 7, 7, 9, 4, 13, 17, 10, 19, 5, 8, 10, 8, 11, 4, 11, 12, 16, 16, 6, 12, 11, 3, 6, 11, 9, 9, 10, 15, 6, 6, 11, 11, 9, 4, 7, 18, 8, 11, 4, 8, 6, 7, 38, 13, 9, 6, 3, 15, 20, 20, 11, 8, 48, 5, 14, 10, 12, 8, 7, 22, 51, 11, 12, 20, 31, 11, 25, 8, 35, 11, 15, 9, 24, 20, 11, 16, 20, 13, 11, 24, 12, 24, 10, 12, 8, 14, 20, 10, 6, 11, 7, 17, 17, 7, 10, 8, 7, 6, 41, 6, 56, 8, 8, 5, 10, 6, 25, 8, 7, 14, 5, 10, 21, 12, 17, 9, 15, 18, 22, 14, 10, 11, 15, 6, 9, 11, 8, 12, 29, 11, 7, 7, 12, 19, 4, 6, 11, 5, 6, 11, 6, 9, 12, 22, 16, 9, 4, 8, 13, 14, 21, 16, 7, 21, 7, 12, 7, 12, 11, 5, 9, 11, 23, 65, 20, 7, 11, 22, 8, 6, 37, 17, 6, 5, 9, 13, 17, 27, 29, 12, 8, 8, 21, 8, 44, 11, 7, 12, 9, 14, 8, 9, 6, 13, 18, 13, 16, 9, 9, 15, 10, 8, 34, 6, 10, 37, 8, 7, 13, 8, 15, 9, 12, 12, 18, 35, 4, 4, 15, 6, 20, 6, 13, 9, 23, 13, 9, 6, 44, 12, 5, 4, 20, 8, 8, 13, 20, 8, 18, 8, 8, 9, 15, 6, 9, 15, 14, 12, 11, 6, 10, 12, 7, 11, 19, 11, 5, 9, 10, 13, 17, 15, 9, 10, 8, 7, 16, 15, 8, 21, 9, 11, 10, 16, 12, 24, 6, 6, 9, 7, 10, 5, 8, 50, 17, 15, 30, 14, 19, 6, 19, 11, 15, 6, 9, 7, 13, 8, 5, 9, 9, 4, 9, 14, 6, 38, 32, 46, 5, 18, 9, 13, 7, 7, 11, 9, 12, 14, 6, 9, 11, 19, 27, 7, 48, 6, 26, 11, 5, 11, 22, 5, 7, 7, 21, 6, 15, 3, 12, 23, 16, 8, 9, 20, 7, 4, 5, 14, 16, 10, 36, 8, 10, 47, 18, 6, 16]
[65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]
outs.shape (400, 65, 512)
logits_ shape (400, 2)
y_smot shape (400, 2)
----- Step 3 -- Loss 0.50464 -- acc 1.00000
Training:  12%|█▏        | 3/25 [02:16<16:41, 45.54s/it]nextBatchs.encoder_inputs_length [12, 13, 17, 27, 7, 12, 7, 16, 9, 21, 24, 7, 29, 13, 23, 13, 35, 31, 26, 5, 10, 27, 7, 4, 9, 17, 18, 13, 16, 6, 35, 56, 7, 6, 10, 5, 8, 6, 6, 6, 9, 6, 9, 19, 83, 9, 17, 20, 2, 8, 8, 15, 11, 27, 9, 11, 7, 10, 5, 16, 19, 10, 15, 8, 7, 8, 16, 17, 5, 7, 22, 13, 3, 17, 45, 17, 9, 10, 12, 21, 23, 25, 4, 14, 8, 4, 24, 18, 14, 15, 6, 5, 17, 19, 5, 8, 15, 11, 16, 13, 18, 7, 9, 6, 21, 6, 22, 48, 20, 5, 14, 19, 7, 7, 8, 6, 10, 11, 19, 32, 18, 17, 8, 9, 10, 13, 20, 8, 16, 20, 10, 5, 8, 9, 6, 6, 17, 17, 7, 16, 15, 14, 11, 14, 8, 127, 13, 18, 5, 12, 8, 42, 32, 7, 8, 32, 8, 42, 9, 12, 40, 9, 16, 10, 11, 7, 16, 98, 4, 11, 14, 10, 52, 5, 12, 11, 11, 18, 28, 17, 7, 11, 13, 23, 5, 20, 35, 6, 19, 5, 7, 12, 7, 11, 34, 42, 10, 20, 5, 8, 19, 7, 21, 10, 6, 9, 10, 20, 66, 8, 12, 11, 26, 6, 13, 6, 7, 7, 9, 8, 27, 11, 23, 11, 7, 6, 9, 8, 13, 27, 16, 13, 15, 7, 7, 25, 24, 14, 21, 5, 8, 8, 18, 44, 7, 21, 13, 5, 9, 10, 19, 4, 9, 80, 6, 9, 9, 57, 9, 32, 12, 17, 16, 15, 7, 16, 4, 6, 9, 15, 12, 4, 16, 15, 4, 20, 14, 16, 14, 19, 8, 15, 17, 10, 18, 6, 6, 7, 14, 12, 12, 5, 10, 12, 12, 19, 12, 12, 18, 18, 6, 12, 7, 8, 16, 11, 9, 6, 10, 5, 7, 21, 13, 13, 15, 31, 10, 7, 9, 12, 18, 16, 23, 20, 8, 17, 11, 9, 8, 10, 17, 33, 7, 26, 9, 12, 10, 39, 24, 13, 16, 8, 6, 15, 12, 9, 18, 5, 8, 8, 8, 25, 15, 7, 27, 8, 15, 10, 13, 9, 5, 5, 12, 30, 38, 7, 6, 11, 6, 11, 22, 32, 10, 18, 15, 11, 21, 9, 3, 7, 10, 24, 21, 11, 22, 10, 24, 46, 10, 11, 9, 11, 8, 11, 8, 27, 19, 12, 5, 9]
[127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127]
outs.shape (400, 127, 512)
logits_ shape (400, 2)
y_smot shape (400, 2)

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

'''

