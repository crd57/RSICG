# -*- coding: utf-8 -*-
"""Based on NELSONZHAO's code(https://github.com/NELSONZHAO/zhihu)
   Define the RNN used in this project.
"""

from tensorflow.python.layers.core import Dense
import numpy as np
import tensorflow as tf
import pickle
from data import *

"""All the parameters and hyper parameters for the model"""
# Learning rate
learning_rate = 0.001
# Optimizer used by the model, 0 for SGD, 1 for Adam, 2 for RMSProp
optimizer_type = 1
# Mini-batch size
batch_size = 128
# Cell type, 0 for LSTM, 1 for GRU
Cell_type = 0
# Activation function used by RNN cell, 0 for tanh, 1 for relu, 2 for sigmoid
activation_type = 1
# Number of cells in each layer
rnn_size = 128
# Number of layers
num_layers = 2
# Embedding size for encoding part and decoding part
encoding_embedding_size = 64
decoding_embedding_size = encoding_embedding_size
# Decoder type, 0 for basic, 1 for beam search
Decoder_type = 0
# Beam width for beam search decoder
beam_width = 3
# Number of max epochs for training
epochs = 500
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = 1
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 5

# 从 data.pickle加载数据
with open('./data.pickle', 'rb') as f:
    target_int_to_letter, target_letter_to_int, data_sets = pickle.load(f)


# 模型输入tensor
def get_inputs():
    """Generate the tf.placeholder for the model input.
    Returns:
        inputs: input of the model, tensor of shape [batch_size, max_input_length].
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        learning_rate: learning rate for the mini-batch training.
        target_sequence_length: tensor of shape [mini-batch size, ],the length for
          each target sequence in the mini-batch.
        max_target_sequence_length: the max length of target sequence across the
          mini-batch for training.
        source_sequence_length: tensor of shape [mini-batch size, ],the length for
          each input sequence in the mini-batch.
    """

    inputs = tf.placeholder(tf.float32, [batch_size, 2048], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    target_sequence_length = tf.placeholder(tf.int32, (batch_size,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    # source_sequence_length = tf.placeholder(tf.int32, (batch_size,), name='source_sequence_length')
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length


# 构建多层RNN, 使用 LSTM or GRU
def construct_cell(rnn_size, num_layers):
    """Construct multi-layer RNN
    Args:
        rnn_size: the number of hidden units in a single RNN layer.
        num_layers: the number of total layers of the RNN.
    Returns:
        cell: multi-layer Rnn.
    """

    def get_cell(rnn_size):
        """Generate single RNN layer as specified.
        Args:
            rnn_size: the number of hidden units in a single RNN layer.
        Returns:
            A single layer of RNN
        """
        activation_collection = {0: tf.nn.tanh,
                                 1: tf.nn.relu,
                                 2: tf.nn.sigmoid}
        if Cell_type:
            return tf.contrib.rnn.GRUCell(rnn_size, activation=activation_collection[activation_type])  # GRU
        else:
            return tf.contrib.rnn.LSTMCell(rnn_size, activation=activation_collection[activation_type])  # LSTM

    cell = tf.contrib.rnn.MultiRNNCell([get_cell(rnn_size) for _ in range(num_layers)])  # 层数
    return cell


# 构建encoder 层
def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length):
    """Construct the encoder part.
       Args:
           input_data: input of the model, tensor of shape [batch_size, max_input_length].
           rnn_size: the number of hidden units in a single RNN layer.
           num_layers: total number of layers of the encoder.
           source_sequence_length: tensor of shape [mini-batch size, ],the length for
             each input sequence in the mini-batch.
           source_vocab_size: total number of symbols of input sequence.
           encoding_embedding_size: size of embedding for each symbol in input sequence.
       Returns:
           encoder_output: RNN output tensor.
           encoder_state: The final state of RNN
    """
    # Encoder embedding
    # encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    # with tf.variable_scope("encoder"):
    #     cell = construct_cell(rnn_size, num_layers)
    #     # Performs fully dynamic unrolling of inputs
    #     encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
    #                                                       sequence_length=source_sequence_length, dtype=tf.float32)
    ## shape(encoder_output[None,rnn_size],encoder_state[None,source_sequence_length,rnn_size]
    w_h = tf.Variable(tf.truncated_normal(shape=[2048, 2*num_layers*rnn_size], dtype=tf.float32))
    b_h = tf.Variable(tf.zeros(shape=[2*num_layers*rnn_size], dtype=tf.float32))
    fc = tf.nn.tanh(tf.matmul(input_data, w_h) + b_h, name="encoder_state")
    fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True,
                                      is_training=True)
    encoder_state = tf.nn.relu(fc)

    w_c = tf.Variable(tf.truncated_normal(shape=[2048, source_sequence_length * rnn_size], dtype=tf.float32))
    b_c = tf.Variable(tf.zeros(shape=[source_sequence_length * rnn_size], dtype=tf.float32))
    c = tf.nn.tanh(tf.matmul(input_data, w_c) + b_c)
    encoder_output = tf.reshape(c, [-1, source_sequence_length, rnn_size], name="encoder_outputs")

    return encoder_output, encoder_state


# 构建Decoder层
def decoding_layer(encoder_outputs, target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    """Construct the decoding part of the model.
    See the guide https://www.tensorflow.org/versions/master/api_guides/python/contrib.seq2seq#Dynamic_Decoding
    Args:
        target_letter_to_int: mapping target sequence symbol to int, dict {symbol:int}.
        decoding_embedding_size: target symbol embedding size.
        num_layers: total number of layers of the decoder.
        rnn_size: the number of hidden units in a single RNN layer.
        target_sequence_length: tensor of shape [mini-batch size, ],the length for
          each target sequence in the mini-batch.
        max_target_sequence_length: the max length of target sequence across the
          mini-batch for training.
        encoder_state: the final state of encoder, feeds to decoder as initial state.
        decoder_input: tensor of shape [mini_batch_size, max_target_sequence_length],
          true result for training.
    Returns:
        training_decoder_output: final output of the decoder during training.
        predicting_decoder_output: final output of the decoder during validation.
        bm_decoder_output: final output of the beam search decoder.
    """
    # 1.对目标序列进行Embedding,使得它们能够传入Decoder中的RNN。
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
    # 2. 构造Decoder中的RNN单元
    cell = construct_cell(rnn_size, num_layers)

    # 3. Output全连接层,默认用线性激活函数
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="dense_layer")

    # 4. Training the decoder
    with tf.variable_scope("decoder"):
        # 得到helper对象
        #         # “TrainingHelper”：训练过程中最常使用的Helper，下一时刻输入就是上一时刻target的真实值
        #         # TrainingHelper用于train阶段，next_inputs方法一样也接收outputs与sample_ids，
        #         # 但是只是从初始化时的inputs返回下一时刻的输入。
        # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size, memory=encoder_outputs)
        #
        # decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=attention_mechanism,
        #                                                    attention_layer_size=rnn_size, name='Attention_Wrapper')
        # initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        # initial_state = initial_state.clone(cell_state=encoder_state)

        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # 构建decoder
        # BasicDecoder的作用就是定义一个封装了decoder应该有的功能的实例，根据Helper实例的不同，这个decoder可以实现不同的功能，
        # 比如在train的阶段，不把输出重新作为输入，而在inference阶段，将输出接到输入。
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,
                                                           helper=training_helper,
                                                           initial_state=encoder_state,
                                                           output_layer=output_layer)
        # dynamic_decode：将定义好的decoder实例传入
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)
    # 5. Predicting decoder ， 与training共享参数
    with tf.variable_scope("decoder", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile([tf.constant(target_letter_to_int['<GO>'], dtype=tf.int32)], [batch_size],
                               name='start_tokens')
        # “GreedyEmbeddingHelper”：预测阶段最常使用的Helper，下一时刻输入是上一时刻概率最大的单词通过embedding之后的向量
        # 用于inference阶段的helper，将output输出后的logits使用argmax获得id再经过embedding layer来获取下一时刻的输入。
        # start_tokens：起始
        # target_letter_to_int：结束
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     target_letter_to_int['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            maximum_iterations=max_target_sequence_length)
        # dynamic_decode返回(final_outputs, final_state, final_sequence_lengths)。其中：final_outputs是tf.contrib.seq2seq.BasicDecoderOutput类型，包括两个字段：rnn_output，sample_id
        tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
        # 树搜索
        bm_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, decoder_embeddings, start_tokens,
                                                          target_letter_to_int['<EOS>'], tiled_encoder_state,
                                                          beam_width, output_layer)

        # impute_finished must be set to false when using beam search decoder
        # https://github.com/tensorflow/tensorflow/issues/11598
        bm_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(bm_decoder,
                                                                    maximum_iterations=max_target_sequence_length)
    return training_decoder_output, predicting_decoder_output, bm_decoder_output


# 对target 数据进行预处理，添加<GO>，去除最后一个字符
def process_decoder_input(targets, vocab_to_int):
    """Process the target sequence as input to train the model.
    a. cut the last symbol of the target since it won't be fed to the network (<EOS>, <PAD>).
    b. add <GO> to each sequence.

    Args:
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        vocab_to_int: dict {output_symbol:int}, mapping output symbol to int.
    Returns:
        decoder_input: the already processed target sequence
    """
    ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], axis=1)

    return decoder_input


# 连接Encoder和Decoder,构建seq2seq模型
def seq2seq_model(input_data, targets, target_sequence_length,
                  max_target_sequence_length, source_sequence_length, decoding_embedding_size,
                  rnn_size, num_layers):
    """Construct the seq2seq model by connecting encoder part and decoder part.

    Args:
        input_data: input of the model, tensor of shape [batch_size, max_input_length].
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        target_sequence_length:
        max_target_sequence_length:
        source_sequence_length: tensor of shape [mini-batch size, ],the length for
          each input sequence in the mini-batch.
        source_vocab_size: total number of symbols of input sequence.
        encoder_embedding_size: size of embedding for each symbol in input sequence.
        decoding_embedding_size: size of embedding for each symbol in target sequence.
        rnn_size: the number of hidden units in a single RNN layer.
        num_layers: total number of layers of the encoder.
    Returns:
        training_decoder_output: final output of the decoder during training.
        predicting_decoder_output: final output of the decoder during validation.
        bm_decoder_output: final output of the beam search decoder.
    """
    # 获取encoder的状态输出
    encoder_outputs, encoder_state = get_encoder_layer(input_data,
                                                       rnn_size,
                                                       num_layers,
                                                       source_sequence_length)
    encoder_state = tf.reshape(encoder_state, [num_layers, 2, batch_size, rnn_size], name="encoder_state_reshape")
    l = tf.unstack(encoder_state, axis=0, name="unstack")
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
         for idx in range(num_layers)]
    )


    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets, target_letter_to_int)
    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output, bm_decoder_output = decoding_layer(encoder_outputs,
                                                                                           target_letter_to_int,
                                                                                           decoding_embedding_size,
                                                                                           num_layers,
                                                                                           rnn_size,
                                                                                           target_sequence_length,
                                                                                           max_target_sequence_length,
                                                                                           rnn_tuple_state,
                                                                                           decoder_input)

    return training_decoder_output, predicting_decoder_output, bm_decoder_output
