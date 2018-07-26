"""Train or test the model"""

from model import *
import six

# python 2 和3的兼容
input = six.moves.input


train_source = None
valid_source = None
test_source = None
def create_image_lists():
    testing_percentage = validation_percentage = 0.1
    file_list = os.listdir(IMAGE_DIR)

    testing_count = int(len(file_list) * testing_percentage)
    validation_count = int(len(file_list) * validation_percentage)
    return {
        TESTING: file_list[:testing_count],
        VALIDATION: file_list[testing_count:(testing_count + validation_count)],
        TRAINING: file_list[(testing_count + validation_count):]
    }


data_sets = create_image_lists()

if isTrain == 1:
    # training the model, make the input and output's size be multiple of mini batch's size.
    train_remainder = len(data_sets[TRAINING]) % batch_size
    valid_remainder = len(data_sets[VALIDATION]) % batch_size
    # 保证是batch的整数倍
    train_source = data_sets[TRAINING] + data_sets[TRAINING][0:batch_size - train_remainder]
    # 不够补上
    valid_source = data_sets[VALIDATION] + data_sets[VALIDATION][0:batch_size - valid_remainder]

elif isTrain == 2:
    test_remainder = len(data_sets[TESTING]) % batch_size
    test_source = data_sets[TESTING] + data_sets[TESTING][0:batch_size - test_remainder]


# 对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    # print([type(sentence) for sentence in sentence_batch])
    return [list(sentence) + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# 定义生成器，用来获取batch
def get_batches(sources, target_pad_int):
    """Generator to generating the mini batches for training and testing
    Args:
        sources: data file name list.
        target_pad_int: an integer representing the symbol of <PAD> for output sequence.
    Yields:
        pad_targets_batch: padded targets mini-batch
        targets_length: tensor of shape (mini_batch_size, ), representing the length for
          each target sequence in the mini-batch.
    """
    for batch_i in range(0, len(sources) // batch_size):
        sources_batch = []
        targets_batch = []
        start_i = batch_i * batch_size
        sources_batch_list = sources[start_i:start_i + batch_size]
        for i in sources_batch_list:
            new_data_sources = np.loadtxt(bottleneck_path + i[:-4] + ".txt", delimiter=',')
            sources_batch.append(new_data_sources)
            new_data_targets = np.loadtxt(LABEL_DIR + i + ".txt")
            targets_batch.append(new_data_targets)
        # 补全序列
        pad_sources_batch = sources_batch
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


#  创建计算图
train_graph = tf.Graph()
with train_graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(train_graph)
    # 获得模型输入
    input_data, targets, lr, target_sequence_length, max_target_sequence_length = get_inputs()

    # define the placeholder and summary of the validation loss and WER
    average_vali_loss = tf.placeholder(dtype=tf.float32, name="average_vali_loss")
    # 错误率
    v_c = tf.summary.scalar("validation_cost", average_vali_loss)

    # 获得seq2seq模型的输出
    training_decoder_output, predicting_decoder_output, bm_decoder_output = seq2seq_model(input_data,  # 输入数据
                                                                                          targets,
                                                                                          target_sequence_length,
                                                                                          max_target_sequence_length,
                                                                                          encoding_embedding_size,
                                                                                          decoding_embedding_size,
                                                                                          rnn_size,
                                                                                          num_layers)

    # tf.identity 返回一个Tensor  参考：https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
    training_logits = tf.identity(training_decoder_output.rnn_output, name='training_logits')
    # 转化成Tensor 训练得分
    predicting_logits = tf.identity(predicting_decoder_output.rnn_output, name='predicting_logits')
    # 预测得分
    # the result of the prediction
    prediction = tf.identity(predicting_decoder_output.sample_id, 'prediction_result')
    # 预测得到的sample_id
    bm_prediction = tf.identity(bm_decoder_output.predicted_ids, 'bm_prediction_result')
    # 树搜索得到的sample_id
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    # 将补全的值进行掩模不参与梯度下降
    # the score of the beam search prediction
    bm_score = tf.identity(bm_decoder_output.beam_search_decoder_output.scores, 'bm_prediction_scores')
    # 树搜索所得到得结果的得分
    with tf.name_scope("optimization"):
        # Loss function, compute the cross entropy
        train_cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,  # 输出层的结果
            targets,  # 目标值
            masks)  #

        optimizer_collection = {0: tf.train.GradientDescentOptimizer(lr),
                                1: tf.train.AdamOptimizer(lr),
                                2: tf.train.RMSPropOptimizer(lr)}
        # Using the optimizer defined by optimizer_type
        optimizer = optimizer_collection[optimizer_type]

        # compute gradient
        gradients = optimizer.compute_gradients(train_cost)
        # apply gradient clipping to prevent gradient explosion
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
        # 梯度裁剪
        # update the RNN
        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)
    # define summary to record training cost.
    training_cost_summary = tf.summary.scalar('training_cost', train_cost)

    with tf.name_scope("validation"):
        # get the max length of the predicting result
        val_seq_len = tf.shape(predicting_logits)[1]  # [batchsize,序列长度,embeding]
        # process the predicting result so that it has the same shape with targets
        predicting_logits = tf.concat([predicting_logits, tf.fill(
            [batch_size, max_target_sequence_length - val_seq_len, tf.shape(predicting_logits)[2]], 0.0)], axis=1)
        # calculate loss
        validation_cost = tf.contrib.seq2seq.sequence_loss(
            predicting_logits,
            targets,
            masks)
    with tf.name_scope("cal_edit_distance"):
        valid_targets_batch_sparse = tf.contrib.layers.dense_to_sparse(targets)
        basic_prediction_sparse = tf.contrib.layers.dense_to_sparse(prediction)
        error = tf.reduce_mean(
            tf.edit_distance(basic_prediction_sparse, tf.to_int32(valid_targets_batch_sparse), normalize=False))
        error_summary = tf.summary.scalar('edit_distance', error)

# create session to run the TensorFlow operations
with tf.Session(graph=train_graph) as sess:
    # define summary file writer
    t_s = tf.summary.FileWriter('./graph/training', sess.graph)
    v_s = tf.summary.FileWriter('./graph/validation', sess.graph)

    # define saver, keep max_model_number of most recent models
    saver = tf.train.Saver(max_to_keep=max_model_number)

    if isTrain == 1:
        # run initializer
        sess.run(tf.global_variables_initializer())

        # train the model
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_source,
                                target_letter_to_int['<PAD>'])):
                # get global step
                step = tf.train.global_step(sess, global_step)
                t_c, _, loss, tr_e, tr_e_s = sess.run(
                    [training_cost_summary, train_op, train_cost, error, error_summary],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths})

                if batch_i % display_step == 0:
                    v_e = 0.0
                    # calculate the word error rate (WER) and validation loss of the model
                    vali_loss = []
                    for (valid_targets_batch,
                         valid_sources_batch,
                         valid_targets_lengths,
                         valid_source_lengths) in get_batches(
                        valid_source,
                        target_letter_to_int['<PAD>']):
                        validation_loss, basic_prediction, v_e, v_e_s = sess.run(
                            [validation_cost, prediction, error, error_summary],
                            {input_data: valid_sources_batch,
                             targets: valid_targets_batch,
                             lr: learning_rate,
                             target_sequence_length: valid_targets_lengths})

                        vali_loss.append(validation_loss)
                        # v_s.add_summary(e_v_m,global_step=step)

                    # calculate the average validation cost and the WER over the validation data set
                    vali_loss = sum(vali_loss) / len(vali_loss)
                    # WER = error
                    vali_summary = sess.run(v_c, {average_vali_loss: vali_loss})
                    # write the cost to summery
                    t_s.add_summary(t_c, global_step=step)
                    t_s.add_summary(tr_e_s, global_step=step)
                    v_s.add_summary(vali_summary, global_step=step)
                    v_s.add_summary(v_e_s, global_step=step)

                    print(
                        'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  '
                        '- Validation loss: {:>6.3f} - Validation error: {} '
                            .format(epoch_i,
                                    epochs, batch_i,
                                    len(train_source) // batch_size,
                                    loss,
                                    vali_loss,
                                    v_e))

            # save the model every epoch
            saver.save(sess, save_path='./ckpt/model.ckpt', global_step=step)
        # save the model when finished
        saver.save(sess, save_path='./ckpt/model.ckpt', global_step=step)
        print('Model Trained and Saved')

    else:
        checkpoint = tf.train.latest_checkpoint('./ckpt')
        saver.restore(sess, checkpoint)

        # use the trained model to perform pronunciation prediction
        if isTrain == 0:
            while True:
                test_input = input(">>")
                converted_input = np.loadtxt(bottleneck_path + test_input + ".txt", delimiter=',')
                # if the decoder type is 0, use the basic decoder, same as set beam width to 0
                if Decoder_type == 0:
                    beam_width = 1
                result = sess.run(
                    [bm_prediction, bm_score, prediction],
                    {input_data: [converted_input] * batch_size,
                     target_sequence_length: [len(converted_input)] * batch_size,
                     })
                print("result:")
                for i in range(beam_width):
                    tmp = []
                    flag = 0
                    for idx in result[0][0, :, i]:
                        tmp.append(target_int_to_letter[idx])
                        if idx == target_letter_to_int['<EOS>']:
                            print(' '.join(tmp))
                            flag = 1
                            break
                    # prediction length exceeds the max length
                    if not flag:
                        print(' '.join(tmp))

                    # print the score of the result
                    print('score: {0:.4f}'.format(result[1][0, :, i][-1]))
                    print('')
        # evaluate the model's performance
        else:
            WER = 0.0
            test_loss = []
            for _, (
                    test_targets_batch, test_sources_batch, test_targets_lengths,
                    test_source_lengths) in enumerate(
                get_batches(test_source,
                            target_letter_to_int['<PAD>'])
            ):
                validation_loss, basic_prediction, WER = sess.run(
                    [validation_cost, prediction, error],
                    {input_data: test_sources_batch,
                     targets: test_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: test_targets_lengths})

                test_loss.append(validation_loss)

            # calculate the average validation cost and the WER over the validation data set
            test_loss = sum(test_loss) / len(test_loss)
            print('Test loss: {:>6.3f}'
                  ' - WER: {:>6.2%} '.format(test_loss, WER))
        # load model from folder
