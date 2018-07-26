# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 17:12
# @Author  : Chen Ruida
# @Email   : crd57@outlook.com
# @File    : data.py
# @Software: PyCharm
import os.path
import sys
import tarfile
import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from urllib.request import urlretrieve

data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
bottleneck_tensor_size = 2048
input_width = 224
input_height = 224
input_depth = 3
model_file_name = 'classify_image_graph_def.pb'

TRAINING = 'training'
TESTING = 'testing'
VALIDATION = 'validation'

IMAGE_DIR = "D:/CODE/RSICG/RSICD/RSICD_images"
LABEL_DIR = "label/"
IMAGE_LABEL_DIR = 'D:/CODE/RSICG/RSICD/annotations_rsicd'
ALL_LABELS_FILE = "dataset_rsicd.json"
bottleneck_path = "bottlenecks/"
summaries_dir = 'tmp/retrain_logs'
final_tensor_name = 'final_result'
output_graph = 'tmp/output_graph.pb'
model_dir = 'model_dir'


# CACHED_GROUND_TRUTH_VECTORS = {}
# sess, image_lists, class_count, labels = None, None, 0, []


def maybe_download_and_extract():
    """
    如果模型不存在，将会下载模型
    :return: None
    """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]  # inception-2015-12-05.tgz
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 回调函数，用于查看下载进度
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # urllib模块提供的 urlretrieve()函数。该函数直接将远程数据下载到本地。
        filepath, _ = urlretrieve(data_url, filepath, _progress)

        # 返回文件的系统状态信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_model_graph():
    """
    创建图
    :return: 图，计算瓶颈层结果的张量,输入所对应的张量
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_file_name)

        # 读取训练好的Inception-v3模型
        # 谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值
        # 加载图
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # 加载读取的Inception-v3模型，并返回数据：输入所对应的张量和计算瓶颈层结果所对应的张量。
            bottleneck_tensor, jpeg_data_tensor = (tf.import_graph_def(
                graph_def, name='',
                return_elements=['pool_3/_reshape:0', 'DecodeJpeg/contents:0']
            ))
    return graph, bottleneck_tensor, jpeg_data_tensor


def run_bottleneck_on_image(image_data, image_data_tensor, bottleneck_tensor):
    """
    使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
    :param image_data: 图片数据
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return:
    """
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # print(image_data_tensor.name)
    # [1,1,2048]
    bottleneck_values = np.squeeze(bottleneck_values)  # [2048]
    return bottleneck_values


def create_bottleneck_file(bottleneck_f_path, jpeg_data_tensor, bottleneck_tensor, file_path):
    image_data = gfile.FastGFile(file_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(image_data, jpeg_data_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (file_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_f_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)
    return bottleneck_values


def save_bottleneck_files():
    global sess
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor = create_model_graph()
    with tf.Session(graph=graph) as session:
        for root, dirs, files in os.walk(IMAGE_DIR):
            for f in files:
                sess = session
                file_path = root + '/' + f
                bottleneck_f_path = bottleneck_path + f[:-4] + ".txt"
                if not os.path.exists(bottleneck_f_path):
                    create_bottleneck_file(bottleneck_f_path, jpeg_data_tensor, bottleneck_tensor, file_path)
                    print(bottleneck_f_path + "   finished")
                else:
                    print(bottleneck_f_path + "is existed")


def words2num(f_json):
    words_set = set()
    for i in f_json:
        for j in i["sentences"][0]["tokens"]:
            words_set.add(j)
    num = 0
    words_to_num = {}
    num_to_words = {}
    for i in words_set:
        words_to_num[i] = num
        num_to_words[num] = i
        num += 1
    words_to_num["<EOS>"] = num
    num_to_words[num] = "<EOS>"
    words_to_num["<PAD>"] = num + 1
    num_to_words[num + 1] = "<PAD>"
    words_to_num["<GO>"] = num + 2
    num_to_words[num + 2] = "<GO>"
    words_to_num["<UNK>"] = num + 3
    num_to_words[num + 3] = "<UNK>"
    return words_set, words_to_num, num_to_words


def save_label(f_json, words_to_num):
    for i in f_json:
        with open(LABEL_DIR + i['filename'] + ".txt", 'w') as f:
            source_int = [words_to_num.get(letter, words_to_num["<UNK>"])
                          for letter in i["sentences"][0]["tokens"]] + [words_to_num['<EOS>']]
            for j in source_int:
                f.write(str(j) + " ")
            print(i['filename'] + "   over")


if __name__ == '__main__':
    with open(r"D:\CODE\RSICG\RSICD\annotations_rsicd\dataset_rsicd.json") as f:
        f_j = json.load(f)
    im_json = f_j["images"]
    words_set, words_to_num, num_to_words = words2num(im_json)
    save_label(im_json, words_to_num)
    data_sets, target_letter_to_int, target_int_to_letter = words2num(im_json)
    with open('data.pickle', 'wb') as f:
        pickle.dump([target_int_to_letter, target_letter_to_int, data_sets], f)
