# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import spectral
import matplotlib.pyplot as plt
import pylab as pl
import scipy
import CNN
import Spatial_dataset as input_data
import parameter
import os
import matplotlib
import scipy.io as io
import pickle as pkl
'''
Load data
'''

DATA_PATH = os.path.join(os.getcwd(),"Data")
FEATURE_PATH = os.path.join(os.getcwd(),"Feature")
# input_image1 = scipy.io.loadmat(os.path.join(DATA_PATH, 'DATA_DSM'))
input_image = scipy.io.loadmat(os.path.join(DATA_PATH, 'DATA_DSM'))['DSM_300']
output_image = scipy.io.loadmat(os.path.join(DATA_PATH, 'DATA_DSM'))['GT_300']

# input_image = np.rot90(input_image)
# output_image = np.rot90(output_image)
input_image=input_image[:,:,np.newaxis]
# input_image = np.rot90(input_image)
# output_image = np.rot90(output_image)
height = output_image.shape[0]
width = output_image.shape[1]
PATCH_SIZE = parameter.patch_size
batch_size = parameter.batch_size
num_examples = parameter.num_examples
conv0=parameter.conv0
conv1 = parameter.conv1
conv2 = parameter.conv2
fc1 = parameter.fc1
fc2 = parameter.fc2
'''
归一化
'''
input_image = input_image.astype(float)
input_image -= np.min(input_image)
input_image /= np.max(input_image)

'''
去均值
'''
def mean_array(data):
    mean_arr = []
    for i in range(data.shape[0]):
        mean_arr.append(np.mean(data[i,:,:]))
    return np.array(mean_arr)

def Patch(data,height_index,width_index):
    transpose_array = data.transpose((2,0,1))
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean = mean_array(transpose_array)
    mean_patch = []
    for i in range(patch.shape[0]):
        mean_patch.append(patch[i] - mean[i])
    mean_patch = np.asarray(mean_patch)
    patch = mean_patch.transpose((1,2,0))
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    return patch

def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, CNN
                                                           .IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):

    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):

    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def decoder():
    # data_sets = input_data.read_data_sets('Test1.mat','test')

    with tf.Session() as sess:

        images_placeholder, labels_placeholder = placeholder_inputs(1)

        logits = CNN.inference(images_placeholder,conv1,conv2,conv0,fc1,fc2)

        eval_correct = CNN.evaluation(logits, labels_placeholder)
        sm = tf.nn.softmax(logits)
        # sess = tf.Session()
        # saver = tf.train.import_meta_graph('mysaver/-499.meta')    saver.restore(sess, 'mysaver/-499')
        check_point_path = 'Weights/'
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


        temp = []

        outputs = np.zeros((height, width))
        predicted_results = [[0 for i in range(width)] for x in range(height)]
        for i in range(height - PATCH_SIZE + 1):
            for j in range(width - PATCH_SIZE + 1):
                target = int(output_image[i +int(PATCH_SIZE / 2) , j +int(PATCH_SIZE / 2) ])
                if target == 0:
                    continue
                else:
                    image_patch = Patch(input_image, i, j)

                    # 获取pooling层特征
                    feature = sess.graph.get_operation_by_name("h_pool2_flat").outputs[0]

                    file_name = 'DSM_' + str(PATCH_SIZE) + '_'+str(i)+'_' + str(j) + '.mat'
                    DSM_feature = {}
                    scipy.io.savemat(os.path.join(FEATURE_PATH, file_name), DSM_feature)

                    # print image_patch
                    prediction = sess.run(sm, feed_dict={images_placeholder: image_patch})



                    # print prediction
                    temp1 = np.argmax(prediction) + 1
                    # print temp1
                    outputs[i + int(PATCH_SIZE / 2)][j + int(PATCH_SIZE / 2)] = temp1
                    predicted_results[i + int(PATCH_SIZE / 2)][j + int(PATCH_SIZE / 2)] = prediction

    return outputs, predicted_results

predicted_image,predicted_results = decoder()

