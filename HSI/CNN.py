from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import  parameter
import tensorflow as tf

NUM_CLASSES = parameter.OUTPUT_CLASSES  #类数
BANDS = parameter.BANDS
IMAGE_SIZE = parameter.patch_size
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE *BANDS

KERNEL_SIZE = parameter.KERNEL_SIZE


def inference(images, conv1_channels, conv2_channels, fc1_units, fc2_units):
    """
    Args:
    images: Images placeholder, from inputs().
    conv1_channels: 第一个卷积层的filter个数.
    conv2_channels: 第二个卷积层的filter个数.
    fc1_units = 第一个全连接隐藏层的单元数.
    fc2_units = 第二个全连接隐藏层的单元数.

    Returns:
    softmax_linear: 经过logits层之后的输出结果张量.
    """

    # Conv 1
    with tf.name_scope('conv_1') as scope:
        weights = tf.get_variable('weights', shape=[5, 5, BANDS, conv1_channels],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv1_channels], initializer=tf.constant_initializer(0.05))

        # converting the 1D array into a 3D image
        x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, BANDS])
        z = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv1 = tf.nn.relu(z + biases)

    # Maxpool 1
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')

    # Conv2
    with tf.variable_scope('h_conv2') as scope:
        weights = tf.get_variable('weights', shape=[5, 5, conv1_channels, conv2_channels],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv2_channels], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_pool1, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv2 = tf.nn.relu(z + biases)

    # Maxpool 2
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')

    # FIXED in python file
    # size_after_conv_and_pool_twice = 4
    size_after_conv_and_pool_twice = int(
        math.ceil((math.ceil(float(IMAGE_SIZE - KERNEL_SIZE + 1) / 2) - KERNEL_SIZE + 1) / 2))

    # Reshape from 4D to 2D
    h_pool2_flat = tf.reshape(h_pool2, [-1, (size_after_conv_and_pool_twice ** 2) * conv2_channels])

    # FC 1
    with tf.name_scope('h_FC1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([((size_after_conv_and_pool_twice ** 2) * conv2_channels), fc1_units],
                                stddev=1.0 / math.sqrt(float(size_after_conv_and_pool_twice))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc1_units]),
                             name='biases')
        h_FC1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)

    # FC 2
    with tf.name_scope('h_FC2'):
        weights = tf.Variable(
            tf.truncated_normal([fc1_units, fc2_units],
                                stddev=1.0 / math.sqrt(float(fc1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc2_units]),
                             name='biases')
        h_FC2 = tf.nn.relu(tf.matmul(h_FC1, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([fc2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(h_FC2, weights) + biases

    return logits

'''
定义损失函数
'''
def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))