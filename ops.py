from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import os

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def fc(input_vector, num_output_length, name='fc'):
    with tf.variable_scope(name):
        #print(input_vector)
        stddev = np.sqrt(1.0 / (np.sqrt(input_vector.get_shape()[-1].value * num_output_length)))
        w = tf.get_variable(
            name='w',
            shape=[input_vector.get_shape()[1], num_output_length],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            name='b',
            shape=[num_output_length],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input_vector, w) + b

def concat_label(x, labels_one_hot, batch_size, duplicate=1):
    x_shape = x.get_shape().as_list()
    #print('x_shape', x_shape)
    labels_one_hot = tf.tile(labels_one_hot, [1, duplicate])
    labels_one_hot_shape = labels_one_hot.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat([x, labels_one_hot], axis=1)
    elif len(x_shape) == 4:
        labels_one_hot = tf.reshape(labels_one_hot, [batch_size, 1, 1, labels_one_hot_shape[-1]])
        return tf.concat([x, labels_one_hot*tf.ones([batch_size, x_shape[1], x_shape[2], labels_one_hot_shape[-1]])], axis=3)

def get_dataset(img_dir, list_file):
    """return a list that each row contain a image_path, id and age label"""
    dataset = []
    with open(list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            split = line.split(' ')
            img_name = split[0]
            id = split[1]
            age = int(split[2])
            if 16 <= age <= 20:
                age = 0
            elif 21 <= age <= 30:
                age = 1
            elif 31 <= age <= 40:
                age = 2
            elif 41 <= age <= 50:
                age = 3
            elif 51 <= age <= 60:
                age = 4
            elif 61 <= age <= 70:
                age = 5
            else:
                age = 6
            img_path = os.path.join(img_dir, img_name)
            dataset.append(img_path + ' ' + id + ' ' + str(age))
        #print(dataset)
    return dataset

def transform(image, flip=True, img_size=256):
    
    with tf.name_scope("preprocess"):
        #[0, 1]->[-1, 1]
        r = image * 2 - 1

        if flip:
            r = tf.image.random_flip_left_right(r)
        
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        r = tf.image.resize_images(r, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return r

        